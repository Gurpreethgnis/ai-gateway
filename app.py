# app.py
import os
import json
import hashlib
import traceback
import asyncio
import time
import logging
import re
from typing import Any, Optional, List, Dict, Literal, Union, Tuple
from fastapi.responses import JSONResponse, StreamingResponse

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from anthropic import Anthropic

# Optional Redis cache
try:
    import redis
except Exception:
    redis = None

# =====================================================
# LOGGING
# =====================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("gateway")

app = FastAPI()

# =====================================================
# VALIDATION ERROR LOGGING
# =====================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    ray = request.headers.get("cf-ray") or ""
    ua = request.headers.get("user-agent") or ""
    log.error(
        "VALIDATION ERROR %s %s -> 422 cf-ray=%s ua=%s errors=%s",
        request.method,
        request.url.path,
        ray,
        ua[:120],
        exc.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# =====================================================
# SECURITY CONFIG
# =====================================================

GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY")
if not GATEWAY_API_KEY:
    raise RuntimeError("GATEWAY_API_KEY is not set. Set it in Railway env vars.")

# Cloudflare Transform Rule injects:
#   X-Origin-Secret: <secret>
ORIGIN_SECRET = os.getenv("ORIGIN_SECRET")  # set in Railway (recommended)

# Optional extra enforcement at app layer (usually handled by CF Access itself)
REQUIRE_CF_ACCESS_HEADERS = os.getenv("REQUIRE_CF_ACCESS_HEADERS", "0") == "1"

# =====================================================
# APP CONFIG
# =====================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "1800"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-0")
OPUS_MODEL = os.getenv("OPUS_MODEL", "claude-opus-4-5")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1200"))
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "250000"))

# Hard timeout for upstream model call (prevents hangs -> CF 502)
UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "30"))

# Model proof prefix (what you want to see inside clients)
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "MYMODEL:")

# --- Token reduction knobs (80/20 first) ---
# Drop known giant IDE boilerplate blocks
STRIP_IDE_BOILERPLATE = os.getenv("STRIP_IDE_BOILERPLATE", "1") == "1"
# Truncate huge tool results sent back to model
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "20000"))
# Truncate huge user messages (rare but happens with repo dumps)
USER_MSG_MAX_CHARS = int(os.getenv("USER_MSG_MAX_CHARS", "120000"))
# Truncate huge system/developer blocks (Continue/Cursor repeats these)
SYSTEM_MAX_CHARS = int(os.getenv("SYSTEM_MAX_CHARS", "40000"))
# Inject “diff-first” rules into system prompt
ENFORCE_DIFF_FIRST = os.getenv("ENFORCE_DIFF_FIRST", "1") == "1"

# Caching modes
ENABLE_PREFIX_CACHE = os.getenv("ENABLE_PREFIX_CACHE", "1") == "1"
PREFIX_CACHE_TTL_SECONDS = int(os.getenv("PREFIX_CACHE_TTL_SECONDS", str(CACHE_TTL_SECONDS)))
ENABLE_TOOL_RESULT_DEDUP = os.getenv("ENABLE_TOOL_RESULT_DEDUP", "1") == "1"

client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

rds = None
if REDIS_URL and redis is not None:
    try:
        rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        rds.ping()
    except Exception as e:
        log.warning("Redis disabled: %r", e)
        rds = None

# =====================================================
# ERROR HARDENING (WITH LOGGING)
# =====================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    ray = request.headers.get("cf-ray") or ""
    log.error("UNHANDLED EXCEPTION (cf-ray=%s): %r", ray, exc)
    log.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal error"})

# -----------------------------------------------------
# Root: sanity checks
# -----------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "ai-gateway"}

# =====================================================
# REQUEST LOGGING
# =====================================================

@app.middleware("http")
async def log_404_middleware(request: Request, call_next):
    resp = await call_next(request)
    if resp.status_code == 404:
        ray = request.headers.get("cf-ray") or ""
        ua = request.headers.get("user-agent") or ""
        log.warning("404 %s %s cf-ray=%s ua=%s", request.method, request.url.path, ray, ua[:120])
    return resp

@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    dt_ms = int((time.time() - t0) * 1000)
    ray = request.headers.get("cf-ray") or ""
    ua = request.headers.get("user-agent") or ""
    log.info(
        "REQ %s %s -> %s (%sms) cf-ray=%s ua=%s",
        request.method, request.url.path, resp.status_code, dt_ms, ray, ua[:120]
    )
    return resp

# =====================================================
# AUTH HELPERS
# =====================================================

def extract_gateway_api_key(request: Request) -> Optional[str]:
    """
    Accept either:
      - X-API-Key: <key>
      - api-key: <key>            (some OpenAI-compatible clients)
      - Authorization: Bearer <key>
    """
    for hdr in ("x-api-key", "api-key"):
        v = request.headers.get(hdr)
        if v:
            return v.strip() or None

    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip() or None

    return None

def is_public_path(path: str) -> bool:
    """
    Allow Cloudflare/cdn-cgi/browser-check assets and other non-API paths.
    """
    if path in ("/", "/favicon.ico"):
        return True
    if path.startswith("/js/"):
        return True
    if path.startswith("/cdn-cgi/"):
        return True
    return False

def is_protected_api_path(path: str) -> bool:
    """
    Only enforce API auth on the actual endpoints you care about.
    """
    return path.startswith("/v1/") or path in (
        "/chat",
        "/health",
        "/debug/origin",
        "/debug/headers",
    )

# =====================================================
# MIDDLEWARE: SECURITY (ONLY FOR API PATHS)
# =====================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    try:
        path = request.url.path

        # Let CF/browser-check/static stuff through without API auth
        if is_public_path(path):
            return await call_next(request)

        # If it's not one of our API endpoints, don't require auth
        if not is_protected_api_path(path):
            return await call_next(request)

        # 1) Origin lockdown (Cloudflare → Railway only)
        if ORIGIN_SECRET:
            got = request.headers.get("x-origin-secret")
            if got != ORIGIN_SECRET:
                log.warning("BLOCK origin secret missing/mismatch path=%s host=%s", path, request.headers.get("host"))
                raise HTTPException(status_code=403, detail="Forbidden")

        # 2) Optional CF Access header enforcement (usually handled by CF Access itself)
        if REQUIRE_CF_ACCESS_HEADERS:
            if not request.headers.get("cf-access-client-id") or not request.headers.get("cf-access-client-secret"):
                log.warning("BLOCK missing CF Access headers path=%s host=%s", path, request.headers.get("host"))
                raise HTTPException(status_code=403, detail="Missing Cloudflare Access headers")

        # 3) API key (client auth)
        api_key = extract_gateway_api_key(request)
        if not api_key:
            present = ",".join(
                [k for k in request.headers.keys() if k.lower() in ("authorization", "x-api-key", "api-key")]
            )
            log.warning("BLOCK missing api key path=%s host=%s present_headers=%s", path, request.headers.get("host"), present)
            raise HTTPException(status_code=401, detail="API key required (X-API-Key/api-key/Authorization: Bearer)")

        if api_key != GATEWAY_API_KEY:
            log.warning("BLOCK invalid api key path=%s host=%s", path, request.headers.get("host"))
            raise HTTPException(status_code=403, detail="Invalid API key")

        return await call_next(request)

    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# =====================================================
# MODELS (YOUR /chat)
# =====================================================

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatReq(BaseModel):
    system: Optional[str] = ""
    messages: List[ChatMessage]
    max_tokens: int = DEFAULT_MAX_TOKENS
    model: Optional[str] = None
    temperature: Optional[float] = 0.2

# =====================================================
# MODELS (OPENAI COMPAT)
# =====================================================

OpenAIRole = Literal["system", "user", "assistant", "tool", "developer"]

# Clients sometimes send content as a string, sometimes as structured parts.
OAContent = Union[str, List[Any], Dict[str, Any], None]

class OAChatMessage(BaseModel):
    role: OpenAIRole
    content: OAContent = None

    class Config:
        extra = "allow"

class OAChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[OAChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False

    class Config:
        extra = "allow"

# =====================================================
# HELPERS
# =====================================================

def is_hard_task(text: str) -> bool:
    t = text.lower()
    return any(
        k in t for k in [
            "architecture", "design doc", "production", "incident",
            "migration", "refactor", "security", "performance",
            "kubernetes", "terraform", "postmortem",
        ]
    )

def strip_model_prefix(m: Optional[str]) -> Optional[str]:
    """
    Some clients may send back whatever they see in /v1/models.
    If you prefix model IDs (MYMODEL:...), strip it before routing.
    """
    if not m:
        return None
    s = m.strip()
    low = s.lower()

    if low.startswith("mymodel:"):
        return s[len("MYMODEL:"):].strip()
    if low.startswith("mymodel-"):
        return s[len("MYMODEL-"):].strip()
    return s

def with_model_prefix(m: str) -> str:
    """
    Add your proof prefix, but never double-prefix.
    """
    base = strip_model_prefix(m) or ""
    return f"{MODEL_PREFIX}{base}"

def map_model_alias(maybe: Optional[str]) -> Optional[str]:
    """
    Map friendly aliases (OpenAI-ish model strings) into Anthropic model IDs.
    If unknown, return None (let routing decide).
    """
    if not maybe:
        return None

    maybe = strip_model_prefix(maybe)
    if not maybe:
        return None

    m = maybe.strip().lower()

    # friendly aliases you expose via /v1/models
    if m in ("sonnet", "sonnet-4", "sonnet4", "claude-sonnet", "claude-sonnet-4"):
        return DEFAULT_MODEL
    if m in ("opus", "opus-4", "opus4", "claude-opus", "claude-opus-4"):
        return OPUS_MODEL

    # already an Anthropic model id
    if m.startswith("claude-"):
        return maybe

    # unknown (e.g., gpt-4o) -> ignore, let heuristic route
    return None

def route_model_from_messages(user_text: str, explicit_model: Optional[str]) -> str:
    mapped = map_model_alias(explicit_model)
    if mapped:
        return mapped
    return OPUS_MODEL if is_hard_task(user_text[:8000]) else DEFAULT_MODEL

def normalize_openai_content_to_text(content: OAContent) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)

def cache_key(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()

def cache_get(key: str):
    if rds is None:
        return None
    try:
        v = rds.get(key)
        return json.loads(v) if v else None
    except Exception:
        return None

def cache_set(key: str, data: Dict[str, Any], ttl: int):
    if rds is None:
        return
    try:
        rds.setex(key, ttl, json.dumps(data))
    except Exception:
        pass

def _rds_get_str(key: str) -> Optional[str]:
    if rds is None:
        return None
    try:
        return rds.get(key)
    except Exception:
        return None

def _rds_set_str(key: str, val: str, ttl: int):
    if rds is None:
        return
    try:
        rds.setex(key, ttl, val)
    except Exception:
        return

def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

async def call_anthropic_with_timeout(payload: Dict[str, Any]):
    """
    Anthropic SDK call is synchronous. Run it in a thread + enforce timeout.
    """
    if client is None:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    async def _run():
        return await asyncio.to_thread(lambda: client.messages.create(**payload))

    try:
        return await asyncio.wait_for(_run(), timeout=UPSTREAM_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Upstream model timed out")

def extract_text_from_anthropic(resp) -> str:
    return "".join(
        block.text for block in getattr(resp, "content", [])
        if getattr(block, "type", None) == "text"
    )

def extract_usage(resp) -> Optional[Dict[str, Any]]:
    usage_obj = getattr(resp, "usage", None)
    if usage_obj is None:
        return None
    try:
        return usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj
    except Exception:
        return None

def anthropic_to_openai_usage(usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not usage:
        return None
    try:
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    except Exception:
        return None

# =====================================================
# TOOL-CALL TRANSLATION (OpenAI <-> Anthropic)
# =====================================================

def _oa_tools_from_body(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports:
      - OpenAI "tools": [{"type":"function","function": {...}}]
      - Legacy "functions": [{name, description, parameters}]
    Returns a normalized list like OpenAI tools.
    """
    tools = []
    if isinstance(parsed.get("tools"), list):
        tools = parsed.get("tools") or []
    elif isinstance(parsed.get("functions"), list):
        # legacy -> tools
        for fn in parsed.get("functions") or []:
            if isinstance(fn, dict) and fn.get("name"):
                tools.append({"type": "function", "function": fn})
    return [t for t in tools if isinstance(t, dict)]

def _anthropic_tools_from_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Anthropic expects:
      [{"name": "...", "description": "...", "input_schema": {...}}]
    """
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        desc = fn.get("description") or ""
        params = fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        out.append({"name": name, "description": desc, "input_schema": params})
    return out

def _anthropic_tool_choice_from_openai(tool_choice: Any) -> Optional[Dict[str, Any]]:
    """
    OpenAI tool_choice:
      - "auto" | "none" | {"type":"function","function":{"name":"x"}}
    Anthropic tool_choice:
      - {"type":"auto"} | {"type":"none"} | {"type":"tool","name":"x"}
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        low = tool_choice.lower().strip()
        if low in ("auto",):
            return {"type": "auto"}
        if low in ("none", "no", "disabled"):
            return {"type": "none"}
        # anything else -> auto
        return {"type": "auto"}
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn = (tool_choice.get("function") or {})
            name = fn.get("name")
            if name:
                return {"type": "tool", "name": name}
        # unknown -> auto
        return {"type": "auto"}
    return None

def _ensure_json_args_str(args: Any) -> str:
    """
    OpenAI expects function.arguments as a JSON string.
    """
    if args is None:
        return "{}"
    if isinstance(args, str):
        # It might already be JSON; keep as-is
        return args
    try:
        return json.dumps(args, ensure_ascii=False)
    except Exception:
        return "{}"

def _parse_json_maybe(s: Any) -> Any:
    if s is None:
        return {}
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        ss = s.strip()
        if not ss:
            return {}
        try:
            return json.loads(ss)
        except Exception:
            return {"_raw": s}
    return {"_raw": s}

def _anthropic_message_content_for_tool_result(tool_call_id: str, tool_text: str) -> List[Dict[str, Any]]:
    """
    Tool results in Anthropic go inside a user-role message with:
      [{"type":"tool_result","tool_use_id":"...","content":[{"type":"text","text":"..."}]}]
    """
    return [{
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": [{"type": "text", "text": tool_text}],
    }]

# =====================================================
# TOKEN REDUCTION (cheap but high impact)
# =====================================================

# Heuristic patterns for Continue/Cursor “giant repeated system” blocks.
# Keep this strict enough to avoid deleting real user content.
_BOILERPLATE_PATTERNS: List[re.Pattern] = [
    re.compile(r"you are continue\b", re.IGNORECASE),
    re.compile(r"continue(?:\.| )?ai\b", re.IGNORECASE),
    re.compile(r"cursor\b.*(agent|instructions)", re.IGNORECASE),
    re.compile(r"tool(?:ing)? instructions", re.IGNORECASE),
    re.compile(r"##\s*tools?\b", re.IGNORECASE),
    re.compile(r"^<\|im_start\|>", re.IGNORECASE),
]

_DIFF_FIRST_RULES = """\
DIFF-FIRST EDITING POLICY:
- When modifying files, respond with unified diffs (git-style) unless user explicitly asks for full file.
- Prefer minimal patches touching the smallest region.
- If you need file context, ask via tool calls for specific files/lines instead of requesting the whole repo.
- Never paste entire large files unless requested; output a patch + brief rationale.
"""

def _looks_like_ide_boilerplate(s: str) -> bool:
    if not s:
        return False
    # Only trigger when it's big (Continue blocks are usually huge)
    if len(s) < 1500:
        return False
    hits = sum(1 for p in _BOILERPLATE_PATTERNS if p.search(s or ""))
    return hits >= 2

def _strip_or_truncate(role: str, text: str, max_chars: int, allow_strip: bool) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (new_text, meta)
    """
    meta: Dict[str, Any] = {"stripped": False, "truncated": False, "before": len(text or ""), "after": len(text or "")}
    if not text:
        return "", meta

    if allow_strip and STRIP_IDE_BOILERPLATE and _looks_like_ide_boilerplate(text):
        meta["stripped"] = True
        text = ""  # drop it
        meta["after"] = 0
        return text, meta

    if max_chars > 0 and len(text) > max_chars:
        meta["truncated"] = True
        # head+tail keeps stack traces / logs useful
        head = text[: int(max_chars * 0.7)]
        tail = text[-int(max_chars * 0.3):]
        text = head + "\n\n[...TRUNCATED...]\n\n" + tail
        meta["after"] = len(text)
        return text, meta

    meta["after"] = len(text)
    return text, meta

def _prefix_cache_key(system_text: str) -> str:
    return "prefix:" + _sha256_text(system_text)

def _maybe_prefix_cache(system_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    If system text is large and stable, cache it and replace with a short reference.
    (This is a gateway-side token saver when IDE repeats the same giant block.)
    """
    meta = {"prefix_cached": False, "prefix_hit": False, "key": None}
    if not ENABLE_PREFIX_CACHE or rds is None:
        return system_text, meta
    if not system_text or len(system_text) < 2000:
        return system_text, meta

    key = _prefix_cache_key(system_text)
    meta["key"] = key

    existing = _rds_get_str(key)
    if existing is not None:
        meta["prefix_hit"] = True
        # Replace with stable short pointer + minimal rules
        return f"[CACHED_SYSTEM_PREFIX:{key}]", meta

    _rds_set_str(key, system_text, PREFIX_CACHE_TTL_SECONDS)
    meta["prefix_cached"] = True
    return f"[CACHED_SYSTEM_PREFIX:{key}]", meta

def _tool_result_dedup(tool_call_id: str, tool_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    If the exact same tool result is resent, replace with short pointer.
    (Useful when agent loops / replays tool outputs.)
    """
    meta = {"tool_dedup": False, "tool_hit": False, "key": None}
    if not ENABLE_TOOL_RESULT_DEDUP or rds is None:
        return tool_text, meta
    if not tool_call_id or not tool_text:
        return tool_text, meta
    if len(tool_text) < 2000:
        return tool_text, meta

    key = "toolres:" + _sha256_text(tool_call_id + ":" + tool_text)
    meta["key"] = key
    existing = _rds_get_str(key)
    if existing is not None:
        meta["tool_hit"] = True
        return f"[CACHED_TOOL_RESULT:{key}]", meta

    _rds_set_str(key, tool_text, PREFIX_CACHE_TTL_SECONDS)
    meta["tool_dedup"] = True
    return tool_text, meta

# =====================================================
# ROUTES
# =====================================================

@app.get("/health")
def health():
    return {
        "ok": True,
        "redis": (rds is not None),
        "default_model": DEFAULT_MODEL,
        "opus_model": OPUS_MODEL,
        "origin_lockdown": bool(ORIGIN_SECRET),
        "require_cf_access_headers": REQUIRE_CF_ACCESS_HEADERS,
        "upstream_timeout_seconds": UPSTREAM_TIMEOUT_SECONDS,
        "model_prefix": MODEL_PREFIX,
        "strip_ide_boilerplate": STRIP_IDE_BOILERPLATE,
        "enforce_diff_first": ENFORCE_DIFF_FIRST,
        "tool_result_max_chars": TOOL_RESULT_MAX_CHARS,
        "system_max_chars": SYSTEM_MAX_CHARS,
        "user_msg_max_chars": USER_MSG_MAX_CHARS,
        "enable_prefix_cache": ENABLE_PREFIX_CACHE,
        "enable_tool_result_dedup": ENABLE_TOOL_RESULT_DEDUP,
    }

@app.get("/debug/origin")
async def debug_origin(req: Request):
    v = req.headers.get("x-origin-secret")
    return {"has_x_origin_secret": bool(v), "x_origin_secret_len": len(v or "")}

@app.get("/debug/headers")
async def debug_headers(req: Request):
    return {
        "has_x_origin_secret": bool(req.headers.get("x-origin-secret")),
        "x_origin_secret_len": len(req.headers.get("x-origin-secret") or ""),
        "has_x_api_key": bool(req.headers.get("x-api-key")),
        "has_api_key": bool(req.headers.get("api-key")),
        "has_authorization": bool(req.headers.get("authorization")),
        "has_cf_access_id": bool(req.headers.get("cf-access-client-id")),
        "has_cf_access_secret": bool(req.headers.get("cf-access-client-secret")),
        "has_cf_ray": bool(req.headers.get("cf-ray")),
        "host": req.headers.get("host"),
        "path": req.url.path,
    }

# ---------------------
# Your existing endpoint
# ---------------------
@app.post("/chat")
async def chat(req: Request, body: ChatReq):
    raw = await req.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")

    ray = req.headers.get("cf-ray") or ""
    t0 = time.time()

    joined_user = "\n".join(m.content for m in body.messages if m.role == "user")
    model = route_model_from_messages(joined_user, body.model)

    payload = {
        "model": model,
        "system": body.system or "",
        "messages": [{"role": m.role, "content": m.content} for m in body.messages],
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
    }

    do_cache = body.temperature is None or body.temperature <= 0.3
    key = cache_key(payload) if do_cache else None

    if key:
        cached = cache_get(key)
        if cached:
            cached["cached"] = True
            return cached

    try:
        resp = await call_anthropic_with_timeout(payload)
    except HTTPException:
        raise
    except Exception as e:
        log.error("CHAT UPSTREAM ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream model error")

    try:
        text = extract_text_from_anthropic(resp)
    except Exception as e:
        log.error("CHAT PARSE ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream response parse error")

    usage = extract_usage(resp)
    data = {"cached": False, "model": model, "text": text, "usage": usage}

    if key:
        cache_set(key, data, CACHE_TTL_SECONDS)

    dt_ms = int((time.time() - t0) * 1000)
    log.info("CHAT OK (cf-ray=%s) model=%s ms=%s cached=%s", ray, model, dt_ms, False)

    return data

# -------------------------------------------
# OpenAI-compatible endpoint (TOOL CALLS + STREAMING)
# -------------------------------------------
@app.post("/v1/chat/completions")
async def openai_chat_completions(req: Request, body: OAChatReq):
    raw = await req.body()

    # DEBUG: inspect incoming structure
    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
        log.info("OA INCOMING KEYS: %s", list(parsed.keys()))
        log.info("OA HAS tools: %s", "tools" in parsed)
        log.info("OA HAS functions: %s", "functions" in parsed)
        log.info("OA HAS tool_choice: %s", "tool_choice" in parsed)
        log.info("OA HAS function_call: %s", "function_call" in parsed)
    except Exception:
        parsed = {}

    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")

    ray = req.headers.get("cf-ray") or ""
    t0 = time.time()

    # -------------------------
    # Tools / tool_choice
    # -------------------------
    oa_tools = _oa_tools_from_body(parsed)
    aa_tools = _anthropic_tools_from_openai(oa_tools)
    aa_tool_choice = _anthropic_tool_choice_from_openai(parsed.get("tool_choice"))

    # legacy: function_call="none"/"auto"/{"name":"x"} (rare)
    if aa_tool_choice is None and "function_call" in parsed:
        fc = parsed.get("function_call")
        if isinstance(fc, str):
            aa_tool_choice = _anthropic_tool_choice_from_openai(fc)
        elif isinstance(fc, dict) and fc.get("name"):
            aa_tool_choice = {"type": "tool", "name": fc["name"]}

    # -------------------------
    # Convert OpenAI messages -> Anthropic
    # including tool role messages
    # -------------------------
    system_parts: List[str] = []
    aa_messages: List[Dict[str, Any]] = []
    user_join: List[str] = []

    reduction_meta: Dict[str, Any] = {
        "system_strip": [],
        "user_trunc": [],
        "tool_trunc": [],
        "prefix_cache": None,
    }

    for m in body.messages:
        role = (m.role or "").lower()
        content_text = normalize_openai_content_to_text(m.content)

        if role in ("system", "developer"):
            # strip/truncate giant repeated blocks
            new_text, meta = _strip_or_truncate(role, content_text, SYSTEM_MAX_CHARS, allow_strip=True)
            reduction_meta["system_strip"].append(meta)
            if new_text.strip():
                system_parts.append(new_text.strip())
            continue

        if role == "tool":
            # OpenAI tool message must include tool_call_id (Cursor/Continue does this)
            tool_call_id = ""
            try:
                tool_call_id = (getattr(m, "tool_call_id", None) or m.__dict__.get("tool_call_id") or "")
            except Exception:
                tool_call_id = ""
            tool_text = content_text or ""
            # tool output truncation
            tool_text, tmeta = _strip_or_truncate("tool", tool_text, TOOL_RESULT_MAX_CHARS, allow_strip=False)
            reduction_meta["tool_trunc"].append(tmeta)

            # dedup very large repeated tool results
            tool_text, dmeta = _tool_result_dedup(tool_call_id, tool_text)
            if dmeta.get("tool_hit") or dmeta.get("tool_dedup"):
                reduction_meta.setdefault("tool_dedup", []).append(dmeta)

            if tool_call_id:
                aa_messages.append({"role": "user", "content": _anthropic_message_content_for_tool_result(tool_call_id, tool_text)})
            else:
                # fallback: include as normal user text (shouldn't happen, but don't break)
                aa_messages.append({"role": "user", "content": tool_text})
            continue

        if role in ("user", "assistant"):
            if role == "user":
                new_text, meta = _strip_or_truncate("user", content_text, USER_MSG_MAX_CHARS, allow_strip=False)
                reduction_meta["user_trunc"].append(meta)
                content_text = new_text
                if content_text:
                    user_join.append(content_text)
            aa_messages.append({"role": role, "content": content_text})
            continue

        # unknown roles -> ignore safely
        continue

    # -------------------------
    # System prompt assembly + diff-first policy
    # -------------------------
    system_text = "\n\n".join([p for p in system_parts if p]).strip()

    if ENFORCE_DIFF_FIRST:
        # Make sure we don't spam this if system is empty-cached reference
        system_text = (system_text + "\n\n" + _DIFF_FIRST_RULES).strip()

    # Truncate system if still too large
    system_text, smeta2 = _strip_or_truncate("system", system_text, SYSTEM_MAX_CHARS, allow_strip=False)
    reduction_meta["system_final"] = smeta2

    # Prefix caching (replaces giant repeated prefixes with a small pointer)
    system_text_cached, pmeta = _maybe_prefix_cache(system_text)
    reduction_meta["prefix_cache"] = pmeta
    system_text = system_text_cached

    joined_user = "\n".join(user_join)

    model = route_model_from_messages(joined_user, body.model)
    max_tokens = int(body.max_tokens or DEFAULT_MAX_TOKENS)
    temperature = body.temperature if body.temperature is not None else 0.2

    payload: Dict[str, Any] = {
        "model": model,
        "system": system_text,
        "messages": aa_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if aa_tools:
        payload["tools"] = aa_tools
        if aa_tool_choice:
            payload["tool_choice"] = aa_tool_choice

    do_cache = temperature <= 0.3
    key = cache_key(payload) if do_cache else None

    # -------------------------
    # Cache hit (stream + non-stream)
    # NOTE: tool-calling responses are cached only if they contain no tool_use blocks (text-only).
    # -------------------------
    if key:
        cached = cache_get(key)
        if cached and isinstance(cached, dict) and "text" in cached and cached.get("tool_calls") is None:
            out_text = cached.get("text", "")
            usage_cached = cached.get("usage")

            if body.stream:
                async def event_gen_cached():
                    chunk_id = f"chatcmpl_cached_{key[:12]}"
                    created = int(time.time())
                    text = out_text or ""
                    step = 60
                    for i in range(0, len(text), step):
                        piece = text[i:i+step]
                        event = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": with_model_prefix(str(cached.get("model", model))),
                            "choices": [{"index": 0, "delta": {"content": piece}}],
                        }
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                    final = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(str(cached.get("model", model))),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

                resp = StreamingResponse(event_gen_cached(), media_type="text/event-stream")
                resp.headers["Cache-Control"] = "no-cache"
                resp.headers["Connection"] = "keep-alive"
                resp.headers["X-Accel-Buffering"] = "no"
                resp.headers["X-Gateway"] = "gursimanoor-gateway"
                resp.headers["X-Model-Source"] = "custom"
                resp.headers["X-Cache"] = "HIT"
                resp.headers["X-Reduction"] = "1"
                dt_ms = int((time.time() - t0) * 1000)
                log.info("OA OK CACHED STREAM (cf-ray=%s) model=%s ms=%s", ray, model, dt_ms)
                return resp

            resp_json = {
                "id": f"chatcmpl_cached_{key[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": with_model_prefix(str(cached.get("model", model))),
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": out_text}, "finish_reason": "stop"}
                ],
                "usage": anthropic_to_openai_usage(usage_cached),
            }
            response = JSONResponse(content=resp_json)
            response.headers["X-Gateway"] = "gursimanoor-gateway"
            response.headers["X-Model-Source"] = "custom"
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Reduction"] = "1"
            dt_ms = int((time.time() - t0) * 1000)
            log.info("OA OK CACHED (cf-ray=%s) model=%s ms=%s", ray, model, dt_ms)
            return response

    # -------------------------
    # STREAMING: true tool-call compatible streaming via anthropic stream
    # -------------------------
    if body.stream:
        if client is None:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

        async def sse_stream():
            """
            Emits OpenAI-style SSE chunks, including tool_calls deltas when Claude emits tool_use.
            """
            chunk_id = f"chatcmpl_{hashlib.sha1((ray + str(time.time())).encode()).hexdigest()[:16]}"
            created = int(time.time())

            q: asyncio.Queue = asyncio.Queue()

            def _worker_stream():
                """
                Runs in a thread: iterate anthropic stream events and push JSON-able deltas into asyncio queue.
                """
                try:
                    with client.messages.stream(**payload) as stream:
                        # We'll accumulate tool_calls as they arrive
                        tool_calls_by_id: Dict[str, Dict[str, Any]] = {}

                        for ev in stream:
                            # Anthropic SDK exposes event objects; treat as dict-ish
                            etype = getattr(ev, "type", None) or (ev.get("type") if isinstance(ev, dict) else None)

                            # Content deltas (text/tool)
                            if etype in ("content_block_start", "content_block_delta", "content_block_stop"):
                                # get content block info
                                block = getattr(ev, "content_block", None) or (ev.get("content_block") if isinstance(ev, dict) else None)
                                index = getattr(ev, "index", None) if hasattr(ev, "index") else (ev.get("index") if isinstance(ev, dict) else None)

                                # For delta events, delta payload varies
                                delta = getattr(ev, "delta", None) or (ev.get("delta") if isinstance(ev, dict) else None)

                                btype = getattr(block, "type", None) if block is not None else None
                                if btype is None and isinstance(block, dict):
                                    btype = block.get("type")

                                # TEXT streaming
                                if btype == "text" and delta is not None:
                                    txt = getattr(delta, "text", None) if not isinstance(delta, dict) else delta.get("text")
                                    if txt:
                                        asyncio.run_coroutine_threadsafe(q.put(("text", txt)), loop)

                                # TOOL streaming: capture tool_use blocks.
                                # Anthropic emits tool_use as a full block (often at block_start).
                                if btype == "tool_use":
                                    # Try to read tool_use fields from block
                                    if isinstance(block, dict):
                                        tool_use_id = block.get("id") or ""
                                        tool_name = block.get("name") or ""
                                        tool_input = block.get("input") or {}
                                    else:
                                        tool_use_id = getattr(block, "id", "") or ""
                                        tool_name = getattr(block, "name", "") or ""
                                        tool_input = getattr(block, "input", {}) or {}

                                    if tool_use_id and tool_name:
                                        # OpenAI tool_call
                                        tc = {
                                            "id": tool_use_id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": _ensure_json_args_str(tool_input),
                                            },
                                        }
                                        tool_calls_by_id[tool_use_id] = tc
                                        asyncio.run_coroutine_threadsafe(q.put(("tool_call", tc)), loop)

                            # End / usage
                            if etype in ("message_stop", "message_end"):
                                break

                        # usage available at end (best-effort)
                        final = stream.get_final_message()
                        usage = extract_usage(final)
                        asyncio.run_coroutine_threadsafe(q.put(("done", usage)), loop)

                except Exception as e:
                    asyncio.run_coroutine_threadsafe(q.put(("error", str(e))), loop)

            loop = asyncio.get_running_loop()
            await asyncio.to_thread(_worker_stream)

            # consume queue and emit SSE
            finished = False
            while not finished:
                kind, payload_item = await q.get()

                if kind == "error":
                    # surface as stop
                    final = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    finished = True
                    continue

                if kind == "text":
                    event = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {"content": payload_item}}],
                    }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue

                if kind == "tool_call":
                    # OpenAI tool_calls delta: provide tool_calls array with one entry
                    event = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [payload_item]},
                        }],
                    }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue

                if kind == "done":
                    usage = payload_item
                    # final chunk
                    final = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    finished = True
                    # cache text-only streams is hard; we do not cache streamed responses
                    continue

        response = StreamingResponse(sse_stream(), media_type="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["X-Gateway"] = "gursimanoor-gateway"
        response.headers["X-Model-Source"] = "custom"
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Reduction"] = "1"
        dt_ms = int((time.time() - t0) * 1000)
        log.info("OA OK STREAM (cf-ray=%s) model=%s ms=%s tools=%s", ray, model, dt_ms, bool(aa_tools))
        return response

    # -------------------------
    # Non-stream upstream call
    # -------------------------
    try:
        resp = await call_anthropic_with_timeout(payload)
    except HTTPException:
        raise
    except Exception as e:
        log.error("OA UPSTREAM ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream model error")

    # -------------------------
    # Parse Anthropic response: text + tool_use blocks
    # -------------------------
    out_text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    try:
        for block in getattr(resp, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                out_text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_use_id = getattr(block, "id", "") or ""
                name = getattr(block, "name", "") or ""
                tool_input = getattr(block, "input", {}) or {}
                if tool_use_id and name:
                    tool_calls.append({
                        "id": tool_use_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _ensure_json_args_str(tool_input),
                        },
                    })
    except Exception as e:
        log.error("OA PARSE ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream response parse error")

    out_text = "".join(out_text_parts)
    usage = extract_usage(resp)

    # -------------------------
    # Cache only when no tool calls (safe text-only caching)
    # -------------------------
    cache_blob = {"cached": False, "model": model, "text": out_text, "usage": usage, "tool_calls": (tool_calls or None)}
    if key and not tool_calls:
        cache_set(key, cache_blob, CACHE_TTL_SECONDS)

    # -------------------------
    # OpenAI response
    # If tool_calls exist, content should usually be "" and finish_reason="tool_calls"
    # -------------------------
    finish_reason = "stop"
    message_obj: Dict[str, Any] = {"role": "assistant", "content": out_text}

    if tool_calls:
        # OpenAI convention: assistant content often empty when emitting tool_calls
        message_obj["content"] = out_text or ""
        message_obj["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    resp_json = {
        "id": f"chatcmpl_{hashlib.sha1((ray + str(time.time())).encode()).hexdigest()[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": with_model_prefix(model),
        "choices": [
            {"index": 0, "message": message_obj, "finish_reason": finish_reason}
        ],
        "usage": anthropic_to_openai_usage(usage),
    }

    response = JSONResponse(content=resp_json)
    response.headers["X-Gateway"] = "gursimanoor-gateway"
    response.headers["X-Model-Source"] = "custom"
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Reduction"] = "1"

    dt_ms = int((time.time() - t0) * 1000)
    log.info(
        "OA OK (cf-ray=%s) model=%s ms=%s tools_in=%s tools_out=%s",
        ray, model, dt_ms, bool(aa_tools), bool(tool_calls)
    )
    return response

# === COMPATIBILITY ALIAS (some clients use non-versioned path) ===
@app.post("/chat/completions")
async def chat_completions_alias(req: Request, body: OAChatReq):
    return await openai_chat_completions(req, body)

# -------------------------
# OpenAI-compatible models
# -------------------------
@app.get("/v1/models")
async def openai_models(req: Request):
    ray = req.headers.get("cf-ray") or ""
    ua = req.headers.get("user-agent") or ""
    log.info("MODELS HIT cf-ray=%s ua=%s", ray, ua[:120])

    payload = {
        "object": "list",
        "data": [
            {"id": with_model_prefix("sonnet"), "object": "model"},
            {"id": with_model_prefix("opus"), "object": "model"},
            {"id": with_model_prefix(DEFAULT_MODEL), "object": "model"},
            {"id": with_model_prefix(OPUS_MODEL), "object": "model"},

            # Also include unprefixed for compatibility
            {"id": "sonnet", "object": "model"},
            {"id": "opus", "object": "model"},
            {"id": DEFAULT_MODEL, "object": "model"},
            {"id": OPUS_MODEL, "object": "model"},
        ],
    }

    resp = JSONResponse(content=payload)
    resp.headers["X-Gateway"] = "gursimanoor-gateway"
    resp.headers["X-Model-Source"] = "custom"
    return resp

# === COMPATIBILITY ALIAS (some clients use non-versioned path) ===
@app.get("/models")
async def models_alias(req: Request):
    return await openai_models(req)
