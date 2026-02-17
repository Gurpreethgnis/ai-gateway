import os
import json
import hashlib
import traceback
import asyncio
import time
import logging
from typing import Any, Optional, List, Dict, Literal, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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
# SECURITY CONFIG
# =====================================================

GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY")
if not GATEWAY_API_KEY:
    raise RuntimeError("GATEWAY_API_KEY is not set. Set it in Railway env vars.")

# Cloudflare Transform Rule injects:
#   X-Origin-Secret: <secret>
ORIGIN_SECRET = os.getenv("ORIGIN_SECRET")  # set in Railway (recommended)

# Optional extra enforcement at app layer
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

# Model proof prefix (what you want to see inside Cursor)
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "MYMODEL:")

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
# Root: helpful for sanity checks + keeps Cursor happy
# -----------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "ai-gateway"}

# =====================================================
# REQUEST LOGGING
# =====================================================

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
      - Authorization: Bearer <key>
    (Cursor often uses Authorization: Bearer ...)
    """
    x = request.headers.get("x-api-key")
    if x:
        return x

    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip() or None

    return None

def is_public_path(path: str) -> bool:
    """
    Allow Cloudflare/cdn-cgi/browser-check assets and other non-API paths.
    Cursor/Electron can trigger these.
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
    return path.startswith("/v1/") or path in ("/chat", "/health", "/debug/origin", "/debug/headers")

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
                raise HTTPException(status_code=403, detail="Forbidden")

        # 2) Optional CF Access header enforcement (usually handled by CF Access itself)
        if REQUIRE_CF_ACCESS_HEADERS:
            if not request.headers.get("cf-access-client-id") or not request.headers.get("cf-access-client-secret"):
                raise HTTPException(status_code=403, detail="Missing Cloudflare Access headers")

        # 3) API key (client auth)
        api_key = extract_gateway_api_key(request)
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required (X-API-Key or Authorization: Bearer)")
        if api_key != GATEWAY_API_KEY:
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
# MODELS (OPENAI COMPAT FOR CURSOR)
# =====================================================

OpenAIRole = Literal["system", "user", "assistant", "tool", "developer"]

# Cursor sometimes sends content as a string, sometimes as structured parts.
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
    Cursor may send back whatever it sees in /v1/models.
    If you prefix model IDs (MYMODEL:...), strip it before routing.
    """
    if not m:
        return None
    s = m.strip()
    low = s.lower()

    # Accept both "MYMODEL:xxx" and "MYMODEL-xxx" style
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
    Map friendly aliases (Cursor/OpenAI model strings) into Anthropic model IDs.
    If unknown, return None (let routing decide).
    """
    if not maybe:
        return None

    # NEW: strip your proof prefix if Cursor sends it back
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
    # If Cursor sends content parts (list/dict), stringify safely.
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
# OpenAI-compatible endpoint for Cursor
# -------------------------------------------
@app.post("/v1/chat/completions")
async def openai_chat_completions(req: Request, body: OAChatReq):
    raw = await req.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")

    # Cursor can work without streaming; add later if you want
    if body.stream:
        raise HTTPException(status_code=400, detail="stream=true not supported yet")

    ray = req.headers.get("cf-ray") or ""
    t0 = time.time()

    # Convert OpenAI messages -> Anthropic
    system_parts: List[str] = []
    aa_messages: List[Dict[str, str]] = []
    user_join: List[str] = []

    for m in body.messages:
        role = (m.role or "").lower()
        content_text = normalize_openai_content_to_text(m.content)

        if role in ("system", "developer"):
            if content_text.strip():
                system_parts.append(content_text.strip())
        elif role in ("user", "assistant"):
            aa_messages.append({"role": role, "content": content_text})
            if role == "user" and content_text:
                user_join.append(content_text)
        else:
            continue

    system_text = "\n\n".join(system_parts).strip()

    joined_user = "\n".join(user_join)
    model = route_model_from_messages(joined_user, body.model)

    max_tokens = int(body.max_tokens or DEFAULT_MAX_TOKENS)
    temperature = body.temperature if body.temperature is not None else 0.2

    payload = {
        "model": model,
        "system": system_text,
        "messages": aa_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    do_cache = temperature <= 0.3
    key = cache_key(payload) if do_cache else None

    # Cache hit
    if key:
        cached = cache_get(key)
        if cached and isinstance(cached, dict) and "text" in cached:
            out_text = cached.get("text", "")
            resp_json = {
                "id": f"chatcmpl_cached_{key[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": with_model_prefix(str(cached.get("model", model))),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": out_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": anthropic_to_openai_usage(cached.get("usage")),
            }

            response = JSONResponse(content=resp_json)
            response.headers["X-Gateway"] = "gursimanoor-gateway"
            response.headers["X-Model-Source"] = "custom"
            response.headers["X-Cache"] = "HIT"

            dt_ms = int((time.time() - t0) * 1000)
            log.info("OA OK CACHED (cf-ray=%s) model=%s ms=%s", ray, model, dt_ms)
            return response

    # Upstream call
    try:
        resp = await call_anthropic_with_timeout(payload)
    except HTTPException:
        raise
    except Exception as e:
        log.error("OA UPSTREAM ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream model error")

    try:
        out_text = extract_text_from_anthropic(resp)
    except Exception as e:
        log.error("OA PARSE ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream response parse error")

    usage = extract_usage(resp)

    cache_blob = {"cached": False, "model": model, "text": out_text, "usage": usage}
    if key:
        cache_set(key, cache_blob, CACHE_TTL_SECONDS)

    resp_json = {
        "id": f"chatcmpl_{hashlib.sha1((ray + str(time.time())).encode()).hexdigest()[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": with_model_prefix(model),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": out_text},
                "finish_reason": "stop",
            }
        ],
        "usage": anthropic_to_openai_usage(usage),
    }

    response = JSONResponse(content=resp_json)
    response.headers["X-Gateway"] = "gursimanoor-gateway"
    response.headers["X-Model-Source"] = "custom"
    response.headers["X-Cache"] = "MISS"

    dt_ms = int((time.time() - t0) * 1000)
    log.info("OA OK (cf-ray=%s) model=%s ms=%s", ray, model, dt_ms)

    return response

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
            # Prefixed (what you want to visually prove in Cursor)
            {"id": with_model_prefix("sonnet"), "object": "model"},
            {"id": with_model_prefix("opus"), "object": "model"},
            {"id": with_model_prefix(DEFAULT_MODEL), "object": "model"},
            {"id": with_model_prefix(OPUS_MODEL), "object": "model"},

            # Optional: also include unprefixed models (safer for scripts / other clients)
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
