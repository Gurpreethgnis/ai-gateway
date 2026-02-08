import os
import json
import hashlib
import traceback
from typing import Any, Optional, List, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from anthropic import Anthropic

# Optional Redis cache
try:
    import redis
except Exception:
    redis = None

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

# Optional (usually unnecessary) extra enforcement at app layer
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

client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

rds = None
if REDIS_URL and redis is not None:
    try:
        rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        rds.ping()
    except Exception:
        rds = None

# =====================================================
# ERROR HARDENING (WITH LOGGING)
# =====================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    # Log full stack trace to Railway logs
    print("UNHANDLED EXCEPTION:", repr(exc))
    print(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal error"})

# =====================================================
# MIDDLEWARE: AUTH ON ALL REQUESTS
# =====================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    try:
        # 1) Origin lockdown
        if ORIGIN_SECRET:
            got = request.headers.get("x-origin-secret")
            if got != ORIGIN_SECRET:
                raise HTTPException(status_code=403, detail="Forbidden")

        # 2) Optional CF headers enforcement
        if REQUIRE_CF_ACCESS_HEADERS:
            cf_id = request.headers.get("cf-access-client-id")
            cf_secret = request.headers.get("cf-access-client-secret")
            if not cf_id or not cf_secret:
                raise HTTPException(status_code=401, detail="Missing Cloudflare Access headers")

        # 3) API key
        api_key = request.headers.get("x-api-key")
        if not api_key:
            raise HTTPException(status_code=401, detail="X-API-Key required")
        if api_key != GATEWAY_API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")

        return await call_next(request)

    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# =====================================================
# MODELS
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

def route_model(req: ChatReq) -> str:
    if req.model:
        return req.model
    joined = "\n".join(m.content for m in req.messages if m.role == "user")[:8000]
    return OPUS_MODEL if is_hard_task(joined) else DEFAULT_MODEL

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
    }

@app.get("/debug/headers")
async def debug_headers(req: Request):
    return {
        "has_x_origin_secret": bool(req.headers.get("x-origin-secret")),
        "x_origin_secret_len": len(req.headers.get("x-origin-secret") or ""),
        "has_api_key": bool(req.headers.get("x-api-key")),
    }


@app.post("/chat")
async def chat(req: Request, body: ChatReq):
    raw = await req.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")

    if not client:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    model = route_model(body)

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
        resp = client.messages.create(**payload)
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream model error")

    text = "".join(
        block.text for block in resp.content
        if getattr(block, "type", None) == "text"
    )

    usage_obj = getattr(resp, "usage", None)
    try:
        usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj
    except Exception:
        usage = None

    data = {"cached": False, "model": model, "text": text, "usage": usage}

    if key:
        cache_set(key, data, CACHE_TTL_SECONDS)

    return data
