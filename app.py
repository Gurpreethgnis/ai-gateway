import os
import json
import hashlib
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
    raise RuntimeError(
        "GATEWAY_API_KEY is not set. Set it in Railway environment variables."
    )

# Recommended: Cloudflare-only origin secret injected by Transform Rule
# Cloudflare should inject:
#   X-Origin-Secret: <secret>
ORIGIN_SECRET = os.getenv("ORIGIN_SECRET")  # set this in Railway

# Optional: enforce presence of CF Access client headers at app layer
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
# ERROR HARDENING
# =====================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, __: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal error"})

# =====================================================
# MIDDLEWARE: AUTH ON ALL REQUESTS
# =====================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # 1) Origin lockdown: require Cloudflare-injected secret header (if configured)
    if ORIGIN_SECRET:
        got = request.headers.get("x-origin-secret")
        if got != ORIGIN_SECRET:
            raise HTTPException(status_code=403, detail="Forbidden")

    # 2) Optional: additional enforcement of CF Access client headers at app layer
    if REQUIRE_CF_ACCESS_HEADERS:
        cf_id = request.headers.get("cf-access-client-id")
        cf_secret = request.headers.get("cf-access-client-secret")
        if not cf_id or not cf_secret:
            raise HTTPException(status_code=401, detail="Missing Cloudflare Access headers")

    # 3) Require X-API-Key for all requests
    api_key = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key required")
    if api_key != GATEWAY_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return await call_next(request)

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
    if not rds:
        return None
    try:
        v = rds.get(key)
        return json.loads(v) if v else None
    except Exception:
        return None

def cache_set(key: str, data: Dict[str, Any], ttl: int):
    if rds
