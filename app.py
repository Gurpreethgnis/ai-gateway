import os
import json
import time
import hashlib
from typing import Any, Optional, List, Dict

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from anthropic import Anthropic

# Optional Redis cache (recommended for multi-device cost savings)
try:
    import redis
except Exception:
    redis = None

app = FastAPI()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GATEWAY_API_KEY = os.environ.get("GATEWAY_API_KEY")  # second lock (recommended)
REDIS_URL = os.environ.get("REDIS_URL")              # optional
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "1800"))  # 30 min default

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-0")
OPUS_MODEL = os.environ.get("OPUS_MODEL", "claude-opus-4-5")
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "1200"))

MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "250000"))  # ~250KB guardrail

client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

rds = None
if REDIS_URL and redis is not None:
    try:
        rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        # quick ping at startup
        rds.ping()
    except Exception:
        rds = None


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatReq(BaseModel):
    system: Optional[str] = ""
    messages: List[ChatMessage]
    max_tokens: int = DEFAULT_MAX_TOKENS
    model: Optional[str] = None  # optional override: "claude-opus-4-5" etc.
    temperature: Optional[float] = 0.2


def require_api_key(x_api_key: Optional[str]):
    # If you set GATEWAY_API_KEY, require it. If you omit it, this lock is disabled.
    if GATEWAY_API_KEY and x_api_key != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def is_hard_task(text: str) -> bool:
    t = text.lower()
    signals = [
        "architecture", "design doc", "production", "incident", "root cause",
        "migration", "refactor", "security", "performance", "optimize",
        "kubernetes", "terraform", "rollout", "zero downtime", "deploy",
        "database migration", "oncall", "postmortem",
    ]
    return any(s in t for s in signals)


def route_model(req: ChatReq) -> str:
    if req.model:
        return req.model

    # Join first ~8k chars of user input to route
    joined = "\n".join([m.content for m in req.messages if m.role == "user"])[:8000]
    if is_hard_task(joined):
        return OPUS_MODEL
    return DEFAULT_MODEL


def cache_key(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not rds:
        return None
    v = rds.get(key)
    if not v:
        return None
    try:
        return json.loads(v)
    except Exception:
        return None


def cache_set(key: str, data: Dict[str, Any], ttl: int):
    if not rds:
        return
    rds.setex(key, ttl, json.dumps(data, ensure_ascii=False))


@app.get("/health")
def health():
    return {
        "ok": True,
        "redis": bool(rds),
        "default_model": DEFAULT_MODEL,
        "opus_model": OPUS_MODEL,
    }


@app.post("/chat")
async def chat(req: Request, body: ChatReq, x_api_key: Optional[str] = Header(default=None)):
    # Guard: key
    require_api_key(x_api_key)

    # Guard: body size
    raw = await req.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail=f"Payload too large (> {MAX_BODY_BYTES} bytes)")

    if not client:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set on server")

    model = route_model(body)

    payload = {
        "model": model,
        "system": body.system or "",
        "messages": [{"role": m.role, "content": m.content} for m in body.messages],
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
    }

    # Cache only when request is deterministic-ish (temperature low)
    do_cache = (body.temperature is None) or (body.temperature <= 0.3)
    key = cache_key(payload) if do_cache else None

    if key:
        cached = cache_get(key)
        if cached:
            cached["cached"] = True
            return cached

    resp = client.messages.create(**payload)

    out = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            out += block.text

    data = {
        "cached": False,
        "model": model,
        "text": out,
        "usage": getattr(resp, "usage", None),
    }

    if key:
        cache_set(key, data, CACHE_TTL_SECONDS)

    return data
