from fastapi import APIRouter, Request
from gateway.config import (
    DEFAULT_MODEL, OPUS_MODEL, ORIGIN_SECRET, REQUIRE_CF_ACCESS_HEADERS,
    UPSTREAM_TIMEOUT_SECONDS, MODEL_PREFIX,
    STRIP_IDE_BOILERPLATE, ENFORCE_DIFF_FIRST, TOOL_RESULT_MAX_CHARS,
    SYSTEM_MAX_CHARS, USER_MSG_MAX_CHARS, ENABLE_ANTHROPIC_CACHE_CONTROL,
)
from gateway.cache import rds

router = APIRouter()

@router.get("/health")
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
        "enable_anthropic_cache_control": ENABLE_ANTHROPIC_CACHE_CONTROL,
    }

@router.get("/debug/origin")
async def debug_origin(req: Request):
    v = req.headers.get("x-origin-secret")
    return {"has_x_origin_secret": bool(v), "x_origin_secret_len": len(v or "")}

@router.get("/debug/headers")
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
