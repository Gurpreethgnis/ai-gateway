from typing import Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from gateway.config import ORIGIN_SECRET, REQUIRE_CF_ACCESS_HEADERS, GATEWAY_API_KEY
from gateway.logging_setup import log

def extract_gateway_api_key(request: Request) -> Optional[str]:
    for hdr in ("x-api-key", "api-key"):
        v = request.headers.get(hdr)
        if v:
            return v.strip() or None

    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip() or None
    return None

def is_public_path(path: str) -> bool:
    return path in ("/", "/favicon.ico", "/metrics", "/health", "/live", "/dashboard") or path.startswith("/js/") or path.startswith("/cdn-cgi/")

def is_protected_api_path(path: str) -> bool:
    return path.startswith("/v1/") or path in ("/chat", "/debug/origin", "/debug/headers", "/chat/completions")

async def security_middleware(request: Request, call_next):
    try:
        path = request.url.path

        if is_public_path(path):
            return await call_next(request)

        if not is_protected_api_path(path):
            return await call_next(request)

        if ORIGIN_SECRET:
            got = request.headers.get("x-origin-secret")
            if got != ORIGIN_SECRET:
                log.warning("BLOCK origin secret missing/mismatch path=%s host=%s", path, request.headers.get("host"))
                raise HTTPException(status_code=403, detail="Forbidden")

        if REQUIRE_CF_ACCESS_HEADERS:
            if not request.headers.get("cf-access-client-id") or not request.headers.get("cf-access-client-secret"):
                log.warning("BLOCK missing CF Access headers path=%s host=%s", path, request.headers.get("host"))
                raise HTTPException(status_code=403, detail="Missing Cloudflare Access headers")

        api_key = extract_gateway_api_key(request)
        if not api_key:
            present = ",".join([k for k in request.headers.keys() if k.lower() in ("authorization", "x-api-key", "api-key")])
            log.warning("BLOCK missing api key path=%s host=%s present_headers=%s", path, request.headers.get("host"), present)
            raise HTTPException(status_code=401, detail="API key required (X-API-Key/api-key/Authorization: Bearer)")

        if api_key != GATEWAY_API_KEY:
            log.warning("BLOCK invalid api key path=%s host=%s", path, request.headers.get("host"))
            raise HTTPException(status_code=403, detail="Invalid API key")

        return await call_next(request)

    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
