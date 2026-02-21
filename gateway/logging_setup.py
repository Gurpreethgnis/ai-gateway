import logging
import time
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse

from gateway.config import LOG_LEVEL

log = logging.getLogger("gateway")

def setup_logging():
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))

async def validation_exception_handler(request: Request, exc):
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

async def log_404_middleware(request: Request, call_next):
    resp = await call_next(request)
    if resp.status_code == 404:
        ray = request.headers.get("cf-ray") or ""
        ua = request.headers.get("user-agent") or ""
        log.warning("404 %s %s cf-ray=%s ua=%s", request.method, request.url.path, ray, ua[:120])
    return resp

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

async def http_exception_handler(_: Request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

async def unhandled_exception_handler(request: Request, exc: Exception):
    ray = request.headers.get("cf-ray") or ""
    log.error("UNHANDLED EXCEPTION (cf-ray=%s): %r", ray, exc)
    log.error(traceback.format_exc())
    # When X-Debug: 1 or DEBUG_ERROR_RESPONSE=1, return the real error so you can fix it (e.g. in Railway logs or curl -H "X-Debug: 1")
    import os
    debug = request.headers.get("x-debug", "").strip() == "1" or os.getenv("DEBUG_ERROR_RESPONSE", "").strip() == "1"
    if debug:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "cf_ray": ray,
            },
        )
    return JSONResponse(status_code=500, content={"detail": "Internal error"})
