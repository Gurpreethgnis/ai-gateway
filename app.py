from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse

from gateway.logging_setup import (
    setup_logging,
    validation_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
    log_404_middleware,
    request_log_middleware,
    log,
)
from gateway.security import security_middleware
from gateway.config import DATABASE_URL, ENABLE_BATCH_API

from gateway.routers.health import router as health_router
from gateway.routers.chat import router as chat_router
from gateway.routers.openai import router as openai_router
from gateway.routers.admin import router as admin_router
from gateway.routers.dashboard import router as dashboard_router

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clear any stale concurrency slots left over from previous deployments
    try:
        from gateway.cache import rds
        if rds:
            for family in ("sonnet", "opus", "haiku", "default"):
                rds.delete(f"concurrency:anthropic:{family}")
            log.info("Cleared stale concurrency slots from Redis")
    except Exception as e:
        log.warning("Could not clear Redis concurrency slots: %r", e)

    if DATABASE_URL:
        try:
            from gateway.db import init_db, create_tables
            init_db()
            await create_tables()
            log.info("Database initialized successfully")
        except Exception as e:
            log.warning("Database initialization failed: %r", e)
    
    # One-time backfill of dashboard counters from existing usage_records
    try:
        from gateway.cache import rds
        if rds and DATABASE_URL:
            if not rds.exists("dashboard:stats:all_time"):
                log.info("Backfilling dashboard counters from existing usage_records...")
                await _backfill_dashboard_counters()
    except Exception as e:
        log.warning("Dashboard backfill failed (non-critical): %r", e)
    
    yield


async def _backfill_dashboard_counters():
    """One-time backfill of Redis counters from existing usage_records."""
    from gateway.db import get_session
    from gateway.cache import rds
    from sqlalchemy import text
    
    if not rds:
        return
    
    try:
        async with get_session() as session:
            row = (await session.execute(text("""
                SELECT COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(cache_read_input_tokens), 0),
                       COALESCE(SUM(gateway_tokens_saved), 0),
                       COALESCE(SUM(cost_usd), 0),
                       COUNT(*)
                  FROM usage_records
            """))).fetchone()
            
            if row:
                input_tok, output_tok, cached, gw_saved, cost, count = row
                pipe = rds.pipeline()
                pipe.hset("dashboard:stats:all_time", mapping={
                    "input_tokens": int(input_tok),
                    "output_tokens": int(output_tok),
                    "cached_tokens": int(cached),
                    "gateway_saved": int(gw_saved),
                    "cost_usd": float(cost),
                    "request_count": int(count),
                })
                pipe.execute()
                log.info("Backfilled dashboard counters: %d requests, %d input tokens", int(count), int(input_tok))
    except Exception as e:
        log.warning("Backfill query failed: %r", e)


app = FastAPI(lifespan=lifespan)

# exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# middleware
app.middleware("http")(log_404_middleware)
app.middleware("http")(request_log_middleware)
app.middleware("http")(security_middleware)

# routes
@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard", status_code=302)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(openai_router)
app.include_router(admin_router)
app.include_router(dashboard_router)

if ENABLE_BATCH_API:
    from gateway.batch import router as batch_router
    app.include_router(batch_router)
