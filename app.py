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
        import asyncio
        from gateway.db import init_db, create_tables
        init_db()
        for attempt in range(3):
            try:
                await asyncio.wait_for(create_tables(), timeout=15)
                log.info("Database initialized successfully")
                break
            except (asyncio.TimeoutError, Exception) as e:
                if attempt < 2:
                    wait = (attempt + 1) * 3
                    log.warning("Database init attempt %d failed: %r, retrying in %ds", attempt + 1, e, wait)
                    await asyncio.sleep(wait)
                else:
                    log.warning("Database initialization failed after 3 attempts: %r (gateway will run without DB)", e)
    
    yield


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
