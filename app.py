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
from gateway.routers.auth import router as auth_router
from gateway.routers.dashboard_api import router as dashboard_api_router

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clear any stale concurrency slots in a background task so Redis I/O never blocks startup.
    def _clear_redis_slots():
        try:
            from gateway.cache import rds
            if rds:
                for family in ("sonnet", "opus", "haiku", "default"):
                    rds.delete(f"concurrency:anthropic:{family}")
                log.info("Cleared stale concurrency slots from Redis")
        except Exception as e:
            log.warning("Could not clear Redis concurrency slots: %r", e)

    import asyncio
    asyncio.create_task(asyncio.to_thread(_clear_redis_slots))

    # Initialize database in background (non-blocking)
    if DATABASE_URL:
        import asyncio
        from gateway.db import init_db, background_db_init
        init_db()  # Creates engine config (instant, no connection)
        app.state.db_init_task = asyncio.create_task(background_db_init())
        log.info("Database initialization started in background")

    # Run model and provider registry init in background so server can accept
    # connections (and /health) quickly; avoids Railway healthcheck timeout.
    import asyncio
    async def _init_registries():
        try:
            from gateway.model_registry import get_model_registry
            registry = get_model_registry()
            await registry.initialize()
            log.info("Model registry initialized with %d models", len(registry.get_all_models()))
        except Exception as e:
            log.warning("Could not initialize model registry: %r", e)
        try:
            from gateway.providers.registry import get_provider_registry
            provider_registry = get_provider_registry()
            await provider_registry.initialize()
            providers = provider_registry.get_available_providers()
            log.info("Provider registry initialized: %s", list(providers.keys()))
        except Exception as e:
            log.warning("Could not initialize provider registry: %r", e)
    asyncio.create_task(_init_registries())
    
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
app.include_router(auth_router)
app.include_router(dashboard_api_router)

if ENABLE_BATCH_API:
    from gateway.batch import router as batch_router
    app.include_router(batch_router)
