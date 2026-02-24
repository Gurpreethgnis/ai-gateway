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

import asyncio

# Routers and db are init'd in lifespan after yield so /live can respond immediately (Railway healthcheck).
# Do not import routers or gateway.db here.

setup_logging()


async def _init_db_async(app: FastAPI) -> None:
    """Initialize DB in background so lifespan can yield first."""
    if not DATABASE_URL:
        return
    from gateway.db import init_db, background_db_init
    init_db()
    app.state.db_init_task = asyncio.create_task(background_db_init())
    log.info("Database initialization started in background")


def _add_routers(app: FastAPI) -> None:
    """Load and mount all routers. Called after server is accepting so /live responds immediately."""
    from gateway.routers.health import router as health_router
    from gateway.routers.chat import router as chat_router
    from gateway.routers.openai import router as openai_router
    from gateway.routers.admin import router as admin_router
    from gateway.routers.dashboard import router as dashboard_router
    from gateway.routers.auth import router as auth_router
    from gateway.routers.dashboard_api import router as dashboard_api_router

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

    log.info("All routers mounted")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Lifespan starting")
    try:
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

        asyncio.create_task(asyncio.to_thread(_clear_redis_slots))

        # DB init in background so we don't import gateway.db before yield (keeps /live fast).
        asyncio.create_task(_init_db_async(app))

        # Run model and provider registry init in background so server can accept
        # connections (and /health) quickly; avoids Railway healthcheck timeout.
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

        # Mount all routers in a background thread so /live is already served; avoids Railway healthcheck timeout.
        asyncio.create_task(asyncio.to_thread(_add_routers, app))

        log.info("Lifespan ready, accepting connections")
        yield
    except Exception as e:
        log.exception("Lifespan failed: %r", e)
        raise


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


@app.get("/live")
@app.get("/live/")
async def live():
    """Minimal liveness probe: no dependencies, returns 200 as soon as the process is up.
    Both /live and /live/ return 200 so Railway healthcheck never gets a 307 redirect."""
    return {"live": True}


# Routes below are mounted in _add_routers() after yield so /live responds before heavy imports.
