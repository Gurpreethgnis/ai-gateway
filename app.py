from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError

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
    if DATABASE_URL:
        try:
            from gateway.db import init_db, create_tables
            init_db()
            await create_tables()
            
            # Manual migration for new columns
            from sqlalchemy import text
            from gateway.db import engine
            async with engine.begin() as conn:
                try:
                    await conn.execute(text("ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS cache_read_input_tokens INTEGER DEFAULT 0"))
                    await conn.execute(text("ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS cache_creation_input_tokens INTEGER DEFAULT 0"))
                    log.info("Database migration (cache columns) applied successfully")
                except Exception as e:
                    log.debug("Migration already applied or failed: %r", e)
            
            log.info("Database initialized successfully")
        except Exception as e:
            log.warning("Database initialization failed: %r", e)
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
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(openai_router)
app.include_router(admin_router)
app.include_router(dashboard_router)

if ENABLE_BATCH_API:
    from gateway.batch import router as batch_router
    app.include_router(batch_router)
