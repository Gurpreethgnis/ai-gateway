from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from gateway.logging_setup import (
    setup_logging,
    validation_exception_handler,
    log_404_middleware,
    request_log_middleware,
)
from gateway.security import security_middleware

from gateway.routers.health import router as health_router
from gateway.routers.chat import router as chat_router
from gateway.routers.openai import router as openai_router

setup_logging()

app = FastAPI()

# exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# middleware
app.middleware("http")(log_404_middleware)
app.middleware("http")(request_log_middleware)
app.middleware("http")(security_middleware)

# routes
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(openai_router)
