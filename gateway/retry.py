import asyncio
from typing import Callable, Any, Optional, Tuple, Type
from functools import wraps

from gateway.config import RETRY_ENABLED, RETRY_MAX_ATTEMPTS, RETRY_BACKOFF_BASE
from gateway.logging_setup import log


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

RETRYABLE_ERROR_KEYWORDS = [
    "timeout",
    "connection",
    "overloaded",
    "rate limit",
    "temporarily unavailable",
    "service unavailable",
]


class RetryExhaustedError(Exception):
    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_error}")


def is_retryable_error(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    if status is None:
        resp = getattr(e, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)

    if status in RETRYABLE_STATUS_CODES:
        return True

    err_str = str(e).lower()
    return any(kw in err_str for kw in RETRYABLE_ERROR_KEYWORDS)


def get_retry_after(e: Exception) -> Optional[float]:
    resp = getattr(e, "response", None)
    if resp is None:
        return None

    headers = getattr(resp, "headers", {})
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return None


async def with_retry(
    func: Callable,
    *args,
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    backoff_base: float = RETRY_BACKOFF_BASE,
    retryable_check: Callable[[Exception], bool] = is_retryable_error,
    **kwargs,
) -> Any:
    if not RETRY_ENABLED:
        return await func(*args, **kwargs)

    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if not retryable_check(e):
                raise

            if attempt == max_attempts - 1:
                log.warning(
                    "Retry exhausted after %d attempts: %r",
                    max_attempts,
                    e,
                )
                raise RetryExhaustedError(max_attempts, e)

            retry_after = get_retry_after(e)
            if retry_after is not None:
                wait_time = retry_after
            else:
                wait_time = backoff_base * (2**attempt)

            wait_time = min(wait_time, 30.0)

            log.info(
                "Retry attempt %d/%d after %.1fs: %r",
                attempt + 1,
                max_attempts,
                wait_time,
                e,
            )

            await asyncio.sleep(wait_time)

    raise RetryExhaustedError(max_attempts, last_error)


def retry_decorator(
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    backoff_base: float = RETRY_BACKOFF_BASE,
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await with_retry(
                func,
                *args,
                max_attempts=max_attempts,
                backoff_base=backoff_base,
                **kwargs,
            )

        return wrapper

    return decorator


async def retry_with_fallback(
    primary_func: Callable,
    fallback_func: Callable,
    *args,
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    **kwargs,
) -> Tuple[Any, bool]:
    try:
        result = await with_retry(primary_func, *args, max_attempts=max_attempts, **kwargs)
        return result, False
    except (RetryExhaustedError, Exception) as e:
        log.warning("Primary function failed, using fallback: %r", e)
        result = await fallback_func(*args, **kwargs)
        return result, True
