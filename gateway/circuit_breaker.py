import time
import asyncio
from typing import Callable, Any, Optional
from dataclasses import dataclass

from gateway.config import (
    CIRCUIT_BREAKER_ENABLED,
    CIRCUIT_BREAKER_THRESHOLD,
    CIRCUIT_BREAKER_TIMEOUT,
)
from gateway.cache import rds
from gateway.logging_setup import log


class CircuitOpenError(Exception):
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker is open. Retry after {retry_after}s")


@dataclass
class CircuitState:
    state: str
    failure_count: int
    last_failure_time: float
    last_success_time: float


class CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str = "anthropic",
        threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        timeout: int = CIRCUIT_BREAKER_TIMEOUT,
    ):
        self.name = name
        self.threshold = threshold
        self.timeout = timeout
        self._key_prefix = f"circuit:{name}"

    def _state_key(self) -> str:
        return f"{self._key_prefix}:state"

    def _failures_key(self) -> str:
        return f"{self._key_prefix}:failures"

    def _last_failure_key(self) -> str:
        return f"{self._key_prefix}:last_failure"

    async def _get_state(self) -> CircuitState:
        if rds is None:
            return CircuitState(self.CLOSED, 0, 0, 0)

        try:
            pipe = rds.pipeline()
            pipe.get(self._state_key())
            pipe.get(self._failures_key())
            pipe.get(self._last_failure_key())
            results = pipe.execute()

            state = results[0] or self.CLOSED
            failures = int(results[1] or 0)
            last_failure = float(results[2] or 0)

            return CircuitState(state, failures, last_failure, 0)
        except Exception as e:
            log.warning("Circuit breaker state read failed: %r", e)
            return CircuitState(self.CLOSED, 0, 0, 0)

    async def _set_state(self, state: str, failures: int = 0):
        if rds is None:
            return

        try:
            pipe = rds.pipeline()
            pipe.set(self._state_key(), state, ex=self.timeout * 2)
            pipe.set(self._failures_key(), str(failures), ex=self.timeout * 2)
            if state == self.OPEN:
                pipe.set(self._last_failure_key(), str(time.time()), ex=self.timeout * 2)
            pipe.execute()
        except Exception as e:
            log.warning("Circuit breaker state write failed: %r", e)

    async def _record_success(self):
        if rds is None:
            return

        try:
            pipe = rds.pipeline()
            pipe.set(self._state_key(), self.CLOSED, ex=self.timeout * 2)
            pipe.set(self._failures_key(), "0", ex=self.timeout * 2)
            pipe.execute()
        except Exception as e:
            log.warning("Circuit breaker success record failed: %r", e)

    async def _record_failure(self):
        if rds is None:
            return

        try:
            failures = rds.incr(self._failures_key())
            rds.expire(self._failures_key(), self.timeout * 2)

            if failures >= self.threshold:
                log.warning(
                    "Circuit breaker OPENING after %d failures (threshold=%d)",
                    failures,
                    self.threshold,
                )
                await self._set_state(self.OPEN, failures)
        except Exception as e:
            log.warning("Circuit breaker failure record failed: %r", e)

    def _should_attempt_reset(self, state: CircuitState) -> bool:
        if state.state != self.OPEN:
            return False
        elapsed = time.time() - state.last_failure_time
        return elapsed >= self.timeout

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if not CIRCUIT_BREAKER_ENABLED:
            return await func(*args, **kwargs)

        state = await self._get_state()

        if state.state == self.OPEN:
            if self._should_attempt_reset(state):
                log.info("Circuit breaker attempting HALF_OPEN probe")
                await self._set_state(self.HALF_OPEN, state.failure_count)
                return await self._execute_with_tracking(func, *args, **kwargs)
            else:
                retry_after = int(self.timeout - (time.time() - state.last_failure_time))
                raise CircuitOpenError(retry_after=max(1, retry_after))

        if state.state == self.HALF_OPEN:
            return await self._execute_with_tracking(func, *args, **kwargs)

        return await self._execute_with_tracking(func, *args, **kwargs)

    async def _execute_with_tracking(self, func: Callable, *args, **kwargs) -> Any:
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            if self._is_transient_error(e):
                await self._record_failure()
            raise

    def _is_transient_error(self, e: Exception) -> bool:
        status = getattr(e, "status_code", None) or getattr(
            getattr(e, "response", None), "status_code", None
        )
        if status in (429, 500, 502, 503, 504):
            return True
        err_str = str(e).lower()
        return any(
            kw in err_str
            for kw in ["timeout", "connection", "overloaded", "rate limit"]
        )


default_circuit_breaker = CircuitBreaker()


async def with_circuit_breaker(func: Callable, *args, **kwargs) -> Any:
    return await default_circuit_breaker.call(func, *args, **kwargs)
