import time
from typing import Optional, Tuple
from dataclasses import dataclass

from gateway.config import RATE_LIMIT_ENABLED, RATE_LIMIT_REQUESTS_PER_MINUTE
from gateway.cache import rds
from gateway.logging_setup import log
from gateway.metrics import record_rate_limit_hit


@dataclass
class RateLimitResult:
    allowed: bool
    current_count: int
    limit: int
    remaining: int
    reset_after: int
    retry_after: Optional[int] = None


class RateLimiter:
    def __init__(
        self,
        default_rpm: int = RATE_LIMIT_REQUESTS_PER_MINUTE,
        window_seconds: int = 60,
    ):
        self.default_rpm = default_rpm
        self.window_seconds = window_seconds

    def _key(self, project_id: str) -> str:
        return f"ratelimit:{project_id}"

    async def check(
        self,
        project_id: str,
        limit_override: Optional[int] = None,
    ) -> RateLimitResult:
        if not RATE_LIMIT_ENABLED:
            return RateLimitResult(
                allowed=True,
                current_count=0,
                limit=self.default_rpm,
                remaining=self.default_rpm,
                reset_after=0,
            )

        if rds is None:
            return RateLimitResult(
                allowed=True,
                current_count=0,
                limit=self.default_rpm,
                remaining=self.default_rpm,
                reset_after=0,
            )

        limit = limit_override or self.default_rpm
        key = self._key(project_id)
        now = time.time()
        window_start = now - self.window_seconds

        try:
            pipe = rds.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, self.window_seconds * 2)
            results = pipe.execute()

            current_count = results[2]
            remaining = max(0, limit - current_count)
            reset_after = self.window_seconds

            if current_count > limit:
                record_rate_limit_hit(project_id)
                log.warning(
                    "Rate limit exceeded for project=%s count=%d limit=%d",
                    project_id,
                    current_count,
                    limit,
                )

                oldest_entries = rds.zrange(key, 0, 0, withscores=True)
                if oldest_entries:
                    oldest_time = oldest_entries[0][1]
                    retry_after = int((oldest_time + self.window_seconds) - now) + 1
                    retry_after = max(1, retry_after)
                else:
                    retry_after = self.window_seconds

                return RateLimitResult(
                    allowed=False,
                    current_count=current_count,
                    limit=limit,
                    remaining=0,
                    reset_after=reset_after,
                    retry_after=retry_after,
                )

            return RateLimitResult(
                allowed=True,
                current_count=current_count,
                limit=limit,
                remaining=remaining,
                reset_after=reset_after,
            )

        except Exception as e:
            log.warning("Rate limit check failed, allowing request: %r", e)
            return RateLimitResult(
                allowed=True,
                current_count=0,
                limit=limit,
                remaining=limit,
                reset_after=0,
            )

    async def get_usage(self, project_id: str) -> Tuple[int, int]:
        if rds is None:
            return 0, self.default_rpm

        key = self._key(project_id)
        now = time.time()
        window_start = now - self.window_seconds

        try:
            rds.zremrangebyscore(key, 0, window_start)
            count = rds.zcard(key)
            return count, self.default_rpm
        except Exception:
            return 0, self.default_rpm

    async def reset(self, project_id: str):
        if rds is None:
            return

        key = self._key(project_id)
        try:
            rds.delete(key)
        except Exception as e:
            log.warning("Failed to reset rate limit for %s: %r", project_id, e)


default_rate_limiter = RateLimiter()


async def check_rate_limit(
    project_id: str,
    limit_override: Optional[int] = None,
) -> RateLimitResult:
    return await default_rate_limiter.check(project_id, limit_override)


def get_rate_limit_headers(result: RateLimitResult) -> dict:
    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        "X-RateLimit-Reset": str(result.reset_after),
    }
    if result.retry_after is not None:
        headers["Retry-After"] = str(result.retry_after)
    return headers
