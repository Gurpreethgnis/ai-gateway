from typing import Optional, Callable
from functools import wraps
import time

from gateway.config import PROMETHEUS_ENABLED
from gateway.logging_setup import log

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    log.warning("prometheus_client not installed, metrics disabled")


if PROMETHEUS_AVAILABLE and PROMETHEUS_ENABLED:
    REQUEST_COUNT = Counter(
        "gateway_requests_total",
        "Total number of requests",
        ["model", "project", "status", "endpoint"],
    )

    REQUEST_LATENCY = Histogram(
        "gateway_request_latency_seconds",
        "Request latency in seconds",
        ["model", "project", "endpoint"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )

    TOKEN_USAGE = Counter(
        "gateway_tokens_total",
        "Total token usage",
        ["model", "project", "type"],
    )

    COST_USD = Counter(
        "gateway_cost_usd_total",
        "Total cost in USD",
        ["model", "project"],
    )

    ACTIVE_REQUESTS = Gauge(
        "gateway_active_requests",
        "Number of currently active requests",
        ["model"],
    )

    CIRCUIT_STATE = Gauge(
        "gateway_circuit_state",
        "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        ["upstream"],
    )

    CACHE_HITS = Counter(
        "gateway_cache_hits_total",
        "Total cache hits",
        ["cache_type"],
    )

    CACHE_MISSES = Counter(
        "gateway_cache_misses_total",
        "Total cache misses",
        ["cache_type"],
    )

    RATE_LIMIT_HITS = Counter(
        "gateway_rate_limit_hits_total",
        "Total rate limit hits",
        ["project"],
    )

    RETRY_COUNT = Counter(
        "gateway_retry_total",
        "Total retry attempts",
        ["model", "attempt"],
    )

    UPSTREAM_ERRORS = Counter(
        "gateway_upstream_errors_total",
        "Total upstream errors",
        ["model", "error_type", "status_code"],
    )

    STREAM_DURATION = Histogram(
        "gateway_stream_duration_seconds",
        "Streaming response duration",
        ["model", "project"],
        buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )

    # Token savings metrics - track per mechanism
    TOKENS_SAVED = Counter(
        "gateway_tokens_saved_total",
        "Tokens saved by reduction mechanism",
        ["mechanism", "project"],
    )

    PROMPT_CACHE_TOKENS = Counter(
        "gateway_prompt_cache_tokens_total",
        "Prompt cache tokens (read=cache hit, write=cache miss)",
        ["type", "model", "project"],  # type: read or write
    )

    PROVIDER_CACHE_EVENTS = Counter(
        "gateway_provider_cache_events_total",
        "Provider cache events by provider/cache_type/hit",
        ["provider", "cache_type", "hit"],
    )

    PROVIDER_CACHE_TOKENS = Counter(
        "gateway_provider_cache_tokens_total",
        "Provider cache token accounting",
        ["provider", "cache_type"],
    )

    GATEWAY_INFO = Info(
        "gateway",
        "Gateway information",
    )
    GATEWAY_INFO.info({"version": "1.0.0", "name": "ai-gateway"})

else:
    class DummyMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, amount=1):
            pass

        def dec(self, amount=1):
            pass

        def set(self, value):
            pass

        def observe(self, value):
            pass

        def info(self, data):
            pass

    REQUEST_COUNT = DummyMetric()
    REQUEST_LATENCY = DummyMetric()
    TOKEN_USAGE = DummyMetric()
    COST_USD = DummyMetric()
    ACTIVE_REQUESTS = DummyMetric()
    CIRCUIT_STATE = DummyMetric()
    CACHE_HITS = DummyMetric()
    CACHE_MISSES = DummyMetric()
    RATE_LIMIT_HITS = DummyMetric()
    RETRY_COUNT = DummyMetric()
    UPSTREAM_ERRORS = DummyMetric()
    STREAM_DURATION = DummyMetric()
    TOKENS_SAVED = DummyMetric()
    PROMPT_CACHE_TOKENS = DummyMetric()
    PROVIDER_CACHE_EVENTS = DummyMetric()
    PROVIDER_CACHE_TOKENS = DummyMetric()
    GATEWAY_INFO = DummyMetric()


def record_request(
    model: str,
    project: str,
    status: int,
    endpoint: str,
    latency_seconds: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    cached: bool = False,
):
    project = project or "default"

    REQUEST_COUNT.labels(
        model=model,
        project=project,
        status=str(status),
        endpoint=endpoint,
    ).inc()

    REQUEST_LATENCY.labels(
        model=model,
        project=project,
        endpoint=endpoint,
    ).observe(latency_seconds)

    if input_tokens > 0:
        TOKEN_USAGE.labels(model=model, project=project, type="input").inc(input_tokens)
    if output_tokens > 0:
        TOKEN_USAGE.labels(model=model, project=project, type="output").inc(output_tokens)

    if cost_usd > 0:
        COST_USD.labels(model=model, project=project).inc(cost_usd)

    if cached:
        CACHE_HITS.labels(cache_type="response").inc()
    else:
        CACHE_MISSES.labels(cache_type="response").inc()


def record_stream_duration(model: str, project: str, duration_seconds: float):
    project = project or "default"
    STREAM_DURATION.labels(model=model, project=project).observe(duration_seconds)


def record_cache_hit(cache_type: str):
    CACHE_HITS.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str):
    CACHE_MISSES.labels(cache_type=cache_type).inc()


def record_rate_limit_hit(project: str):
    project = project or "default"
    RATE_LIMIT_HITS.labels(project=project).inc()


def record_retry(model: str, attempt: int):
    RETRY_COUNT.labels(model=model, attempt=str(attempt)).inc()


def record_upstream_error(model: str, error_type: str, status_code: int):
    UPSTREAM_ERRORS.labels(
        model=model,
        error_type=error_type,
        status_code=str(status_code),
    ).inc()


def set_circuit_state(upstream: str, state: str):
    state_map = {"closed": 0, "open": 1, "half_open": 2}
    CIRCUIT_STATE.labels(upstream=upstream).set(state_map.get(state, 0))


def set_active_requests(model: str, count: int):
    ACTIVE_REQUESTS.labels(model=model).set(count)


def increment_active_requests(model: str):
    ACTIVE_REQUESTS.labels(model=model).inc()


def decrement_active_requests(model: str):
    ACTIVE_REQUESTS.labels(model=model).dec()


def record_tokens_saved(mechanism: str, project: str, tokens: int):
    """
    Record tokens saved by a specific mechanism.
    mechanism: 'prompt_cache', 'file_dedup', 'diff_first', 'context_pruning', 'boilerplate_strip'
    """
    project = project or "default"
    TOKENS_SAVED.labels(mechanism=mechanism, project=project).inc(tokens)


def record_prompt_cache_tokens(cache_type: str, model: str, project: str, tokens: int):
    """
    Record prompt cache token usage.
    cache_type: 'read' (cache hit, ~10% cost) or 'write' (cache miss, full cost)
    """
    project = project or "default"
    PROMPT_CACHE_TOKENS.labels(type=cache_type, model=model, project=project).inc(tokens)


def record_provider_cache_event(provider: str, cache_type: str, hit: bool, tokens: int = 0):
    """Record provider-specific cache event and optional token usage."""
    provider = provider or "unknown"
    cache_type = cache_type or "unknown"
    PROVIDER_CACHE_EVENTS.labels(provider=provider, cache_type=cache_type, hit="1" if hit else "0").inc()
    if tokens > 0:
        PROVIDER_CACHE_TOKENS.labels(provider=provider, cache_type=cache_type).inc(tokens)


def get_metrics_output() -> bytes:
    if PROMETHEUS_AVAILABLE and PROMETHEUS_ENABLED:
        return generate_latest()
    return b"# Prometheus metrics disabled\n"


def get_metrics_content_type() -> str:
    if PROMETHEUS_AVAILABLE and PROMETHEUS_ENABLED:
        return CONTENT_TYPE_LATEST
    return "text/plain"


def track_request_time(model: str, project: str, endpoint: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            increment_active_requests(model)
            try:
                result = await func(*args, **kwargs)
                status = 200
                return result
            except Exception as e:
                status = getattr(e, "status_code", 500)
                raise
            finally:
                decrement_active_requests(model)
                latency = time.time() - start
                record_request(
                    model=model,
                    project=project,
                    status=status,
                    endpoint=endpoint,
                    latency_seconds=latency,
                )

        return wrapper

    return decorator
