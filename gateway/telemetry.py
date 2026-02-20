import json
import traceback
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

from gateway.cache import rds
from gateway.logging_setup import log


@dataclass
class ErrorEvent:
    timestamp: str
    error_type: str
    message: str
    cf_ray: str
    model: str
    project_id: Optional[str]
    stack_trace: str
    upstream_status: Optional[int]
    request_path: str
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RequestEvent:
    timestamp: str
    cf_ray: str
    project_id: Optional[str]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    cached: bool
    status_code: int
    has_tools: bool
    stream: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


TELEMETRY_STREAM_KEY = "telemetry:errors"
REQUEST_STREAM_KEY = "telemetry:requests"
MAX_STREAM_LENGTH = 10000


async def emit_error(
    error_type: str,
    message: str,
    cf_ray: str = "",
    model: str = "",
    project_id: Optional[str] = None,
    upstream_status: Optional[int] = None,
    request_path: str = "",
    exception: Optional[Exception] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    stack_trace = ""
    if exception is not None:
        stack_trace = traceback.format_exc()

    event = ErrorEvent(
        timestamp=datetime.utcnow().isoformat(),
        error_type=error_type,
        message=message[:2000],
        cf_ray=cf_ray,
        model=model,
        project_id=project_id,
        stack_trace=stack_trace[:4000],
        upstream_status=upstream_status,
        request_path=request_path,
        extra=extra,
    )

    log.error(
        "TELEMETRY ERROR type=%s model=%s project=%s status=%s cf-ray=%s: %s",
        error_type,
        model,
        project_id,
        upstream_status,
        cf_ray,
        message[:200],
    )

    if rds is not None:
        try:
            rds.xadd(
                TELEMETRY_STREAM_KEY,
                event.to_dict(),
                maxlen=MAX_STREAM_LENGTH,
            )
        except Exception as e:
            log.warning("Failed to emit error to Redis stream: %r", e)


async def emit_request(
    cf_ray: str,
    project_id: Optional[str],
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
    cached: bool,
    status_code: int,
    has_tools: bool,
    stream: bool,
):
    event = RequestEvent(
        timestamp=datetime.utcnow().isoformat(),
        cf_ray=cf_ray,
        project_id=project_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cached=cached,
        status_code=status_code,
        has_tools=has_tools,
        stream=stream,
    )

    if rds is not None:
        try:
            rds.xadd(
                REQUEST_STREAM_KEY,
                {k: str(v) for k, v in event.to_dict().items()},
                maxlen=MAX_STREAM_LENGTH,
            )
        except Exception as e:
            log.warning("Failed to emit request to Redis stream: %r", e)


async def get_recent_errors(count: int = 100) -> List[Dict[str, Any]]:
    if rds is None:
        return []

    try:
        entries = rds.xrevrange(TELEMETRY_STREAM_KEY, count=count)
        return [
            {"id": entry_id.decode() if isinstance(entry_id, bytes) else entry_id, **{
                k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                for k, v in data.items()
            }}
            for entry_id, data in entries
        ]
    except Exception as e:
        log.warning("Failed to get recent errors: %r", e)
        return []


async def get_error_stats(hours: int = 24) -> Dict[str, Any]:
    errors = await get_recent_errors(1000)

    cutoff = datetime.utcnow().timestamp() - (hours * 3600)

    filtered = []
    for e in errors:
        try:
            ts = datetime.fromisoformat(e.get("timestamp", "")).timestamp()
            if ts >= cutoff:
                filtered.append(e)
        except Exception:
            continue

    by_type: Dict[str, int] = {}
    by_model: Dict[str, int] = {}
    by_status: Dict[str, int] = {}

    for e in filtered:
        error_type = e.get("error_type", "unknown")
        model = e.get("model", "unknown")
        status = e.get("upstream_status", "none")

        by_type[error_type] = by_type.get(error_type, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
        by_status[str(status)] = by_status.get(str(status), 0) + 1

    return {
        "total_errors": len(filtered),
        "hours": hours,
        "by_type": by_type,
        "by_model": by_model,
        "by_status": by_status,
    }
