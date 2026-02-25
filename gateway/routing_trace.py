"""Routing trace event storage for dashboard observability."""

from __future__ import annotations

import json
import time
import inspect
from typing import Any, Dict, List

from gateway.cache import rds
from gateway.logging_setup import log

_TRACE_KEY = "dashboard:routing_trace"
_MAX_STORED = 200


async def record_routing_trace_event(event: Dict[str, Any]):
    """Record a routing trace event in Redis list."""
    if rds is None:
        return

    payload = dict(event or {})
    payload.setdefault("timestamp", int(time.time()))

    try:
        rds.lpush(_TRACE_KEY, json.dumps(payload, ensure_ascii=False))
        rds.ltrim(_TRACE_KEY, 0, _MAX_STORED - 1)
    except Exception as exc:
        log.debug("Failed to record routing trace: %r", exc)


async def get_recent_routing_trace(limit: int = 25) -> List[Dict[str, Any]]:
    """Get recent routing trace events."""
    if rds is None:
        return []

    try:
        raw_items = rds.lrange(_TRACE_KEY, 0, max(0, min(limit, _MAX_STORED) - 1))
        if inspect.isawaitable(raw_items):
            raw_items = await raw_items
        items: List[Dict[str, Any]] = []
        for raw in raw_items:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="ignore")
                if not isinstance(raw, str):
                    continue
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    items.append(parsed)
            except Exception:
                continue
        return items
    except Exception as exc:
        log.debug("Failed to read routing trace: %r", exc)
        return []
