"""
Webhook dispatch module.

Provides fire_webhook() for signed HTTP delivery and dispatch() which fans
out to every matching webhook (project-specific + global fallback from config).
"""
import asyncio
import hashlib
import hmac
import json
import logging
import fnmatch
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from gateway import config

log = logging.getLogger("gateway")


async def fire_webhook(url: str, payload: Dict[str, Any], secret: Optional[str] = None) -> None:
    """
    POST *payload* as JSON to *url*, optionally signing with HMAC-SHA256.

    The signature header is ``X-Gateway-Signature: sha256=<hex>`` — compatible
    with GitHub-style webhook verification.  All errors are swallowed so that a
    broken webhook endpoint never affects the request path.
    """
    try:
        body = json.dumps(payload, default=str).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AI-Gateway-Webhook/1.0",
            "X-Gateway-Event": payload.get("event", "unknown"),
        }
        if secret:
            sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            headers["X-Gateway-Signature"] = f"sha256={sig}"

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, content=body, headers=headers)
            if resp.status_code >= 400:
                log.debug("Webhook %s → HTTP %d", url, resp.status_code)
    except Exception as exc:
        log.debug("Webhook delivery failed for %s: %r", url, exc)


async def dispatch(event: str, data: Dict[str, Any], project_id: Optional[int] = None) -> None:
    """
    Dispatch an event to all matching webhooks, then fire each as a background task.

    Lookup order:
    1. Project-specific webhooks where project_id matches (if provided)
    2. Global webhooks (project_id IS NULL in DB)
    3. Global fallback env var WEBHOOK_URL (config.WEBHOOK_URL)

    Each webhook's ``events`` field is a comma-separated list of glob patterns
    (e.g. ``"*"``, ``"request.*,error.*"``).  A webhook fires if any pattern matches.
    """
    payload = {
        "event": event,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "project_id": project_id,
        "data": data,
    }

    fired_urls: List[str] = []

    # DB webhooks
    try:
        from gateway.db import get_session, Webhook, db_ready
        from sqlalchemy import select, or_

        if db_ready:
            async with get_session() as session:
                conditions = [Webhook.project_id.is_(None)]
                if project_id is not None:
                    conditions.append(Webhook.project_id == project_id)

                rows = (await session.execute(
                    select(Webhook).where(
                        Webhook.is_active == True,  # noqa: E712
                        or_(*conditions),
                    )
                )).scalars().all()

                for wh in rows:
                    patterns = [p.strip() for p in wh.events.split(",") if p.strip()]
                    if any(fnmatch.fnmatch(event, pat) for pat in patterns):
                        asyncio.create_task(fire_webhook(wh.url, payload, secret=wh.secret or None))
                        fired_urls.append(wh.url)
    except Exception as exc:
        log.debug("dispatch DB lookup failed: %r", exc)

    # Global env-var fallback (fire if not already triggered by a DB entry with same URL)
    fallback_url = getattr(config, "WEBHOOK_URL", "")
    if fallback_url and fallback_url not in fired_urls:
        fallback_secret = getattr(config, "WEBHOOK_SECRET", "") or None
        asyncio.create_task(fire_webhook(fallback_url, payload, secret=fallback_secret))
