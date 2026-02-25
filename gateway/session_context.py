"""Session context persistence for cross-provider continuity."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from gateway.cache import rds
from gateway.canonical_format import CanonicalBlock, CanonicalMessage, canonical_to_openai_messages, to_canonical_messages
from gateway.config import CACHE_TTL_SECONDS
from gateway.logging_setup import log


def _session_key(session_id: str) -> str:
    return f"session_ctx:{session_id}"


def _serialize_canonical(messages: List[CanonicalMessage]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for message in messages:
        payload.append(
            {
                "role": message.role,
                "content": [asdict(block) for block in message.content],
            }
        )
    return payload


def _deserialize_canonical(raw_messages: List[Dict[str, Any]]) -> List[CanonicalMessage]:
    output: List[CanonicalMessage] = []
    for message in raw_messages or []:
        blocks = [CanonicalBlock(**block) for block in message.get("content", []) if isinstance(block, dict)]
        output.append(CanonicalMessage(role=message.get("role", "user"), content=blocks))
    return output


def _dedupe_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for message in messages:
        if deduped and deduped[-1] == message:
            continue
        deduped.append(message)
    return deduped


async def load_session_context(session_id: str) -> Dict[str, Any]:
    """Load stored context for a session id."""
    if not session_id or rds is None:
        return {}

    try:
        raw = rds.get(_session_key(session_id))
        if not raw:
            return {}
        if isinstance(raw, bytes):
            raw_text = raw.decode("utf-8", errors="ignore")
        elif isinstance(raw, str):
            raw_text = raw
        else:
            return {}

        payload = json.loads(raw_text)
        canonical_messages = _deserialize_canonical(payload.get("messages", []))
        return {
            "messages": canonical_to_openai_messages(canonical_messages),
            "system_prompt": payload.get("system_prompt", ""),
            "last_provider": payload.get("last_provider"),
            "last_model": payload.get("last_model"),
            "updated_at": payload.get("updated_at"),
        }
    except Exception as exc:
        log.warning("Failed to load session context for %s: %r", session_id, exc)
        return {}


async def merge_session_context(
    session_id: str,
    incoming_messages: List[Dict[str, Any]],
    incoming_system_prompt: str = "",
    max_messages: int = 60,
) -> Tuple[List[Dict[str, Any]], str]:
    """Merge incoming turn with stored session history."""
    stored = await load_session_context(session_id)
    stored_messages = stored.get("messages") or []
    stored_system = stored.get("system_prompt") or ""

    if not stored_messages:
        return incoming_messages, incoming_system_prompt or stored_system

    merged = _dedupe_messages(list(stored_messages) + list(incoming_messages or []))
    if len(merged) > max_messages:
        merged = merged[-max_messages:]

    system_prompt = incoming_system_prompt or stored_system
    return merged, system_prompt


async def persist_session_context(
    session_id: str,
    messages: List[Dict[str, Any]],
    system_prompt: str = "",
    last_provider: Optional[str] = None,
    last_model: Optional[str] = None,
    ttl_seconds: int = CACHE_TTL_SECONDS,
    max_messages: int = 80,
):
    """Persist canonicalized session context to Redis."""
    if not session_id or rds is None:
        return

    try:
        trimmed = list(messages or [])[-max_messages:]
        canonical_messages = to_canonical_messages(trimmed)
        payload = {
            "messages": _serialize_canonical(canonical_messages),
            "system_prompt": system_prompt or "",
            "last_provider": last_provider,
            "last_model": last_model,
            "updated_at": int(time.time()),
        }
        rds.setex(_session_key(session_id), ttl_seconds, json.dumps(payload))
    except Exception as exc:
        log.warning("Failed to persist session context for %s: %r", session_id, exc)


async def persist_session_turn(
    session_id: str,
    input_messages: List[Dict[str, Any]],
    assistant_content: str,
    system_prompt: str = "",
    last_provider: Optional[str] = None,
    last_model: Optional[str] = None,
):
    """Persist a complete turn by appending assistant output to provided input messages."""
    if not session_id:
        return

    output_messages = list(input_messages or [])
    if assistant_content:
        output_messages.append({"role": "assistant", "content": assistant_content})

    await persist_session_context(
        session_id=session_id,
        messages=output_messages,
        system_prompt=system_prompt,
        last_provider=last_provider,
        last_model=last_model,
    )
