"""
Canonical message format for cross-provider normalization.

This module defines a provider-agnostic, typed representation for chat messages
and helpers to convert between canonical and provider-oriented formats.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$")


@dataclass
class CanonicalBlock:
    """A single canonical content block."""

    type: str  # text | image | tool_use | tool_result | opaque
    text: Optional[str] = None
    mime_type: Optional[str] = None
    data: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    arguments: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class CanonicalMessage:
    """Canonical message with typed blocks."""

    role: str  # system | user | assistant
    content: List[CanonicalBlock] = field(default_factory=list)


def _safe_json_dumps(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except Exception:
        return "{}"


def _normalize_role(role: Any) -> str:
    role_str = str(role or "user")
    if role_str in {"system", "user", "assistant"}:
        return role_str
    return "user"


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, (dict, list)):
        try:
            return json.dumps(content)
        except Exception:
            return str(content)
    return str(content)


def _parse_data_url(url: str) -> Optional[CanonicalBlock]:
    match = _DATA_URL_RE.match(url or "")
    if not match:
        return None
    return CanonicalBlock(type="image", mime_type=match.group(1), data=match.group(2))


def _content_to_blocks(content: Any) -> List[CanonicalBlock]:
    blocks: List[CanonicalBlock] = []

    if isinstance(content, str):
        return [CanonicalBlock(type="text", text=content)]

    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                blocks.append(CanonicalBlock(type="text", text=item))
                continue

            if not isinstance(item, dict):
                blocks.append(CanonicalBlock(type="text", text=str(item)))
                continue

            block_type = item.get("type")
            if block_type == "text":
                blocks.append(CanonicalBlock(type="text", text=item.get("text", "")))
            elif block_type == "image":
                source = item.get("source", {})
                blocks.append(
                    CanonicalBlock(
                        type="image",
                        mime_type=source.get("media_type", "image/png"),
                        data=source.get("data", ""),
                    )
                )
            elif block_type == "image_url":
                image_url = (item.get("image_url") or {}).get("url", "")
                parsed = _parse_data_url(image_url)
                if parsed:
                    blocks.append(parsed)
                else:
                    blocks.append(CanonicalBlock(type="opaque", raw=item))
            elif block_type == "tool_use":
                blocks.append(
                    CanonicalBlock(
                        type="tool_use",
                        tool_call_id=item.get("id", ""),
                        tool_name=item.get("name", ""),
                        arguments=_safe_json_dumps(item.get("input", {})),
                    )
                )
            elif block_type == "tool_result":
                blocks.append(
                    CanonicalBlock(
                        type="tool_result",
                        tool_call_id=item.get("tool_use_id") or item.get("tool_call_id") or "",
                        text=_stringify_content(item.get("content", "")),
                    )
                )
            else:
                blocks.append(CanonicalBlock(type="opaque", raw=item))

        return blocks

    return [CanonicalBlock(type="text", text=_stringify_content(content))]


def to_canonical_messages(messages: List[Dict[str, Any]]) -> List[CanonicalMessage]:
    """Convert mixed provider/OpenAI-style messages into canonical format."""
    canonical: List[CanonicalMessage] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool":
            canonical.append(
                CanonicalMessage(
                    role="user",
                    content=[
                        CanonicalBlock(
                            type="tool_result",
                            tool_call_id=msg.get("tool_call_id", ""),
                            text=_stringify_content(content),
                        )
                    ],
                )
            )
            continue

        blocks = _content_to_blocks(content)

        if role == "assistant" and msg.get("tool_calls"):
            for tool_call in msg.get("tool_calls", []):
                function = tool_call.get("function", {})
                blocks.append(
                    CanonicalBlock(
                        type="tool_use",
                        tool_call_id=tool_call.get("id", ""),
                        tool_name=function.get("name", ""),
                        arguments=function.get("arguments", "{}"),
                    )
                )

        canonical.append(CanonicalMessage(role=_normalize_role(role), content=blocks))

    return canonical


def canonical_to_openai_messages(messages: List[CanonicalMessage]) -> List[Dict[str, Any]]:
    """Convert canonical messages into OpenAI-compatible message format."""
    result: List[Dict[str, Any]] = []

    for msg in messages:
        role = _normalize_role(msg.role)

        text_parts: List[str] = []
        rich_content: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[CanonicalBlock] = []

        for block in msg.content:
            if block.type == "text":
                value = block.text or ""
                text_parts.append(value)
                rich_content.append({"type": "text", "text": value})
            elif block.type == "image" and block.data:
                mime_type = block.mime_type or "image/png"
                rich_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{block.data}"},
                    }
                )
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.tool_call_id or "",
                        "type": "function",
                        "function": {
                            "name": block.tool_name or "",
                            "arguments": block.arguments or "{}",
                        },
                    }
                )
            elif block.type == "tool_result":
                tool_results.append(block)
            elif block.type == "opaque" and block.raw:
                text_parts.append(_stringify_content(block.raw))

        if role == "user" and tool_results and not text_parts and not rich_content:
            for tool_result in tool_results:
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.tool_call_id or "",
                        "content": tool_result.text or "",
                    }
                )
            continue

        message_content: Any
        if rich_content and any(part.get("type") == "image_url" for part in rich_content):
            message_content = rich_content
        else:
            message_content = "\n".join([part for part in text_parts if part])

        message_payload: Dict[str, Any] = {"role": role, "content": message_content}
        if role == "assistant" and tool_calls:
            message_payload["tool_calls"] = tool_calls
        result.append(message_payload)

        if role == "user" and tool_results:
            for tool_result in tool_results:
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.tool_call_id or "",
                        "content": tool_result.text or "",
                    }
                )

    return result


def canonical_to_anthropic_messages(messages: List[CanonicalMessage]) -> List[Dict[str, Any]]:
    """Convert canonical messages into Anthropic-compatible message format."""
    result: List[Dict[str, Any]] = []

    for msg in messages:
        role = _normalize_role(msg.role)
        if role == "system":
            continue

        blocks: List[Dict[str, Any]] = []
        for block in msg.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text or ""})
            elif block.type == "image" and block.data:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.mime_type or "image/png",
                            "data": block.data,
                        },
                    }
                )
            elif block.type == "tool_use":
                args: Any
                try:
                    args = json.loads(block.arguments or "{}")
                except Exception:
                    args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.tool_call_id or "",
                        "name": block.tool_name or "",
                        "input": args,
                    }
                )
            elif block.type == "tool_result":
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_call_id or "",
                        "content": block.text or "",
                    }
                )
            elif block.type == "opaque" and block.raw:
                blocks.append({"type": "text", "text": _stringify_content(block.raw)})

        if not blocks:
            blocks = [{"type": "text", "text": ""}]

        result.append({"role": role, "content": blocks})

    return result


def canonical_to_text_messages(
    messages: List[CanonicalMessage],
    include_tool_results: bool = True,
    include_tool_use: bool = False,
) -> List[Dict[str, str]]:
    """Convert canonical messages to plain text message content."""
    result: List[Dict[str, str]] = []

    for msg in messages:
        role = _normalize_role(msg.role)
        text_parts: List[str] = []

        for block in msg.content:
            if block.type == "text" and block.text:
                text_parts.append(block.text)
            elif block.type == "tool_result" and include_tool_results:
                text_parts.append(f"[Tool result]: {block.text or ''}")
            elif block.type == "tool_use" and include_tool_use:
                tool_name = block.tool_name or "tool"
                text_parts.append(f"[Tool call]: {tool_name}")
            elif block.type == "image":
                text_parts.append("[Image omitted]")
            elif block.type == "opaque" and block.raw:
                text_parts.append(_stringify_content(block.raw))

        content = "\n".join([part for part in text_parts if part]).strip()
        if content:
            result.append({"role": role, "content": content})

    return result