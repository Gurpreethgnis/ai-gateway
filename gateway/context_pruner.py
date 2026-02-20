import re
from typing import List, Dict, Any, Optional, Tuple

from gateway.config import ENABLE_CONTEXT_PRUNING, CONTEXT_MAX_TOKENS
from gateway.logging_setup import log


CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    total = 4

    content = message.get("content", "")
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        total += estimate_tokens(result_content)
                    elif isinstance(result_content, list):
                        for rc in result_content:
                            if isinstance(rc, dict) and rc.get("type") == "text":
                                total += estimate_tokens(rc.get("text", ""))

    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                total += estimate_tokens(fn.get("name", ""))
                total += estimate_tokens(fn.get("arguments", ""))

    return total


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def summarize_message(message: Dict[str, Any], max_chars: int = 200) -> str:
    role = message.get("role", "unknown")
    content = message.get("content", "")

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    texts.append("[tool_result]")
                elif block.get("type") == "tool_use":
                    texts.append(f"[tool:{block.get('name', 'unknown')}]")
        text = " ".join(texts)
    else:
        text = str(content)

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."

    return f"{role}: {text}"


def create_summary_block(messages: List[Dict[str, Any]]) -> str:
    summaries = []
    for msg in messages[-10:]:
        summaries.append(summarize_message(msg, max_chars=150))

    if len(messages) > 10:
        summaries.insert(0, f"[...{len(messages) - 10} earlier messages omitted...]")

    return "\n".join(summaries)


def find_important_messages(messages: List[Dict[str, Any]]) -> List[int]:
    important_indices = set()

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = str(msg.get("content", "")).lower()

        if role in ("system", "developer"):
            important_indices.add(i)
            continue

        if any(kw in content for kw in [
            "error", "bug", "fix", "issue", "problem",
            "requirement", "must", "critical", "important",
        ]):
            important_indices.add(i)

        if msg.get("tool_calls"):
            important_indices.add(i)

    return sorted(important_indices)


async def prune_context(
    messages: List[Dict[str, Any]],
    max_tokens: int = CONTEXT_MAX_TOKENS,
    system_text: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not ENABLE_CONTEXT_PRUNING:
        return messages, {"pruned": False, "reason": "disabled"}

    system_tokens = estimate_tokens(system_text)
    message_tokens = estimate_messages_tokens(messages)
    total_tokens = system_tokens + message_tokens

    if total_tokens <= max_tokens:
        return messages, {
            "pruned": False,
            "total_tokens": total_tokens,
            "max_tokens": max_tokens,
        }

    log.info(
        "Context pruning: %d tokens > %d max, pruning...",
        total_tokens,
        max_tokens,
    )

    pruned = []
    meta = {
        "pruned": True,
        "original_messages": len(messages),
        "original_tokens": total_tokens,
    }

    system_messages = []
    regular_messages = []

    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            system_messages.append(msg)
        else:
            regular_messages.append(msg)

    pruned.extend(system_messages)

    keep_recent = 10
    recent_messages = regular_messages[-keep_recent:] if len(regular_messages) > keep_recent else regular_messages

    if len(regular_messages) > keep_recent:
        dropped = regular_messages[:-keep_recent]

        important = find_important_messages(dropped)
        important_msgs = [dropped[i] for i in important if i < len(dropped)]

        summary_text = create_summary_block(dropped)

        if estimate_tokens(summary_text) < estimate_messages_tokens(dropped) // 2:
            pruned.append({
                "role": "user",
                "content": f"[Previous conversation summary ({len(dropped)} messages):\n{summary_text}]",
            })

            for imp_msg in important_msgs[:3]:
                pruned.append(imp_msg)

            meta["summarized_messages"] = len(dropped)
            meta["kept_important"] = len(important_msgs[:3])

    pruned.extend(recent_messages)

    final_tokens = estimate_messages_tokens(pruned) + system_tokens

    if final_tokens > max_tokens:
        log.warning(
            "Context still too large after pruning: %d tokens (max %d)",
            final_tokens,
            max_tokens,
        )

        while len(pruned) > 3 and estimate_messages_tokens(pruned) + system_tokens > max_tokens:
            for i, msg in enumerate(pruned):
                if msg.get("role") not in ("system", "developer"):
                    pruned.pop(i)
                    break

    meta["final_messages"] = len(pruned)
    meta["final_tokens"] = estimate_messages_tokens(pruned) + system_tokens
    meta["tokens_saved"] = total_tokens - meta["final_tokens"]

    log.info(
        "Context pruned: %d -> %d messages, %d -> %d tokens (saved %d)",
        meta["original_messages"],
        meta["final_messages"],
        meta["original_tokens"],
        meta["final_tokens"],
        meta["tokens_saved"],
    )

    return pruned, meta


def truncate_long_content(
    messages: List[Dict[str, Any]],
    max_content_chars: int = 50000,
) -> List[Dict[str, Any]]:
    result = []

    for msg in messages:
        content = msg.get("content")

        if isinstance(content, str) and len(content) > max_content_chars:
            head = content[: int(max_content_chars * 0.7)]
            tail = content[-int(max_content_chars * 0.3):]
            truncated = head + "\n\n[...TRUNCATED...]\n\n" + tail

            result.append({**msg, "content": truncated})
        else:
            result.append(msg)

    return result
