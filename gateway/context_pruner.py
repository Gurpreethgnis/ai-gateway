import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple

from gateway.config import ENABLE_CONTEXT_PRUNING, CONTEXT_MAX_TOKENS
from gateway.logging_setup import log


CHARS_PER_TOKEN = 4


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(str(block.get("content", "")))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


def score_message_importance(
    message: Dict[str, Any],
    index: int,
    total_messages: int,
) -> float:
    """Score message importance (0.0-1.0) for structured pruning."""
    role = message.get("role", "user")
    text = _message_text(message).lower()

    recency = (index + 1) / max(total_messages, 1)
    score = 0.25 + (0.35 * recency)

    if role in ("system", "developer"):
        score += 0.45

    if message.get("tool_calls"):
        score += 0.25

    if has_tool_result_blocks(message.get("content")) or has_tool_use_blocks(message.get("content")):
        score += 0.20

    priority_keywords = [
        "error", "exception", "traceback", "bug", "fix", "critical", "security",
        "requirement", "must", "constraint", "regression", "failing",
    ]
    if any(keyword in text for keyword in priority_keywords):
        score += 0.20

    if "```" in text or "def " in text or "class " in text:
        score += 0.10

    return max(0.0, min(1.0, score))


def _summary_cache_key(messages: List[Dict[str, Any]]) -> str:
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8", errors="ignore")).hexdigest()
    return f"ctxsummary:{digest}"


def _get_cached_summary(messages: List[Dict[str, Any]]) -> Optional[str]:
    try:
        from gateway.cache import rds
        if rds is None:
            return None
        key = _summary_cache_key(messages)
        cached = rds.get(key)
        return cached if isinstance(cached, str) and cached.strip() else None
    except Exception:
        return None


def _store_cached_summary(messages: List[Dict[str, Any]], summary: str, ttl: int = 1800):
    try:
        from gateway.cache import rds
        if rds is None or not summary:
            return
        key = _summary_cache_key(messages)
        rds.setex(key, ttl, summary)
    except Exception:
        return


def resolve_effective_context_limit(
    max_tokens: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    """Resolve pruning limit using provider/model registry when available."""
    effective = max_tokens or CONTEXT_MAX_TOKENS

    try:
        if not model:
            return effective

        from gateway.model_registry import get_model_registry
        registry = get_model_registry()
        model_info = registry.get_model(model)
        if not model_info and provider and not model.startswith(f"{provider}/"):
            model_info = registry.get_model(f"{provider}/{model}")

        if model_info and getattr(model_info, "context_window", 0):
            provider_limit = int(model_info.context_window * 0.8)
            if provider_limit > 0:
                return min(effective, provider_limit)
    except Exception:
        pass

    return effective


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


def has_tool_result_blocks(content: Any) -> bool:
    """Check if message content contains tool_result blocks."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and (
            block.get("type") == "tool_result" or
            block.get("tool_use_id") or
            block.get("tool_result_for")
        )
        for block in content
    )


def has_tool_use_blocks(content: Any) -> bool:
    """Check if message content contains tool_use blocks."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_use"
        for block in content
    )


def get_tool_use_ids(messages: List[Dict[str, Any]], indices: List[int]) -> set:
    """Get all tool_use IDs from assistant messages at given indices."""
    tool_use_ids = set()
    for i in indices:
        if i >= len(messages):
            continue
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id")
                    if tool_id:
                        tool_use_ids.add(tool_id)
    return tool_use_ids


def get_tool_result_ids(message: Dict[str, Any]) -> set:
    """Get all tool_use_ids referenced in tool_result blocks."""
    result_ids = set()
    content = message.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                tool_id = block.get("tool_use_id") or block.get("tool_result_for")
                if tool_id:
                    result_ids.add(tool_id)
    return result_ids


def find_tool_use_message(messages: List[Dict[str, Any]], tool_use_id: str) -> int:
    """Find the index of the assistant message containing the given tool_use_id."""
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if block.get("id") == tool_use_id:
                        return i
    return -1


def ensure_tool_pairs(
    messages: List[Dict[str, Any]],
    keep_indices: List[int],
) -> List[int]:
    """
    Ensure that for every tool_result in kept messages, the corresponding
    tool_use message is also kept. Returns expanded list of indices.
    """
    keep_set = set(keep_indices)
    
    # For each message we're keeping, check for tool_results
    for i in list(keep_set):
        if i >= len(messages):
            continue
        msg = messages[i]
        result_ids = get_tool_result_ids(msg)
        
        # Find the assistant message with corresponding tool_use
        for tool_id in result_ids:
            tool_use_idx = find_tool_use_message(messages, tool_id)
            if tool_use_idx >= 0 and tool_use_idx not in keep_set:
                keep_set.add(tool_use_idx)
    
    return sorted(keep_set)


def strip_orphaned_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove tool_result blocks that don't have a corresponding tool_use in previous messages.
    This ensures Anthropic API doesn't reject the request.
    """
    # Build set of all tool_use IDs from assistant messages
    available_tool_use_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_id = block.get("id")
                        if tool_id:
                            available_tool_use_ids.add(tool_id)
    
    # Filter messages to remove orphaned tool_results
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", [])
        
        # Only process user messages with list content (potential tool_results)
        if role == "user" and isinstance(content, list):
            # Filter out orphaned tool_result blocks
            filtered_blocks = []
            for block in content:
                if isinstance(block, dict):
                    tool_id = block.get("tool_use_id") or block.get("tool_result_for")
                    if tool_id:
                        # This is a tool_result block - check if tool_use exists
                        if tool_id in available_tool_use_ids:
                            filtered_blocks.append(block)
                        else:
                            log.debug("Stripping orphaned tool_result: %s", tool_id)
                    else:
                        # Not a tool_result block, keep it
                        filtered_blocks.append(block)
                else:
                    filtered_blocks.append(block)
            
            # Only add message if it has content left
            if filtered_blocks:
                result.append({**msg, "content": filtered_blocks})
            else:
                # Replace empty message with placeholder to maintain conversation flow
                result.append({"role": "user", "content": "(previous context)"})
        else:
            result.append(msg)
    
    return result


def strip_orphaned_tool_use(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove tool_use blocks from assistant messages that don't have a corresponding 
    tool_result in the following user messages. This ensures Anthropic API doesn't 
    reject requests with orphaned tool_use blocks.
    """
    # Build set of all tool_result IDs from user messages
    available_tool_result_ids = set()
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        tool_id = block.get("tool_use_id") or block.get("tool_result_for")
                        if tool_id:
                            available_tool_result_ids.add(tool_id)
    
    # Filter messages to remove orphaned tool_use blocks
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", [])
        
        # Only process assistant messages with list content (potential tool_use)
        if role == "assistant" and isinstance(content, list):
            # Filter out orphaned tool_use blocks
            filtered_blocks = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use":
                        tool_id = block.get("id")
                        if tool_id:
                            # This is a tool_use block - check if tool_result exists
                            if tool_id in available_tool_result_ids:
                                filtered_blocks.append(block)
                            else:
                                log.debug("Stripping orphaned tool_use: %s", tool_id)
                        else:
                            # tool_use without ID, keep it (malformed but not our problem)
                            filtered_blocks.append(block)
                    else:
                        # Not a tool_use block, keep it
                        filtered_blocks.append(block)
                else:
                    filtered_blocks.append(block)
            
            # Only add message if it has content left
            if filtered_blocks:
                result.append({**msg, "content": filtered_blocks})
            # If all blocks were removed, skip the message entirely
        else:
            result.append(msg)
    
    return result


async def prune_context(
    messages: List[Dict[str, Any]],
    max_tokens: int = CONTEXT_MAX_TOKENS,
    system_text: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not ENABLE_CONTEXT_PRUNING:
        return messages, {"pruned": False, "reason": "disabled"}

    max_tokens = resolve_effective_context_limit(max_tokens, provider=provider, model=model)

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
    recent_start_idx = max(0, len(regular_messages) - keep_recent)
    recent_indices = list(range(recent_start_idx, len(regular_messages)))
    
    # Ensure tool pairs are preserved for recent messages
    recent_indices = ensure_tool_pairs(regular_messages, recent_indices)
    recent_messages = [regular_messages[i] for i in recent_indices if i < len(regular_messages)]

    if len(regular_messages) > keep_recent:
        # Messages not in recent set
        dropped_indices = [i for i in range(len(regular_messages)) if i not in recent_indices]
        dropped = [regular_messages[i] for i in dropped_indices]

        if dropped:
            # Stage 1: importance scoring for structured retention
            scored = [
                (i, dropped[i], score_message_importance(dropped[i], i, len(dropped)))
                for i in range(len(dropped))
            ]
            scored.sort(key=lambda item: item[2], reverse=True)

            important_indices = sorted([idx for idx, _, score in scored if score >= 0.62][:4])
            important_msgs = [dropped[i] for i in important_indices if i < len(dropped)]

            # Stage 2: summary for lower-importance dropped context
            low_importance_msgs = [m for idx, m, score in scored if score < 0.62]

            summary_text = _get_cached_summary(low_importance_msgs or dropped)
            if not summary_text:
                summary_text = create_summary_block(low_importance_msgs or dropped)
                _store_cached_summary(low_importance_msgs or dropped, summary_text)

            if estimate_tokens(summary_text) < estimate_messages_tokens(dropped) // 2:
                pruned.append({
                    "role": "user",
                    "content": f"[CONTEXT SUMMARY ({len(dropped)} messages):\n{summary_text}]",
                })

                # Don't add important messages that might have orphaned tool_results
                # Just add non-tool messages
                for imp_msg in important_msgs[:3]:
                    if not has_tool_result_blocks(imp_msg.get("content")):
                        pruned.append(imp_msg)

                meta["summarized_messages"] = len(dropped)
                meta["kept_important"] = min(3, len(important_msgs))
                meta["used_summary_cache"] = bool(_get_cached_summary(low_importance_msgs or dropped))

    pruned.extend(recent_messages)
    
    # Final safety: strip any orphaned tool_results and tool_use blocks
    pruned = strip_orphaned_tool_results(pruned)
    pruned = strip_orphaned_tool_use(pruned)

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
                    # Don't remove if it would orphan tool_results
                    if msg.get("role") == "assistant" and has_tool_use_blocks(msg.get("content")):
                        # Check if next message has tool_results for this
                        continue
                    pruned.pop(i)
                    break
        
        # Re-strip orphaned tool_results and tool_use after aggressive pruning
        pruned = strip_orphaned_tool_results(pruned)
        pruned = strip_orphaned_tool_use(pruned)

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
