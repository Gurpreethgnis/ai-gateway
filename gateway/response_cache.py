"""
Response Cache - Exact-match caching for identical requests.

Layer 1 of the multi-layer caching strategy.
Returns cached response immediately for identical requests.
"""

import json
import hashlib
from typing import Dict, Any, Optional, List

from gateway.cache import rds, cache_key
from gateway.logging_setup import log
from gateway.metrics import record_cache_hit, record_cache_miss


# Default TTL: 30 minutes
RESPONSE_CACHE_TTL = 1800


def compute_response_cache_key(
    messages: List[Dict],
    model: str,
    system: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
) -> str:
    """
    Compute cache key for a request.
    
    Key is based on:
    - System prompt
    - Messages (role + content)
    - Model
    - Tools (if any)
    """
    key_data = {
        "system": system or "",
        "messages": [
            {"role": m.get("role"), "content": _normalize_content(m.get("content"))}
            for m in messages
        ],
        "model": model,
    }
    
    if tools:
        # Sort tools for consistent hashing
        key_data["tools"] = sorted(
            [{"name": t.get("function", {}).get("name", "")} for t in tools],
            key=lambda x: x["name"],
        )
    
    raw = json.dumps(key_data, sort_keys=True, ensure_ascii=False).encode()
    return f"response:{hashlib.sha256(raw).hexdigest()}"


def _normalize_content(content: Any) -> str:
    """Normalize content to string for hashing."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(f"tool_result:{block.get('content', '')}")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


async def check_response_cache(
    messages: List[Dict],
    model: str,
    system: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
) -> Optional[Dict]:
    """
    Check if an exact-match cached response exists.
    
    Returns:
        Cached response dict if found, None otherwise.
    """
    if rds is None:
        return None
    
    try:
        key = compute_response_cache_key(messages, model, system, tools)
        cached = rds.get(key)
        
        if cached:
            log.info("Response cache HIT: %s", key[:20])
            record_cache_hit("response")
            return json.loads(cached)
        
        record_cache_miss("response")
        return None
        
    except Exception as e:
        log.warning("Response cache check failed: %r", e)
        return None


async def store_response_cache(
    messages: List[Dict],
    model: str,
    response: Dict,
    system: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
    ttl: int = RESPONSE_CACHE_TTL,
):
    """
    Store a response in the cache.
    
    Only caches successful, non-streaming responses.
    """
    if rds is None:
        return
    
    # Don't cache tool call responses (they're context-dependent)
    if response.get("tool_calls"):
        return
    
    # Don't cache very short responses (might be errors)
    content = response.get("content", "")
    if len(content) < 50:
        return
    
    try:
        key = compute_response_cache_key(messages, model, system, tools)
        
        # Store minimal response data
        cache_data = {
            "content": content,
            "model": response.get("model", model),
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "finish_reason": response.get("finish_reason", "stop"),
            "cached": True,
        }
        
        rds.setex(key, ttl, json.dumps(cache_data))
        log.debug("Stored response in cache: %s", key[:20])
        
    except Exception as e:
        log.warning("Response cache store failed: %r", e)


async def get_response_cache_stats() -> Dict[str, Any]:
    """Get response cache statistics."""
    if rds is None:
        return {"enabled": False}
    
    try:
        # Count response cache keys
        cursor = 0
        count = 0
        
        while True:
            cursor, keys = rds.scan(cursor, match="response:*", count=100)
            count += len(keys)
            if cursor == 0:
                break
        
        return {
            "enabled": True,
            "cached_responses": count,
            "ttl_seconds": RESPONSE_CACHE_TTL,
        }
        
    except Exception as e:
        log.warning("Failed to get response cache stats: %r", e)
        return {"enabled": True, "error": str(e)}
