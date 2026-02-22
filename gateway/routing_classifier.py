"""
Phase 2: LLM-based routing classification using local Ollama model.

This module is called only for ambiguous requests where Phase 1 heuristics
couldn't confidently decide between local, sonnet, or opus tiers.
"""

import json
import hashlib
import asyncio
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

import httpx

from gateway.config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_DEFAULT_MODEL,
    ROUTING_CLASSIFIER_MODEL,
    ROUTING_CLASSIFIER_TIMEOUT,
    ROUTING_CLASSIFIER_CACHE_SIZE,
    LOCAL_CF_ACCESS_CLIENT_ID,
    LOCAL_CF_ACCESS_CLIENT_SECRET,
)
from gateway.logging_setup import log


# Classification prompt
CLASSIFIER_SYSTEM_PROMPT = """You are a request classifier for an AI coding gateway. Classify the incoming request into one of three tiers.

TIERS:
- "local": Simple tasks a 14B parameter model can handle well -- explanations, small code edits, single-file changes, boilerplate, Q&A, formatting, documentation, simple bug fixes, adding comments, type hints, docstrings.
- "sonnet": Moderate tasks needing strong reasoning -- multi-file edits, non-trivial refactors, debugging complex issues, writing tests for complex logic, code review with suggestions, medium complexity algorithms.
- "opus": Complex tasks requiring deep expertise -- system architecture, security audits, large-scale migrations, distributed systems design, production incident analysis, breaking-change planning, performance optimization across systems.

Consider:
1. Task complexity (single edit vs multi-step reasoning)
2. Scope (one file vs many files, one function vs entire system)
3. Domain difficulty (formatting vs distributed systems)
4. Required reasoning depth (lookup vs multi-step analysis)

Respond with ONLY a JSON object, no other text:
{"tier": "local"|"sonnet"|"opus", "reason": "<one sentence>"}"""


class ClassificationCache:
    """LRU cache for classification results."""
    
    def __init__(self, maxsize: int = 256):
        self._cache: Dict[str, Tuple[str, str]] = {}
        self._maxsize = maxsize
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Tuple[str, str]]:
        """Get cached classification (tier, reason)."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, key: str, tier: str, reason: str) -> None:
        """Cache classification result."""
        if key in self._cache:
            # Update existing
            self._access_order.remove(key)
        elif len(self._cache) >= self._maxsize:
            # Evict least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        
        self._cache[key] = (tier, reason)
        self._access_order.append(key)


# Global cache instance
_classification_cache = ClassificationCache(maxsize=ROUTING_CLASSIFIER_CACHE_SIZE)


def _hash_messages(messages: List[Dict[str, Any]]) -> str:
    """Generate cache key from last 2 messages."""
    # Take last 2 messages for cache key
    recent = messages[-2:] if len(messages) > 1 else messages
    content = json.dumps(recent, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _truncate_messages_for_classification(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Truncate messages to ~2000 chars for fast classification."""
    truncated = []
    total_chars = 0
    
    # Start from most recent messages
    for msg in reversed(messages):
        content = msg.get("content", "")
        if isinstance(content, str):
            msg_chars = len(content)
        elif isinstance(content, list):
            msg_chars = sum(len(str(block)) for block in content)
        else:
            msg_chars = 0
        
        if total_chars + msg_chars > 2000:
            break
        
        truncated.insert(0, msg)
        total_chars += msg_chars
    
    return truncated or messages[-1:]  # At least include last message


async def classify_with_llm(
    messages: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Call local Ollama model to classify request into local/sonnet/opus.
    
    Returns:
        (tier, reason) tuple, or ("sonnet", "classification_failed") on error
    """
    # Check cache first
    cache_key = _hash_messages(messages)
    cached = _classification_cache.get(cache_key)
    if cached:
        log.info("ROUTING CLASSIFIER: cache hit (key=%s)", cache_key)
        return cached
    
    # Validate configuration
    if not LOCAL_LLM_BASE_URL:
        log.warning("ROUTING CLASSIFIER: LOCAL_LLM_BASE_URL not set, falling back")
        return ("sonnet", "no_local_llm_configured")
    
    if not LOCAL_CF_ACCESS_CLIENT_ID or not LOCAL_CF_ACCESS_CLIENT_SECRET:
        log.warning("ROUTING CLASSIFIER: CF Access credentials not set, falling back")
        return ("sonnet", "no_cf_access_configured")
    
    # Build classification request
    model = ROUTING_CLASSIFIER_MODEL or LOCAL_LLM_DEFAULT_MODEL
    truncated_messages = _truncate_messages_for_classification(messages)
    
    # Convert to simple user message for classification
    user_content = "\n\n".join(
        msg.get("content", "") if isinstance(msg.get("content"), str) else str(msg.get("content", ""))
        for msg in truncated_messages
    )
    
    ollama_messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify this request:\n\n{user_content}"}
    ]
    
    payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 50,  # Short response
        }
    }
    
    headers = {
        "CF-Access-Client-Id": LOCAL_CF_ACCESS_CLIENT_ID,
        "CF-Access-Client-Secret": LOCAL_CF_ACCESS_CLIENT_SECRET,
        "Content-Type": "application/json",
    }
    
    url = f"{LOCAL_LLM_BASE_URL.rstrip('/')}/api/chat"
    
    log.info("ROUTING CLASSIFIER: calling Ollama model=%s", model)
    
    try:
        async with httpx.AsyncClient(timeout=ROUTING_CLASSIFIER_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
        
        if resp.status_code >= 400:
            log.error(
                "ROUTING CLASSIFIER: Ollama error status=%d response=%s",
                resp.status_code, resp.text[:200]
            )
            return ("sonnet", "ollama_error")
        
        data = resp.json()
        response_text = data.get("message", {}).get("content", "")
        
        # Parse JSON response
        # Try to extract JSON from response (might have markdown code blocks)
        json_str = response_text.strip()
        if "```" in json_str:
            # Extract from code block
            lines = json_str.split("\n")
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                elif in_code_block or (line.strip().startswith("{") or json_lines):
                    json_lines.append(line)
            json_str = "\n".join(json_lines)
        
        parsed = json.loads(json_str)
        tier = parsed.get("tier", "sonnet")
        reason = parsed.get("reason", "llm_classified")
        
        # Validate tier
        if tier not in ("local", "sonnet", "opus"):
            log.warning("ROUTING CLASSIFIER: invalid tier=%s, defaulting to sonnet", tier)
            tier = "sonnet"
            reason = "invalid_tier_response"
        
        # Cache result
        _classification_cache.set(cache_key, tier, reason)
        
        log.info("ROUTING CLASSIFIER: classified as tier=%s reason=%s", tier, reason)
        return (tier, reason)
    
    except asyncio.TimeoutError:
        log.warning("ROUTING CLASSIFIER: timeout after %ds", ROUTING_CLASSIFIER_TIMEOUT)
        return ("sonnet", "classification_timeout")
    
    except httpx.RequestError as e:
        log.warning("ROUTING CLASSIFIER: request error: %r", e)
        return ("sonnet", "ollama_unreachable")
    
    except json.JSONDecodeError as e:
        log.warning("ROUTING CLASSIFIER: JSON parse error: %r response=%s", e, response_text[:200])
        return ("sonnet", "invalid_json_response")
    
    except Exception as e:
        log.error("ROUTING CLASSIFIER: unexpected error: %r", e)
        return ("sonnet", "classification_error")
