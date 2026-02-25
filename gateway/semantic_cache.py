"""
Semantic Cache - Layer 2 caching for similar queries.

Uses embeddings to find semantically similar previous queries
and return cached responses without hitting the model.
"""

import hashlib
import json
from typing import Optional, Dict, Any, List

from gateway.canonical_format import to_canonical_messages
from gateway.logging_setup import log
from gateway import config


# =============================================================================
# Configuration
# =============================================================================

ENABLE_SEMANTIC_CACHE = getattr(config, "ENABLE_SEMANTIC_CACHE", False)
SEMANTIC_SIMILARITY_THRESHOLD = getattr(config, "SEMANTIC_SIMILARITY_THRESHOLD", 0.95)
SEMANTIC_CACHE_TTL = getattr(config, "SEMANTIC_CACHE_TTL", 3600)  # 1 hour


# =============================================================================
# Embedding Functions
# =============================================================================

async def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding vector for text.
    
    Uses local Ollama model for embeddings to avoid external API calls.
    Falls back to simple hash-based pseudo-embedding if unavailable.
    """
    if not text or len(text.strip()) < 10:
        return None
    
    try:
        import httpx
        
        ollama_url = getattr(config, "OLLAMA_URL", None) or getattr(config, "LOCAL_LLM_BASE_URL", None)
        if not ollama_url:
            return _fallback_embedding(text)
        
        url = f"{ollama_url.rstrip('/')}/api/embeddings"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={
                "model": "nomic-embed-text",  # Common embedding model
                "prompt": text[:2000],  # Truncate long texts
            })
            
            if resp.status_code == 200:
                data = resp.json()
                return data.get("embedding")
            else:
                log.debug("Embedding API error: %d", resp.status_code)
                return _fallback_embedding(text)
                
    except Exception as e:
        log.debug("Embedding failed, using fallback: %r", e)
        return _fallback_embedding(text)


def _fallback_embedding(text: str) -> List[float]:
    """
    Simple hash-based pseudo-embedding for when Ollama is unavailable.
    Not semantically meaningful but provides consistent hashing.
    """
    # Create a deterministic "embedding" from text hash
    text_hash = hashlib.sha256(text.lower().encode()).hexdigest()
    
    # Convert to float vector (384 dimensions to match common embedding sizes)
    embedding = []
    for i in range(0, min(len(text_hash), 64), 2):
        val = int(text_hash[i:i+2], 16) / 255.0
        embedding.append(val)
    
    # Pad to 384 dimensions
    while len(embedding) < 384:
        embedding.extend(embedding[:min(len(embedding), 384 - len(embedding))])
    
    return embedding[:384]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


# =============================================================================
# Cache Functions
# =============================================================================

def compute_context_hash(messages: List[Dict], system: str = "") -> str:
    """
    Compute hash of conversation context (excluding last message).
    Used to ensure semantic matches are from same conversation context.
    """
    # Use all messages except the last one (the query)
    context_messages = messages[:-1] if len(messages) > 1 else []
    
    canonical_context = to_canonical_messages(context_messages[-5:])
    canonical_payload = []
    for msg in canonical_context:
        text_fragments = []
        for block in msg.content:
            if block.type == "text" and block.text:
                text_fragments.append(block.text)
            elif block.type == "tool_use" and block.tool_name:
                text_fragments.append(f"[tool:{block.tool_name}]")
            elif block.type == "tool_result" and block.text:
                text_fragments.append(f"[tool_result:{block.text}]")
        canonical_payload.append({"role": msg.role, "content": "\n".join(text_fragments)[:200]})

    content = json.dumps({
        "system": system[:500] if system else "",
        "context": canonical_payload,
    }, sort_keys=True)
    
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def check_semantic_cache(
    last_message: str,
    context_hash: str,
    model: str = None,
) -> Optional[Dict[str, Any]]:
    """
    Find semantically similar cached queries.
    
    Args:
        last_message: The user's query text
        context_hash: Hash of conversation context
        model: Optional model filter
        
    Returns:
        Cached response dict if found, None otherwise
    """
    if not ENABLE_SEMANTIC_CACHE:
        return None
    
    try:
        from gateway.cache import rds
        
        if not rds:
            return None
        
        # Get embedding of current query
        query_embedding = await get_embedding(last_message)
        if not query_embedding:
            return None
        
        # Get all semantic cache entries for this context
        cache_key_pattern = f"semantic:{context_hash}:*"
        
        # Scan for matching keys
        cursor = 0
        best_match = None
        best_similarity = 0.0
        
        while True:
            cursor, keys = rds.scan(cursor, match=cache_key_pattern, count=100)
            
            for key in keys:
                try:
                    cached = rds.hgetall(key)
                    if not cached:
                        continue
                    
                    # Decode cached embedding
                    cached_embedding_str = cached.get(b"embedding") or cached.get("embedding")
                    if not cached_embedding_str:
                        continue
                    
                    cached_embedding = json.loads(cached_embedding_str)
                    
                    # Compute similarity
                    similarity = cosine_similarity(query_embedding, cached_embedding)
                    
                    if similarity >= SEMANTIC_SIMILARITY_THRESHOLD and similarity > best_similarity:
                        best_similarity = similarity
                        
                        response_str = cached.get(b"response") or cached.get("response")
                        if response_str:
                            best_match = {
                                "response": json.loads(response_str),
                                "similarity": similarity,
                                "cached_query": (cached.get(b"query") or cached.get("query", b"")).decode() if isinstance(cached.get(b"query") or cached.get("query"), bytes) else cached.get("query", ""),
                            }
                            
                except Exception as e:
                    log.debug("Error checking semantic cache key %s: %r", key, e)
                    continue
            
            if cursor == 0:
                break
        
        if best_match:
            log.info("Semantic cache HIT (similarity=%.3f)", best_similarity)
            return best_match["response"]
        
        return None
        
    except Exception as e:
        log.warning("Semantic cache check failed: %r", e)
        return None


async def store_semantic_cache(
    last_message: str,
    context_hash: str,
    response: Dict[str, Any],
    model: str = None,
):
    """
    Store query and response in semantic cache.
    
    Args:
        last_message: The user's query text
        context_hash: Hash of conversation context
        response: The response to cache
        model: Model that generated the response
    """
    if not ENABLE_SEMANTIC_CACHE:
        return
    
    try:
        from gateway.cache import rds
        
        if not rds:
            return
        
        # Get embedding
        embedding = await get_embedding(last_message)
        if not embedding:
            return
        
        # Generate unique key for this query
        query_hash = hashlib.sha256(last_message.encode()).hexdigest()[:12]
        cache_key = f"semantic:{context_hash}:{query_hash}"
        
        # Store in Redis hash
        rds.hset(cache_key, mapping={
            "query": last_message[:500],
            "embedding": json.dumps(embedding),
            "response": json.dumps(response),
            "model": model or "",
        })
        
        # Set TTL
        rds.expire(cache_key, SEMANTIC_CACHE_TTL)
        
        log.debug("Stored in semantic cache: %s", cache_key)
        
    except Exception as e:
        log.warning("Failed to store in semantic cache: %r", e)


# =============================================================================
# Utility Functions
# =============================================================================

def extract_last_user_message(messages: List[Dict]) -> str:
    """Extract text from the last user message."""
    canonical = to_canonical_messages(messages)
    for msg in reversed(canonical):
        if msg.role != "user":
            continue

        parts = []
        for block in msg.content:
            if block.type == "text" and block.text:
                parts.append(block.text)

        text = " ".join(parts).strip()
        if text:
            return text
    
    return ""


async def check_and_store_semantic_cache(
    messages: List[Dict],
    system: str,
    model: str,
    response: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Combined check and store for semantic cache.
    
    If response is None, checks cache and returns cached response.
    If response is provided, stores it and returns None.
    """
    last_message = extract_last_user_message(messages)
    if not last_message or len(last_message) < 20:
        return None
    
    context_hash = compute_context_hash(messages, system)
    
    if response is None:
        # Check mode
        return await check_semantic_cache(last_message, context_hash, model)
    else:
        # Store mode
        await store_semantic_cache(last_message, context_hash, response, model)
        return None
