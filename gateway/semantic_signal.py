"""
Semantic routing signal: use embeddings to classify queries.

This is Phase B of the cascade routing enhancement. Embeddings provide a semantic
similarity signal that can improve routing decisions beyond keyword matching.

Usage:
1. Compute embedding of user's query
2. Compare to prototype embeddings for each tier (local, sonnet, opus)
3. Use similarity scores as additional routing signals
"""

import json
import hashlib
from typing import Optional, Tuple, List, Dict, Any
import httpx

from gateway.config import (
    ENABLE_SEMANTIC_ROUTING_SIGNAL,
    EMBEDDING_API_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
    SEMANTIC_EMBEDDING_CACHE_TTL,
)
from gateway.logging_setup import log


# Prototype embeddings (will be learned from routing_outcomes over time)
# For now, we start with None and can manually seed or train later
PROTOTYPE_EMBEDDINGS: Dict[str, Optional[List[float]]] = {
    "local": None,    # Simple tasks: explain code, fix typo, add comment
    "sonnet": None,   # Moderate tasks: multi-file refactor, test writing
    "opus": None,     # Complex tasks: architecture, security audit, system design
}


async def compute_query_embedding(query_text: str) -> Optional[List[float]]:
    """
    Compute embedding vector for a query using the configured embedding API.
    
    Args:
        query_text: The user's query text (last user message)
    
    Returns:
        Embedding vector (list of floats), or None on error
    """
    if not ENABLE_SEMANTIC_ROUTING_SIGNAL:
        return None
    
    if not EMBEDDING_API_URL or not EMBEDDING_API_KEY:
        log.debug("Semantic routing enabled but embedding API not configured")
        return None
    
    # Truncate query for embedding (most models have token limits)
    query_truncated = query_text[:500]
    
    try:
        # Check cache first
        cached = await _get_cached_embedding(query_truncated)
        if cached:
            return cached
        
        # Call embedding API (OpenAI-compatible format)
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                EMBEDDING_API_URL,
                headers={
                    "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": query_truncated,
                    "model": EMBEDDING_MODEL,
                }
            )
            resp.raise_for_status()
            data = resp.json()
        
        # Extract embedding vector
        embedding = data.get("data", [{}])[0].get("embedding")
        if not embedding:
            log.warning("Embedding API response missing embedding vector")
            return None
        
        # Cache it
        await _cache_embedding(query_truncated, embedding)
        
        return embedding
    
    except Exception as e:
        log.warning("Failed to compute query embedding: %r", e)
        return None


async def compute_semantic_signals(query_embedding: List[float]) -> Dict[str, float]:
    """
    Compute semantic similarity to prototype embeddings for each tier.
    
    Args:
        query_embedding: Query embedding vector
    
    Returns:
        Dict mapping tier -> similarity score (0.0-1.0)
    """
    if not query_embedding:
        return {}
    
    signals = {}
    
    for tier, prototype in PROTOTYPE_EMBEDDINGS.items():
        if prototype:
            similarity = _cosine_similarity(query_embedding, prototype)
            signals[f"semantic_{tier}_affinity"] = similarity
    
    return signals


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
    
    Returns:
        Cosine similarity (-1.0 to 1.0, normalized to 0.0-1.0)
    """
    if len(vec_a) != len(vec_b):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a * a for a in vec_a) ** 0.5
    magnitude_b = sum(b * b for b in vec_b) ** 0.5
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    cosine = dot_product / (magnitude_a * magnitude_b)
    
    # Normalize to 0-1 range (cosine is -1 to 1)
    return (cosine + 1.0) / 2.0


async def _get_cached_embedding(query_text: str) -> Optional[List[float]]:
    """Get cached embedding from Redis."""
    try:
        from gateway.cache import rds
        if not rds:
            return None
        
        cache_key = f"embedding:{_hash_query(query_text)}"
        cached_json = rds.get(cache_key)
        if cached_json:
            return json.loads(cached_json)
    except Exception as e:
        log.debug("Failed to get cached embedding: %r", e)
    
    return None


async def _cache_embedding(query_text: str, embedding: List[float]) -> None:
    """Cache embedding to Redis."""
    try:
        from gateway.cache import rds
        if not rds:
            return
        
        cache_key = f"embedding:{_hash_query(query_text)}"
        rds.setex(
            cache_key,
            SEMANTIC_EMBEDDING_CACHE_TTL,
            json.dumps(embedding)
        )
    except Exception as e:
        log.debug("Failed to cache embedding: %r", e)


def _hash_query(query_text: str) -> str:
    """Hash query text for cache key."""
    return hashlib.sha256(query_text.encode('utf-8')).hexdigest()[:16]


def update_prototype_embeddings(
    tier: str,
    embeddings: List[List[float]],
) -> None:
    """
    Update prototype embedding for a tier by averaging successful query embeddings.
    
    This is called periodically (e.g., daily batch job) to learn from routing outcomes.
    
    Args:
        tier: "local", "sonnet", or "opus"
        embeddings: List of embedding vectors from successful queries of this tier
    """
    if not embeddings:
        return
    
    # Compute centroid (average embedding)
    dim = len(embeddings[0])
    centroid = [0.0] * dim
    
    for embedding in embeddings:
        for i, val in enumerate(embedding):
            centroid[i] += val
    
    for i in range(dim):
        centroid[i] /= len(embeddings)
    
    PROTOTYPE_EMBEDDINGS[tier] = centroid
    log.info("Updated prototype embedding for tier=%s from %d samples", tier, len(embeddings))


def get_prototype_embeddings() -> Dict[str, Optional[List[float]]]:
    """Get current prototype embeddings (for debugging/inspection)."""
    return PROTOTYPE_EMBEDDINGS.copy()
