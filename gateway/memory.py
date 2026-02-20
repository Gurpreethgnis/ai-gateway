import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx

from gateway.config import (
    ENABLE_MEMORY_LAYER,
    EMBEDDING_API_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
    DATABASE_URL,
)
from gateway.logging_setup import log


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


async def get_embedding(text: str) -> Optional[List[float]]:
    if not EMBEDDING_API_KEY:
        log.warning("Embedding API key not configured")
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                EMBEDDING_API_URL,
                headers={
                    "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text[:8000],
                },
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]
            return embedding

    except Exception as e:
        log.warning("Failed to get embedding: %r", e)
        return None


async def store_memory(
    project_id: int,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    if not ENABLE_MEMORY_LAYER or not DATABASE_URL:
        return False

    if len(content) < 50:
        return False

    try:
        from gateway.db import get_session, EmbeddingChunk
        from sqlalchemy import select

        content_hash = compute_hash(content)

        async with get_session() as session:
            existing = await session.execute(
                select(EmbeddingChunk).where(
                    EmbeddingChunk.project_id == project_id,
                    EmbeddingChunk.content_hash == content_hash,
                )
            )
            if existing.scalar_one_or_none():
                return False

            embedding = await get_embedding(content)
            embedding_str = json.dumps(embedding) if embedding else None

            chunk = EmbeddingChunk(
                project_id=project_id,
                content_hash=content_hash,
                text=content[:10000],
                embedding_vector=embedding_str,
                metadata_json=json.dumps(metadata or {}),
                created_at=datetime.utcnow(),
            )
            session.add(chunk)
            return True

    except Exception as e:
        log.warning("Failed to store memory: %r", e)
        return False


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


async def recall_memory(
    project_id: int,
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.7,
) -> List[Dict[str, Any]]:
    if not ENABLE_MEMORY_LAYER or not DATABASE_URL:
        return []

    try:
        from gateway.db import get_session, EmbeddingChunk
        from sqlalchemy import select

        query_embedding = await get_embedding(query)
        if not query_embedding:
            return []

        async with get_session() as session:
            result = await session.execute(
                select(EmbeddingChunk)
                .where(
                    EmbeddingChunk.project_id == project_id,
                    EmbeddingChunk.embedding_vector.isnot(None),
                )
                .limit(1000)
            )
            chunks = result.scalars().all()

            scored = []
            for chunk in chunks:
                try:
                    chunk_embedding = json.loads(chunk.embedding_vector)
                    similarity = cosine_similarity(query_embedding, chunk_embedding)

                    if similarity >= min_similarity:
                        scored.append({
                            "id": chunk.id,
                            "text": chunk.text,
                            "similarity": similarity,
                            "metadata": json.loads(chunk.metadata_json or "{}"),
                            "created_at": chunk.created_at.isoformat(),
                        })
                except Exception:
                    continue

            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return scored[:top_k]

    except Exception as e:
        log.warning("Failed to recall memory: %r", e)
        return []


async def extract_memorable_content(
    messages: List[Dict[str, Any]],
    project_id: int,
) -> int:
    if not ENABLE_MEMORY_LAYER:
        return 0

    stored_count = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant" and isinstance(content, str) and len(content) > 200:
            if any(kw in content.lower() for kw in [
                "solution", "implement", "here's how", "the issue",
                "architecture", "design", "pattern",
            ]):
                success = await store_memory(
                    project_id,
                    content[:5000],
                    {"role": role, "type": "assistant_response"},
                )
                if success:
                    stored_count += 1

        if role == "user" and isinstance(content, str):
            if any(kw in content.lower() for kw in [
                "requirement", "must", "should", "need to",
                "important", "always", "never", "rule",
            ]):
                success = await store_memory(
                    project_id,
                    content[:5000],
                    {"role": role, "type": "user_requirement"},
                )
                if success:
                    stored_count += 1

    return stored_count


async def build_memory_context(
    project_id: int,
    current_query: str,
    max_chars: int = 2000,
) -> str:
    if not ENABLE_MEMORY_LAYER:
        return ""

    memories = await recall_memory(project_id, current_query, top_k=3)

    if not memories:
        return ""

    context_parts = ["[Relevant context from previous conversations:]"]

    char_count = len(context_parts[0])
    for mem in memories:
        text = mem["text"]
        if char_count + len(text) > max_chars:
            remaining = max_chars - char_count - 50
            if remaining > 100:
                text = text[:remaining] + "..."
            else:
                break

        context_parts.append(f"- {text[:500]}...")
        char_count += len(text) + 10

    return "\n".join(context_parts)


async def get_memory_stats(project_id: int) -> Dict[str, Any]:
    if not DATABASE_URL:
        return {"enabled": ENABLE_MEMORY_LAYER, "count": 0}

    try:
        from gateway.db import get_session, EmbeddingChunk
        from sqlalchemy import select, func

        async with get_session() as session:
            result = await session.execute(
                select(func.count(EmbeddingChunk.id)).where(
                    EmbeddingChunk.project_id == project_id
                )
            )
            count = result.scalar() or 0

            return {
                "enabled": ENABLE_MEMORY_LAYER,
                "count": count,
                "project_id": project_id,
            }

    except Exception as e:
        log.warning("Failed to get memory stats: %r", e)
        return {"enabled": ENABLE_MEMORY_LAYER, "count": 0, "error": str(e)}


async def clear_memory(project_id: int) -> int:
    if not DATABASE_URL:
        return 0

    try:
        from gateway.db import get_session, EmbeddingChunk
        from sqlalchemy import delete

        async with get_session() as session:
            result = await session.execute(
                delete(EmbeddingChunk).where(
                    EmbeddingChunk.project_id == project_id
                )
            )
            return result.rowcount

    except Exception as e:
        log.warning("Failed to clear memory: %r", e)
        return 0
