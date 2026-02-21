import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

from gateway.config import ENABLE_FILE_HASH_CACHE, FILE_HASH_CACHE_TTL, DATABASE_URL
from gateway.cache import rds
from gateway.logging_setup import log
from gateway.metrics import record_cache_hit, record_cache_miss


def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


async def check_file_cache(
    project_id: Optional[int],
    content: str,
    file_path: Optional[str] = None,
) -> Tuple[bool, Optional[str], str]:
    if not ENABLE_FILE_HASH_CACHE:
        return False, None, content

    if len(content) < 500:
        return False, None, content

    content_hash = compute_content_hash(content)

    # Check Redis first
    if rds is not None:
        cache_key = f"filecache:{project_id or 'default'}:{content_hash}"
        try:
            cached = rds.get(cache_key)
            if cached:
                record_cache_hit("file_hash")
                ref_msg = f"[FILE_REF:{content_hash[:12]}] (unchanged from previous request; call get_file_by_hash if you need the content)"
                return True, content_hash, ref_msg
        except Exception as e:
            log.warning("File cache Redis check failed: %r", e)

    # Check database
    if DATABASE_URL and project_id:
        try:
            from gateway.db import get_session, FileHashEntry
            from sqlalchemy import select

            async with get_session() as session:
                cutoff = datetime.utcnow() - timedelta(seconds=FILE_HASH_CACHE_TTL)
                query = select(FileHashEntry).where(
                    FileHashEntry.project_id == project_id,
                    FileHashEntry.content_hash == content_hash,
                    FileHashEntry.last_seen >= cutoff,
                )
                result = await session.execute(query)
                entry = result.scalar_one_or_none()

                if entry:
                    entry.last_seen = datetime.utcnow()
                    record_cache_hit("file_hash")

                    if rds is not None:
                        try:
                            rds.setex(
                                f"filecache:{project_id}:{content_hash}",
                                FILE_HASH_CACHE_TTL,
                                "1",
                            )
                        except Exception:
                            pass

                    ref_msg = f"[FILE_REF:{content_hash[:12]}] (unchanged from previous request; call get_file_by_hash if you need the content)"
                    return True, content_hash, ref_msg

        except Exception as e:
            log.warning("File cache DB check failed: %r", e)

    record_cache_miss("file_hash")
    return False, content_hash, content


async def store_file_hash(
    project_id: Optional[int],
    content: str,
    content_hash: str,
    file_path: Optional[str] = None,
):
    if not ENABLE_FILE_HASH_CACHE:
        return

    # Store full content in Redis with longer TTL for retrieval
    if rds is not None:
        cache_key = f"filecache:{project_id or 'default'}:{content_hash}"
        content_key = f"filecontent:{project_id or 'default'}:{content_hash}"
        try:
            rds.setex(cache_key, FILE_HASH_CACHE_TTL, "1")
            # Store full content for retrieval (longer TTL for stable files)
            rds.setex(content_key, FILE_HASH_CACHE_TTL * 2, content)
        except Exception as e:
            log.warning("File cache Redis store failed: %r", e)

    if DATABASE_URL and project_id:
        try:
            from gateway.db import get_session, FileHashEntry
            from sqlalchemy import select
            from sqlalchemy.dialects.postgresql import insert

            async with get_session() as session:
                existing = await session.execute(
                    select(FileHashEntry).where(
                        FileHashEntry.project_id == project_id,
                        FileHashEntry.content_hash == content_hash,
                    )
                )
                entry = existing.scalar_one_or_none()

                if entry:
                    entry.last_seen = datetime.utcnow()
                    if file_path:
                        entry.file_path = file_path
                    # Store full content in DB for retrieval
                    entry.full_content = content
                else:
                    new_entry = FileHashEntry(
                        project_id=project_id,
                        file_path=file_path or "unknown",
                        content_hash=content_hash,
                        content_preview=content[:500],
                        full_content=content,  # Store full content
                        char_count=len(content),
                        last_seen=datetime.utcnow(),
                    )
                    session.add(new_entry)

        except Exception as e:
            log.warning("File cache DB store failed: %r", e)


async def process_tool_result(
    project_id: Optional[int],
    content: str,
    tool_name: Optional[str] = None,
) -> str:
    if not ENABLE_FILE_HASH_CACHE:
        return content

    if len(content) < 500:
        return content

    is_file_content = False
    if tool_name:
        file_tools = ["read_file", "read", "cat", "view_file", "get_file"]
        is_file_content = any(ft in tool_name.lower() for ft in file_tools)

    if not is_file_content and len(content) < 2000:
        return content

    file_path = None
    if is_file_content and "\n" in content:
        first_line = content.split("\n")[0]
        if "/" in first_line or "\\" in first_line:
            file_path = first_line.strip()[:256]

    is_cached, content_hash, result = await check_file_cache(
        project_id, content, file_path
    )

    if is_cached:
        return result

    if content_hash:
        await store_file_hash(project_id, content, content_hash, file_path)

    return content


async def get_file_by_hash(
    project_id: Optional[int],
    content_hash: str,
) -> Optional[str]:
    """
    Retrieve full file content by hash. Used when model needs context after seeing FILE_REF placeholder.
    """
    if not ENABLE_FILE_HASH_CACHE:
        return None
    
    # Try Redis first
    if rds is not None:
        content_key = f"filecontent:{project_id or 'default'}:{content_hash}"
        try:
            content = rds.get(content_key)
            if content:
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="ignore")
                log.debug("Retrieved file content from Redis: %s chars", len(content))
                return content
        except Exception as e:
            log.warning("File content Redis retrieval failed: %r", e)
    
    # Try database
    if DATABASE_URL and project_id:
        try:
            from gateway.db import get_session, FileHashEntry
            from sqlalchemy import select

            async with get_session() as session:
                query = select(FileHashEntry).where(
                    FileHashEntry.project_id == project_id,
                    FileHashEntry.content_hash == content_hash,
                )
                result = await session.execute(query)
                entry = result.scalar_one_or_none()

                if entry and hasattr(entry, 'full_content') and entry.full_content:
                    log.debug("Retrieved file content from DB: %s chars", len(entry.full_content))
                    # Warm Redis cache
                    if rds is not None:
                        try:
                            rds.setex(
                                f"filecontent:{project_id}:{content_hash}",
                                FILE_HASH_CACHE_TTL * 2,
                                entry.full_content,
                            )
                        except Exception:
                            pass
                    return entry.full_content

        except Exception as e:
            log.warning("File content DB retrieval failed: %r", e)
    
    return None


async def get_file_cache_stats(project_id: Optional[int] = None) -> Dict[str, Any]:
    stats = {
        "enabled": ENABLE_FILE_HASH_CACHE,
        "ttl_seconds": FILE_HASH_CACHE_TTL,
        "unique_files": 0,
        "total_chars_in_cache": 0,
        "estimated_chars_saved": 0,  # Actual savings from not re-transmitting
    }

    if not DATABASE_URL:
        return stats

    try:
        from gateway.db import get_session, FileHashEntry
        from sqlalchemy import select, func

        async with get_session() as session:
            # Count unique cached files
            query = select(
                func.count(FileHashEntry.id),
                func.sum(FileHashEntry.char_count),
            )
            if project_id:
                query = query.where(FileHashEntry.project_id == project_id)

            result = await session.execute(query)
            row = result.one()

            unique_count = int(row[0] or 0)
            total_chars = int(row[1] or 0)
            
            stats["unique_files"] = unique_count
            stats["total_chars_in_cache"] = total_chars
            
            # Estimate savings: each file hit saves (char_count - short_ref_length)
            # FILE_REF messages are ~80 chars; saved = total_chars - (unique_count * 80)
            # But this is per-hit, so we need hit count. For now, estimate conservatively:
            # If a file is cached, assume at least 1 additional hit (2x total) = total_chars saved
            stats["estimated_chars_saved"] = total_chars if unique_count > 0 else 0

    except Exception as e:
        log.warning("Failed to get file cache stats: %r", e)

    return stats
