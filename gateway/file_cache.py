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

    if rds is not None:
        cache_key = f"filecache:{project_id or 'default'}:{content_hash}"
        try:
            cached = rds.get(cache_key)
            if cached:
                record_cache_hit("file_hash")
                ref_msg = f"[FILE_REF:{content_hash[:12]}] (unchanged from previous request)"
                return True, content_hash, ref_msg
        except Exception as e:
            log.warning("File cache Redis check failed: %r", e)

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

                    ref_msg = f"[FILE_REF:{content_hash[:12]}] (unchanged from previous request)"
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

    if rds is not None:
        cache_key = f"filecache:{project_id or 'default'}:{content_hash}"
        try:
            rds.setex(cache_key, FILE_HASH_CACHE_TTL, "1")
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
                else:
                    new_entry = FileHashEntry(
                        project_id=project_id,
                        file_path=file_path or "unknown",
                        content_hash=content_hash,
                        content_preview=content[:500],
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


async def get_file_cache_stats(project_id: Optional[int] = None) -> Dict[str, Any]:
    stats = {
        "enabled": ENABLE_FILE_HASH_CACHE,
        "ttl_seconds": FILE_HASH_CACHE_TTL,
        "entries": 0,
        "total_chars_saved": 0,
    }

    if not DATABASE_URL:
        return stats

    try:
        from gateway.db import get_session, FileHashEntry
        from sqlalchemy import select, func

        async with get_session() as session:
            query = select(
                func.count(FileHashEntry.id),
                func.sum(FileHashEntry.char_count),
            )
            if project_id:
                query = query.where(FileHashEntry.project_id == project_id)

            result = await session.execute(query)
            row = result.one()

            stats["entries"] = int(row[0] or 0)
            stats["total_chars_saved"] = int(row[1] or 0)

    except Exception as e:
        log.warning("Failed to get file cache stats: %r", e)

    return stats
