import time
import uuid

from gateway.cache import rds
from gateway.config import ANTHROPIC_MAX_CONCURRENCY, ANTHROPIC_QUEUE_TIMEOUT
from gateway.logging_setup import log

def acquire_concurrency_slot(model_name: str) -> str:
    """
    Blocks until a concurrency slot is available for the given model.
    Uses a Redis Sorted Set (ZSET) to track active connections by timestamp.
    Returns a unique request ID representing the acquired slot.
    """
    if not rds:
        return ""

    # Group concurrency by broad model class to match Anthropic tiers
    # e.g., "claude-3-5-sonnet-20241022" -> "sonnet"
    family = "default"
    if "sonnet" in model_name.lower():
        family = "sonnet"
    elif "opus" in model_name.lower():
        family = "opus"
    elif "haiku" in model_name.lower():
        family = "haiku"

    zset_key = f"concurrency:anthropic:{family}"
    req_id = str(uuid.uuid4())

    log.debug("QUEUE Attempting to acquire concurrency slot for %s", family)

    while True:
        now = time.time()
        
        # 1. Prune dead/stale connections (e.g., streams that crashed)
        cutoff = now - ANTHROPIC_QUEUE_TIMEOUT
        rds.zremrangebyscore(zset_key, "-inf", cutoff)
        
        # 2. Check current active connections
        active_count = rds.zcard(zset_key)
        
        if active_count < ANTHROPIC_MAX_CONCURRENCY:
            # Slot available! Add ourselves.
            # Use ZADD with mapping for newer Redis-py compatibility
            added = rds.zadd(zset_key, {req_id: now}, nx=True)
            if added:
                log.info("QUEUE Acquired slot %s/%s for %s (req_id=%s)", active_count + 1, ANTHROPIC_MAX_CONCURRENCY, family, req_id[:8])
                return req_id
                
        # Queue is full, chill for a moment before polling again
        log.info("QUEUE Waiting for Anthropic concurrency slot (%s >= %s)", active_count, ANTHROPIC_MAX_CONCURRENCY)
        time.sleep(0.5)

def release_concurrency_slot(model_name: str, req_id: str):
    """
    Releases the concurrency slot so the next queued request can proceed.
    """
    if not rds or not req_id:
        return

    family = "default"
    if "sonnet" in model_name.lower():
        family = "sonnet"
    elif "opus" in model_name.lower():
        family = "opus"
    elif "haiku" in model_name.lower():
        family = "haiku"

    zset_key = f"concurrency:anthropic:{family}"
    rds.zrem(zset_key, req_id)
    log.debug("QUEUE Released slot for %s (req_id=%s)", family, req_id[:8])
