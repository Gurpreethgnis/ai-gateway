import json
import hashlib
from typing import Any, Dict, Optional

from gateway.config import REDIS_URL
from gateway.logging_setup import log

try:
    import redis  # type: ignore
except Exception:
    redis = None

rds = None
if REDIS_URL and redis is not None:
    try:
        rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        rds.ping()
    except Exception as e:
        log.warning("Redis disabled: %r", e)
        rds = None

def cache_key(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()

def cache_get(key: str):
    if rds is None:
        return None
    try:
        v = rds.get(key)
        return json.loads(v) if v else None
    except Exception:
        return None

def cache_set(key: str, data: Dict[str, Any], ttl: int):
    if rds is None:
        return
    try:
        rds.setex(key, ttl, json.dumps(data))
    except Exception:
        pass

def rds_get_str(key: str) -> Optional[str]:
    if rds is None:
        return None
    try:
        return rds.get(key)
    except Exception:
        return None

def rds_set_str(key: str, val: str, ttl: int):
    if rds is None:
        return
    try:
        rds.setex(key, ttl, val)
    except Exception:
        return

def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()
