import re
from typing import Dict, Any, List, Tuple
from gateway.config import (
    STRIP_IDE_BOILERPLATE,
    TOOL_RESULT_MAX_CHARS,
    USER_MSG_MAX_CHARS,
    SYSTEM_MAX_CHARS,
    ENFORCE_DIFF_FIRST,
    ENABLE_PREFIX_CACHE,
    PREFIX_CACHE_TTL_SECONDS,
    ENABLE_TOOL_RESULT_DEDUP,
)
from gateway.cache import rds_get_str, rds_set_str, sha256_text

_BOILERPLATE_PATTERNS: List[re.Pattern] = [
    re.compile(r"you are continue\b", re.IGNORECASE),
    re.compile(r"continue(?:\.| )?ai\b", re.IGNORECASE),
    re.compile(r"cursor\b.*(agent|instructions)", re.IGNORECASE),
    re.compile(r"tool(?:ing)? instructions", re.IGNORECASE),
    re.compile(r"##\s*tools?\b", re.IGNORECASE),
    re.compile(r"^<\|im_start\|>", re.IGNORECASE),
]

DIFF_FIRST_RULES = """\
DIFF-FIRST EDITING POLICY:
- When modifying files, respond with unified diffs (git-style) unless user explicitly asks for full file.
- Prefer minimal patches touching the smallest region.
- If you need file context, ask via tool calls for specific files/lines instead of requesting the whole repo.
- Never paste entire large files unless requested; output a patch + brief rationale.
"""

def looks_like_ide_boilerplate(s: str) -> bool:
    if not s or len(s) < 1500:
        return False
    hits = sum(1 for p in _BOILERPLATE_PATTERNS if p.search(s or ""))
    return hits >= 2

def strip_or_truncate(role: str, text: str, max_chars: int, allow_strip: bool) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"stripped": False, "truncated": False, "before": len(text or ""), "after": len(text or "")}
    if not text:
        return "", meta

    if allow_strip and STRIP_IDE_BOILERPLATE and looks_like_ide_boilerplate(text):
        meta["stripped"] = True
        text = ""
        meta["after"] = 0
        return text, meta

    if max_chars > 0 and len(text) > max_chars:
        meta["truncated"] = True
        head = text[: int(max_chars * 0.7)]
        tail = text[-int(max_chars * 0.3):]
        text = head + "\n\n[...TRUNCATED...]\n\n" + tail
        meta["after"] = len(text)
        return text, meta

    meta["after"] = len(text)
    return text, meta

def prefix_cache_key(system_text: str) -> str:
    return "prefix:" + sha256_text(system_text)

def maybe_prefix_cache(system_text: str) -> Tuple[str, Dict[str, Any]]:
    meta = {"prefix_cached": False, "prefix_hit": False, "key": None}
    if not ENABLE_PREFIX_CACHE:
        return system_text, meta
    if not system_text or len(system_text) < 2000:
        return system_text, meta

    key = prefix_cache_key(system_text)
    meta["key"] = key
    existing = rds_get_str(key)
    if existing is not None:
        meta["prefix_hit"] = True
        return f"[CACHED_SYSTEM_PREFIX:{key}]", meta

    rds_set_str(key, system_text, PREFIX_CACHE_TTL_SECONDS)
    meta["prefix_cached"] = True
    return f"[CACHED_SYSTEM_PREFIX:{key}]", meta

def tool_result_dedup(tool_call_id: str, tool_text: str) -> Tuple[str, Dict[str, Any]]:
    meta = {"tool_dedup": False, "tool_hit": False, "key": None}
    if not ENABLE_TOOL_RESULT_DEDUP:
        return tool_text, meta
    if not tool_call_id or not tool_text or len(tool_text) < 2000:
        return tool_text, meta

    key = "toolres:" + sha256_text(tool_call_id + ":" + tool_text)
    meta["key"] = key
    existing = rds_get_str(key)
    if existing is not None:
        meta["tool_hit"] = True
        return f"[CACHED_TOOL_RESULT:{key}]", meta

    rds_set_str(key, tool_text, PREFIX_CACHE_TTL_SECONDS)
    meta["tool_dedup"] = True
    return tool_text, meta

def enforce_diff_first(system_text: str) -> str:
    if not ENFORCE_DIFF_FIRST:
        return system_text
    if DIFF_FIRST_RULES in (system_text or ""):
        return system_text
    return (system_text + "\n\n" + DIFF_FIRST_RULES).strip()

# exported limits for callers
LIMITS = {
    "tool_result_max": TOOL_RESULT_MAX_CHARS,
    "user_msg_max": USER_MSG_MAX_CHARS,
    "system_max": SYSTEM_MAX_CHARS,
}

