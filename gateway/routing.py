from typing import Optional

from gateway.config import DEFAULT_MODEL, OPUS_MODEL, MODEL_PREFIX

def is_hard_task(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in [
        "architecture", "design doc", "production", "incident",
        "migration", "refactor", "security", "performance",
        "kubernetes", "terraform", "postmortem",
    ])

def strip_model_prefix(m: Optional[str]) -> Optional[str]:
    if not m:
        return None
    s = m.strip()
    low = s.lower()
    if low.startswith("mymodel:"):
        return s[len("MYMODEL:"):].strip()
    if low.startswith("mymodel-"):
        return s[len("MYMODEL-"):].strip()
    return s

def with_model_prefix(m: str) -> str:
    base = strip_model_prefix(m) or ""
    return f"{MODEL_PREFIX}{base}"

def map_model_alias(maybe: Optional[str]) -> Optional[str]:
    if not maybe:
        return None
    maybe = strip_model_prefix(maybe)
    if not maybe:
        return None
    m = maybe.strip().lower()

    if m in ("sonnet", "sonnet-4", "sonnet4", "claude-sonnet", "claude-sonnet-4"):
        return DEFAULT_MODEL
    if m in ("opus", "opus-4", "opus4", "claude-opus", "claude-opus-4"):
        return OPUS_MODEL
    if m.startswith("claude-"):
        return maybe
    return None

def route_model_from_messages(user_text: str, explicit_model: Optional[str]) -> str:
    mapped = map_model_alias(explicit_model)
    if mapped:
        return mapped
    return OPUS_MODEL if is_hard_task((user_text or "")[:8000]) else DEFAULT_MODEL
