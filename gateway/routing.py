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
    prefix_lower = MODEL_PREFIX.lower()
    prefix_dash = prefix_lower.rstrip(":") + "-"
    
    if low.startswith(prefix_lower):
        return s[len(MODEL_PREFIX):].strip()
    if low.startswith(prefix_dash):
        return s[len(prefix_dash):].strip()
    return s

def with_model_prefix(m: str) -> str:
    base = strip_model_prefix(m) or ""
    return f"{MODEL_PREFIX}{base}"

# Known model IDs; used for validation. New Anthropic IDs (e.g. claude-4-6-*) are passed through.
VALID_ANTHROPIC_MODELS = {
    "claude-sonnet-4-0",
    "claude-opus-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
}


def _looks_like_anthropic_model_id(model: str) -> bool:
    """True if string looks like an Anthropic model ID so we pass it through (e.g. claude-sonnet-4-6)."""
    parts = model.split("-")
    return len(parts) >= 3 and parts[0] == "claude"


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

    if "opus" in m and not m.startswith("claude-"):
        return OPUS_MODEL
    if "sonnet" in m and not m.startswith("claude-"):
        return DEFAULT_MODEL

    if m in VALID_ANTHROPIC_MODELS:
        return m

    # Pass through any claude-X-Y-* style ID so new models (e.g. claude-sonnet-4-6) work without code changes
    if _looks_like_anthropic_model_id(m):
        return m

    if m.startswith("claude-"):
        for valid in VALID_ANTHROPIC_MODELS:
            if valid.startswith(m) or m.startswith(valid.split("-")[0] + "-" + valid.split("-")[1]):
                return valid
        return DEFAULT_MODEL

    return None

# Model Fallback Chains (Rotation)
# Used when a model hits its hourly or concurrent rate limit (429).
MODEL_FALLBACKS = {
    # Sonnet 3.5 Chain
    "claude-3-5-sonnet-20241022": [
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307",
    ],
    "claude-3-5-sonnet-20240620": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307",
    ],
    "claude-3-5-sonnet-latest": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-haiku-20240307",
    ],
    
    # Opus Chain
    "claude-3-opus-20240229": [
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ],
    "claude-3-opus-latest": [
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20241022",
    ],

    # Haiku Chain
    "claude-3-haiku-20240307": [
        "claude-3-5-haiku-20241022",  # if available
        "claude-3-5-sonnet-20240620",
    ]
}

def get_fallback_model(current_model: str, attempt: int) -> str:
    """
    Returns the next model in the rotation for the given attempt index.
    If no specific fallback is defined, returns Haiku as the ultimate baseline.
    """
    base = strip_model_prefix(current_model) or ""
    chain = MODEL_FALLBACKS.get(base, [])
    
    if not chain:
        # Generic fallback for unknown models
        return "claude-3-haiku-20240307" if attempt > 0 else current_model

    # Use modulo to cycle through the chain if multiple retries happen
    fallback_idx = (attempt - 1) % len(chain)
    return chain[fallback_idx]

def route_model_from_messages(user_text: str, explicit_model: Optional[str]) -> str:
    mapped = map_model_alias(explicit_model)
    if mapped:
        return mapped
    return OPUS_MODEL if is_hard_task((user_text or "")[:8000]) else DEFAULT_MODEL
