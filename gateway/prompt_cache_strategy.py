"""Provider-aware prompt caching helpers and strategy utilities."""

import re
from typing import Any, Dict, List, Optional, Tuple

from gateway import config

_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{12}\b")
_ISO_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")
_EPOCH_RE = re.compile(r"\b\d{10,13}\b")


def infer_provider_from_model(model: str) -> str:
    """Infer provider from model ID/name."""
    model_lower = (model or "").lower()
    if not model_lower:
        return "unknown"

    local_allowlist = [str(m).lower() for m in getattr(config, "LOCAL_LLM_MODEL_ALLOWLIST", [])]
    local_default = str(getattr(config, "LOCAL_LLM_DEFAULT_MODEL", "") or "").lower()

    if model_lower in local_allowlist or (local_default and model_lower == local_default):
        return "ollama"
    if model_lower.startswith("local:") or model_lower.startswith("ollama:"):
        return "ollama"
    if model_lower.startswith("local/") or model_lower.startswith("ollama/"):
        return "ollama"
    if model_lower.startswith("llama") or model_lower.startswith("qwen") or model_lower.startswith("mistral"):
        return "ollama"
    if model_lower.startswith("phi") or model_lower.startswith("gemma") or model_lower.startswith("mixtral"):
        return "ollama"
    if model_lower.startswith("deepseek"):
        return "ollama"

    if "claude" in model_lower:
        return "anthropic"
    if "gpt" in model_lower or model_lower.startswith("o1"):
        return "openai"
    if "gemini" in model_lower:
        return "gemini"
    if model_lower.startswith("openai/"):
        return "openai"
    if model_lower.startswith("anthropic/"):
        return "anthropic"
    if model_lower.startswith("gemini/"):
        return "gemini"
    if model_lower.startswith("groq/"):
        return "groq"
    if model_lower.startswith("ollama/") or ":" in model_lower:
        return "ollama"
    return "unknown"


def classify_model_tier(model: str) -> str:
    """Classify models into broad routing tiers for cache sharing control."""
    model_lower = (model or "").lower()
    if any(tag in model_lower for tag in ["opus", "o1", "4o", "sonnet-4"]):
        return "smart"
    if any(tag in model_lower for tag in ["haiku", "mini", "flash", "8b", "7b"]):
        return "fast"
    return "balanced"


def stabilize_system_prompt(system_text: Optional[str]) -> str:
    """Normalize dynamic tokens in system prompts for cache-friendly prefix matching."""
    if not system_text:
        return ""
    stabilized = system_text
    stabilized = _UUID_RE.sub("<UUID>", stabilized)
    stabilized = _ISO_TS_RE.sub("<TIMESTAMP>", stabilized)
    stabilized = _EPOCH_RE.sub("<EPOCH>", stabilized)
    return stabilized


def apply_anthropic_prompt_cache_strategy(
    system_text: str,
    messages: List[Dict[str, Any]],
    enable_cache_control: Optional[bool] = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Build Anthropic cache-control blocks and annotate recent substantial user messages.

    Returns:
        system_param, transformed_messages
    """
    if enable_cache_control is None:
        enable_cache_control = bool(getattr(config, "ENABLE_ANTHROPIC_CACHE_CONTROL", True))

    transformed_messages = list(messages)
    system_param: Any = system_text

    if enable_cache_control and system_text and len(system_text) >= 1024:
        from gateway.platform_constitution import get_cacheable_system_blocks

        system_blocks = get_cacheable_system_blocks(include_constitution=True, include_diff_rules=True)
        system_blocks.append({"type": "text", "text": system_text})
        system_param = system_blocks

    if enable_cache_control and transformed_messages:
        user_msg_indices = [i for i, m in enumerate(transformed_messages) if m.get("role") == "user"]
        for idx in user_msg_indices[-2:]:
            msg = transformed_messages[idx]
            content = msg.get("content")
            if isinstance(content, str):
                if len(content) > 2048:
                    msg["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
            elif isinstance(content, list) and content:
                last_block = content[-1]
                if isinstance(last_block, dict) and last_block.get("type") == "text":
                    text_len = len(last_block.get("text", ""))
                    if text_len > 2048:
                        last_block["cache_control"] = {"type": "ephemeral"}

    return system_param, transformed_messages


def extract_openai_cached_tokens(raw_response: Any) -> int:
    """Best-effort extraction of cached prompt tokens from OpenAI response payload."""
    usage = None
    if isinstance(raw_response, dict):
        usage = raw_response.get("usage")
    else:
        usage = getattr(raw_response, "usage", None)

    if usage is None:
        return 0

    details = None
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
    else:
        details = getattr(usage, "prompt_tokens_details", None)

    if not details:
        return 0

    if isinstance(details, dict):
        return int(details.get("cached_tokens", 0) or 0)

    return int(getattr(details, "cached_tokens", 0) or 0)
