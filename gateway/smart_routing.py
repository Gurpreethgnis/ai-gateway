"""
Smart Routing - Simplified wrapper around routing_engine.

This module provides backward compatibility with the old interface while
delegating to the new preference-based routing system.

For new code, use gateway.routing_engine directly.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from gateway.config import (
    ENABLE_SMART_ROUTING,
    DEFAULT_MODEL,
    OPUS_MODEL,
)
from gateway.logging_setup import log


# =============================================================================
# Legacy Data Classes (for backward compatibility)
# =============================================================================

@dataclass
class RoutingDecision:
    """Legacy routing decision - wraps new RoutingDecision."""
    provider: str        # "local" | "anthropic" | "openai" | etc
    model: str           # resolved model name
    tier: str            # "local" | "sonnet" | "opus" | "gpt4" | etc
    score: float
    reasons: List[str]
    phase: str           # "explicit" | "preference" | "cascade"
    
    @property
    def is_opus(self) -> bool:
        return self.tier == "opus" or "opus" in self.model.lower()
    
    @property
    def primary_model(self) -> str:
        """Compatibility with new RoutingDecision."""
        return self.model
    
    @property
    def cascade_chain(self) -> List[str]:
        """Compatibility with new RoutingDecision."""
        return [self.model]


@dataclass
class RoutingSignals:
    """Legacy routing signals - kept for compatibility."""
    band: str            # "LOCAL" | "CLAUDE" | "AMBIGUOUS"
    confidence: float    # 0.0-1.0
    reasons: List[str]
    fallback_tier: str   # "local" | "sonnet" | "opus"


# =============================================================================
# Main Routing Function
# =============================================================================

async def route_request(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] = None,
    project_id: Optional[int] = None,
    explicit_model: Optional[str] = None,
    system_prompt: str = "",
    cost_quality_bias: Optional[float] = None,
    speed_quality_bias: Optional[float] = None,
    cascade_enabled: Optional[bool] = None,
    max_cascade_attempts: Optional[int] = None,
) -> RoutingDecision:
    """
    Route a request to the appropriate model.
    
    This delegates to the new routing_engine with preference-based scoring.
    
    Args:
        messages: Conversation messages
        tools: Tool definitions (optional)
        project_id: Project ID for loading preferences (optional)
        explicit_model: Explicitly requested model (optional)
        system_prompt: System prompt text
        cost_quality_bias: Override cost preference (0=cheapest, 1=quality)
        speed_quality_bias: Override speed preference (0=fastest, 1=quality)
        cascade_enabled: Whether cascade is enabled (for preference object)
        max_cascade_attempts: Max cascade attempts (for preference object)
        
    Returns:
        RoutingDecision with selected model
    """
    if not ENABLE_SMART_ROUTING:
        # Smart routing disabled, use default model
        return RoutingDecision(
            provider="anthropic",
            model=DEFAULT_MODEL,
            tier="sonnet",
            score=1.0,
            reasons=["smart_routing_disabled"],
            phase="disabled",
        )
    
    # Handle explicit model request
    if explicit_model:
        provider, tier = _infer_provider_tier(explicit_model)
        return RoutingDecision(
            provider=provider,
            model=explicit_model,
            tier=tier,
            score=1.0,
            reasons=["explicit_model_requested"],
            phase="explicit",
        )
    
    # Delegate to new routing engine
    try:
        from gateway.routing_engine import get_routing_decision_async
        
        decision = await get_routing_decision_async(
            messages=messages,
            tools=tools or [],
            system_prompt=system_prompt,
            explicit_model=explicit_model,
            cost_quality_bias=cost_quality_bias,
            speed_quality_bias=speed_quality_bias,
            cascade_enabled=cascade_enabled,
            max_cascade_attempts=max_cascade_attempts,
            project_id=project_id,
        )
        
        # Convert to legacy format
        provider, tier = _infer_provider_tier(decision.primary_model)
        
        return RoutingDecision(
            provider=provider or decision.provider,
            model=decision.primary_model,
            tier=tier,
            score=decision.scores.get(decision.primary_model, 0.5),
            reasons=[decision.reasoning],
            phase="preference",
        )
        
    except Exception as e:
        log.warning("Routing engine failed, using default: %r", e)
        return RoutingDecision(
            provider="anthropic",
            model=DEFAULT_MODEL,
            tier="sonnet",
            score=0.5,
            reasons=[f"routing_fallback:{str(e)[:50]}"],
            phase="fallback",
        )


def _infer_provider_tier(model: str) -> tuple:
    """Infer provider and tier from model name."""
    model_lower = model.lower()
    
    # Provider detection
    if "claude" in model_lower:
        provider = "anthropic"
    elif "gpt" in model_lower or "o1" in model_lower:
        provider = "openai"
    elif "gemini" in model_lower:
        provider = "gemini"
    elif "llama" in model_lower or "mixtral" in model_lower:
        if "/" in model:
            provider = "ollama"
        else:
            provider = "groq"
    elif "ollama/" in model_lower or ":" in model_lower:
        provider = "ollama"
    else:
        provider = "anthropic"
    
    # Tier detection
    if "opus" in model_lower:
        tier = "opus"
    elif "sonnet" in model_lower:
        tier = "sonnet"
    elif "haiku" in model_lower:
        tier = "haiku"
    elif "gpt-4o" in model_lower:
        tier = "gpt4o"
    elif "gpt-4" in model_lower:
        tier = "gpt4"
    elif "flash" in model_lower:
        tier = "flash"
    elif provider == "ollama":
        tier = "local"
    else:
        tier = "standard"
    
    return provider, tier


# =============================================================================
# Helper Functions (kept for compatibility)
# =============================================================================

def is_simple_question_last_message(messages: List[Dict]) -> bool:
    """Check if the last user message is a simple question."""
    SIMPLE_PATTERNS = [
        "explain this", "what does", "how do i", "can you explain",
        "help me understand", "summarize", "what is",
    ]
    
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        
        content_lower = content.lower()[:500]
        
        return any(p in content_lower for p in SIMPLE_PATTERNS)
    
    return False


def get_user_intent_text(messages: List[Dict], max_chars: int = 600) -> str:
    """Extract the user's intent from the last message."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        
        return content[:max_chars] if content else ""
    
    return ""


# =============================================================================
# Deprecated Functions (log warnings)
# =============================================================================

def compute_routing_signals(messages: List[Dict], tools: List[Dict]) -> RoutingSignals:
    """Deprecated: Use routing_engine.extract_request_features instead."""
    log.warning("compute_routing_signals is deprecated - use routing_engine")
    return RoutingSignals(
        band="CLAUDE",
        confidence=0.5,
        reasons=["deprecated_function"],
        fallback_tier="sonnet",
    )


async def determine_model(messages: List[Dict], tools: List[Dict], project_id: Optional[int] = None) -> str:
    """Deprecated: Use route_request instead."""
    log.warning("determine_model is deprecated - use route_request")
    decision = await route_request(messages, tools, project_id)
    return decision.model
