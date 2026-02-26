"""
Routing Engine - Preference-based model selection.

Replaces rule-based routing with preference-driven scoring that automatically
selects models based on user-configured cost/quality and speed/quality trade-offs.
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from gateway.logging_setup import log


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RoutingPreferences:
    """User-configurable routing preferences."""
    cost_quality_bias: float = 0.5      # 0=cheapest, 1=best quality
    speed_quality_bias: float = 0.5     # 0=fastest, 1=best quality
    cascade_enabled: bool = True
    max_cascade_attempts: int = 2
    
    def __post_init__(self):
        # Clamp values to valid range
        self.cost_quality_bias = max(0.0, min(1.0, self.cost_quality_bias))
        self.speed_quality_bias = max(0.0, min(1.0, self.speed_quality_bias))


@dataclass
class RequestFeatures:
    """Extracted features from an incoming request."""
    estimated_tokens: int = 0
    has_tools: bool = False
    has_images: bool = False
    task_complexity: float = 0.5    # 0.0 (trivial) - 1.0 (expert-level)
    is_code_task: bool = False
    conversation_turns: int = 1
    last_message_length: int = 0
    has_tool_results: bool = False  # Mid-loop tool processing
    
    # Detailed signals
    file_count: int = 0
    deep_reasoning_keywords: int = 0
    simple_question: bool = False


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    primary_model: str              # Best match model ID
    provider: str                   # Provider name
    cascade_chain: List[str]        # Ordered fallback list of model IDs
    scores: Dict[str, float]        # All model scores for debugging
    reasoning: str                  # Human-readable explanation
    features: RequestFeatures       # Extracted features for logging


# =============================================================================
# Feature Extraction
# =============================================================================

# Patterns for task classification
CODE_TASK_PATTERNS = [
    r"```", r"def\s+\w+", r"function\s+\w+", r"class\s+\w+",
    r"\.py\b", r"\.js\b", r"\.ts\b", r"\.java\b", r"\.go\b", r"\.rs\b",
    r"import\s+", r"from\s+\w+\s+import", r"require\s*\(",
    r"implement", r"refactor", r"debug", r"fix\s+(?:the\s+)?(?:bug|error|issue)",
    r"write\s+(?:a\s+)?(?:function|method|class|code)",
]

SIMPLE_QUESTION_PATTERNS = [
    r"^explain\s+", r"^what\s+(?:is|does|are)", r"^how\s+(?:do|does|can|to)",
    r"^can\s+you\s+explain", r"^help\s+me\s+understand",
    r"^summarize", r"^describe", r"^tell\s+me\s+about",
]

DEEP_REASONING_KEYWORDS = [
    "architect", "architecture", "migrate", "migration", "security audit",
    "incident", "postmortem", "distributed system", "breaking change",
    "backwards compat", "production deployment", "system design",
    "scalability", "infrastructure", "kubernetes", "terraform",
    "performance optimization", "database schema", "api design",
]


def extract_request_features(messages: List[Dict], body: Dict) -> RequestFeatures:
    """
    Extract routing-relevant features from a request.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        body: Full request body
        
    Returns:
        RequestFeatures with extracted signals
    """
    features = RequestFeatures()
    
    # Estimate tokens from message content
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            # Handle content blocks (images, text, etc.)
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total_chars += len(block.get("text", ""))
                    elif block.get("type") == "image":
                        features.has_images = True
                elif isinstance(block, str):
                    total_chars += len(block)
    
    features.estimated_tokens = total_chars // 4  # Rough estimate
    features.conversation_turns = len([m for m in messages if m.get("role") == "user"])
    
    # Check for tools
    features.has_tools = bool(body.get("tools") or body.get("functions"))
    
    # Get last user message for analysis
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_user_msg = content
            elif isinstance(content, list):
                last_user_msg = " ".join(
                    b.get("text", "") for b in content 
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            break
    
    features.last_message_length = len(last_user_msg)
    
    # Check for tool results (mid-loop processing)
    features.has_tool_results = any(
        msg.get("role") == "user" and 
        isinstance(msg.get("content"), list) and
        any(b.get("type") == "tool_result" for b in msg.get("content", []) if isinstance(b, dict))
        for msg in messages
    )
    
    # Code task detection
    combined_text = last_user_msg.lower()
    features.is_code_task = any(
        re.search(pattern, combined_text, re.IGNORECASE)
        for pattern in CODE_TASK_PATTERNS
    )
    
    # Simple question detection
    features.simple_question = any(
        re.search(pattern, combined_text, re.IGNORECASE)
        for pattern in SIMPLE_QUESTION_PATTERNS
    )
    
    # Deep reasoning keyword count
    features.deep_reasoning_keywords = sum(
        1 for kw in DEEP_REASONING_KEYWORDS
        if kw.lower() in combined_text
    )
    
    # File reference count (heuristic)
    file_patterns = [r"[\w/]+\.\w{1,5}\b", r"```\w+\s*\n"]
    for pattern in file_patterns:
        features.file_count += len(re.findall(pattern, last_user_msg))
    
    # Compute task complexity (0-1)
    complexity_signals = [
        features.has_tools * 0.15,
        features.has_images * 0.10,
        min(features.conversation_turns / 20, 1.0) * 0.15,
        min(features.estimated_tokens / 50000, 1.0) * 0.20,
        min(features.deep_reasoning_keywords / 3, 1.0) * 0.25,
        min(features.file_count / 5, 1.0) * 0.15,
    ]
    features.task_complexity = sum(complexity_signals)
    
    # Reduce complexity for simple questions
    if features.simple_question and features.conversation_turns <= 2:
        features.task_complexity *= 0.5
    
    return features


# =============================================================================
# Model Scoring
# =============================================================================

def score_model(
    model: "ModelInfo",  # From model_registry
    features: RequestFeatures,
    prefs: RoutingPreferences,
) -> float:
    """
    Score a model for a given request based on user preferences.
    
    Higher score = better match.
    
    Args:
        model: ModelInfo from registry
        features: Extracted request features
        prefs: User routing preferences
        
    Returns:
        Score between 0.0 and 1.0
    """
    # Normalize cost (inverse - lower cost = higher score)
    # Use log scale since costs vary by orders of magnitude
    max_cost = 20.0  # Approximate max (Opus at $15/M input)
    if model.cost_per_1m_input > 0:
        cost_score = 1.0 - (math.log1p(model.cost_per_1m_input) / math.log1p(max_cost))
    else:
        cost_score = 1.0  # Free models get max cost score
    
    # Normalize latency (inverse - lower latency = higher score)
    max_latency = 2000  # 2 seconds max expected
    speed_score = 1.0 - min(model.avg_latency_ms / max_latency, 1.0)
    
    # Base quality score
    quality_score = model.quality_rating
    
    # Task-specific quality adjustments
    if features.is_code_task and model.code_quality_boost > 1.0:
        quality_score *= model.code_quality_boost
    
    # Clamp quality to [0, 1]
    quality_score = min(quality_score, 1.0)
    
    # Complexity adjustment: complex tasks weight quality more heavily
    complexity_weight = 0.5 + (features.task_complexity * 0.5)  # 0.5 to 1.0
    
    # Apply user preferences
    # cost_quality_bias: 0 = all cost, 1 = all quality
    cost_vs_quality = (
        (1.0 - prefs.cost_quality_bias) * cost_score +
        prefs.cost_quality_bias * quality_score * complexity_weight
    )
    
    # speed_quality_bias: 0 = all speed, 1 = all quality
    speed_vs_quality = (
        (1.0 - prefs.speed_quality_bias) * speed_score +
        prefs.speed_quality_bias * quality_score * complexity_weight
    )
    
    # Final score: geometric mean of both preference axes
    # This ensures both dimensions contribute
    final_score = math.sqrt(cost_vs_quality * speed_vs_quality)
    
    return final_score


def score_local_model(
    model: "ModelInfo",
    features: RequestFeatures,
    prefs: RoutingPreferences,
) -> float:
    """
    Score local (Ollama) models when in local-only mode.
    
    Since all local models are free, cost is irrelevant - we focus on
    speed vs quality trade-off and task specialization.
    """
    quality = model.quality_rating
    
    # Task specialization boost
    if features.is_code_task and "coder" in model.id.lower():
        quality *= 1.20
    
    # Long context preference
    if features.estimated_tokens > 16000 and model.context_window >= 128000:
        quality *= 1.10
    
    # Clamp quality
    quality = min(quality, 1.0)
    
    # Speed score (normalized)
    speed = 1.0 - min(model.avg_latency_ms / 2000, 1.0)
    
    # Apply speed_quality_bias (cost doesn't matter for local)
    final = (
        (1.0 - prefs.speed_quality_bias) * speed +
        prefs.speed_quality_bias * quality
    )
    
    return final


# =============================================================================
# Routing Decision
# =============================================================================

def filter_eligible_models(
    features: RequestFeatures,
    models: List["ModelInfo"],
) -> List["ModelInfo"]:
    """
    Filter models based on hard constraints.
    
    A model is ineligible if:
    - Tools required but not supported
    - Images present but no vision support
    - Context too large for model window
    - Model is disabled
    """
    eligible = []
    
    for model in models:
        # Skip disabled models
        if not model.is_enabled:
            continue
        
        # Tools required
        if features.has_tools and not model.supports_tools:
            log.debug("Model %s filtered: no tool support", model.id)
            continue
        
        # Vision required
        if features.has_images and not model.supports_vision:
            log.debug("Model %s filtered: no vision support", model.id)
            continue
        
        # Context window (leave 20% for response)
        max_input = int(model.context_window * 0.8)
        if features.estimated_tokens > max_input:
            log.debug(
                "Model %s filtered: context %d > max %d",
                model.id, features.estimated_tokens, max_input
            )
            continue
        
        eligible.append(model)
    
    return eligible


def get_routing_decision(
    features: RequestFeatures,
    prefs: RoutingPreferences,
    registry: "ModelRegistry",
    exclude_providers: Optional[List[str]] = None,
    enabled_models: Optional[List[Any]] = None,
) -> RoutingDecision:
    """
    Make routing decision based on request features and preferences.

    Args:
        features: Extracted request features
        prefs: User routing preferences
        registry: Model registry with available models
        exclude_providers: List of provider names to exclude
        enabled_models: If set, use this list instead of registry.get_enabled_models() (e.g. project overrides)

    Returns:
        RoutingDecision with primary model and cascade chain
    """
    all_models = enabled_models if enabled_models is not None else registry.get_enabled_models()

    # Exclude specified providers
    if exclude_providers:
        all_models = [m for m in all_models if m.provider not in exclude_providers]
    
    if not all_models:
        raise ValueError("No models available in registry")
    
    # Filter by hard constraints
    eligible = filter_eligible_models(features, all_models)
    
    if not eligible:
        # Fallback: relax constraints and try again with any enabled model
        log.warning("No eligible models after filtering, using any available")
        eligible = [m for m in all_models if m.is_enabled]
        if not eligible:
            raise ValueError("No enabled models available")
    
    # Check if all eligible models are local (free)
    all_local = all(m.provider == "ollama" for m in eligible)
    
    # Score models
    if all_local:
        scored = [(m, score_local_model(m, features, prefs)) for m in eligible]
    else:
        scored = [(m, score_model(m, features, prefs)) for m in eligible]
    
    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    
    primary = scored[0][0]
    primary_score = scored[0][1]
    
    # Build cascade chain based on preferences
    if prefs.cascade_enabled:
        if prefs.cost_quality_bias < 0.3:
            # Cost-conscious: include more models for fallback
            chain = [m.id for m, s in scored[:prefs.max_cascade_attempts + 1]]
        elif prefs.cost_quality_bias > 0.7:
            # Quality-focused: fewer fallbacks, trust the best
            chain = [primary.id]
        else:
            # Balanced: top 2 models
            chain = [m.id for m, s in scored[:min(2, len(scored))]]
    else:
        chain = [primary.id]
    
    # Build reasoning string
    reasoning_parts = [
        f"Selected {primary.id} (score={primary_score:.3f})",
        f"Preferences: cost_quality={prefs.cost_quality_bias:.1f}, speed_quality={prefs.speed_quality_bias:.1f}",
        f"Task: complexity={features.task_complexity:.2f}, code={features.is_code_task}, tools={features.has_tools}",
    ]
    
    if len(chain) > 1:
        reasoning_parts.append(f"Cascade: {' -> '.join(chain)}")
    
    return RoutingDecision(
        primary_model=primary.id,
        provider=primary.provider,
        cascade_chain=chain,
        scores={m.id: s for m, s in scored},
        reasoning=" | ".join(reasoning_parts),
        features=features,
    )


# =============================================================================
# High-Level API
# =============================================================================

async def route_request(
    messages: List[Dict],
    body: Dict,
    preferences: Optional[RoutingPreferences] = None,
    registry: Optional["ModelRegistry"] = None,
    exclude_providers: Optional[List[str]] = None,
    project_id: Optional[int] = None,
) -> RoutingDecision:
    """
    Main entry point for routing a request.

    Args:
        messages: Conversation messages
        body: Full request body
        preferences: User preferences (uses defaults if None)
        registry: Model registry (uses global if None)
        exclude_providers: List of provider names to exclude
        project_id: If set, only models enabled for this project are considered (dashboard toggles)

    Returns:
        RoutingDecision
    """
    from gateway.model_registry import get_model_registry

    if preferences is None:
        from gateway.config import (
            DEFAULT_COST_QUALITY_BIAS,
            DEFAULT_SPEED_QUALITY_BIAS,
            DEFAULT_CASCADE_ENABLED,
            DEFAULT_MAX_CASCADE_ATTEMPTS,
        )
        preferences = RoutingPreferences(
            cost_quality_bias=DEFAULT_COST_QUALITY_BIAS,
            speed_quality_bias=DEFAULT_SPEED_QUALITY_BIAS,
            cascade_enabled=DEFAULT_CASCADE_ENABLED,
            max_cascade_attempts=DEFAULT_MAX_CASCADE_ATTEMPTS,
        )

    if registry is None:
        registry = get_model_registry()

    # Apply project-specific model toggles (disabled models in dashboard are excluded)
    enabled_models = await registry.get_enabled_models_for_project(project_id)

    # Extract features
    features = extract_request_features(messages, body)

    # Get routing decision
    decision = get_routing_decision(
        features, preferences, registry,
        exclude_providers=exclude_providers,
        enabled_models=enabled_models,
    )
    
    log.info(
        "Routing decision: %s (score=%.3f) | %s",
        decision.primary_model,
        decision.scores.get(decision.primary_model, 0),
        decision.reasoning,
    )
    
    return decision


# Convenience wrapper for cascade_router compatibility
async def get_routing_decision_async(
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    explicit_model: Optional[str] = None,
    cost_quality_bias: Optional[float] = None,
    speed_quality_bias: Optional[float] = None,
    cascade_enabled: Optional[bool] = None,
    max_cascade_attempts: Optional[int] = None,
    exclude_providers: Optional[List[str]] = None,
    project_id: Optional[int] = None,
) -> RoutingDecision:
    """
    Async wrapper for routing decision with simplified interface.

    This is the preferred interface for cascade_router and other callers.
    project_id: When set, only models enabled for this project are used (respects dashboard toggles).
    """
    from gateway.config import (
        DEFAULT_COST_QUALITY_BIAS,
        DEFAULT_SPEED_QUALITY_BIAS,
        DEFAULT_CASCADE_ENABLED,
        DEFAULT_MAX_CASCADE_ATTEMPTS,
    )
    from gateway.model_registry import get_model_registry

    # Build body dict
    body = {
        "messages": messages,
        "system": system_prompt,
    }
    if tools:
        body["tools"] = tools
    if explicit_model:
        body["model"] = explicit_model

    # Build preferences (use loaded values; fall back to config defaults when None)
    prefs = RoutingPreferences(
        cost_quality_bias=cost_quality_bias if cost_quality_bias is not None else DEFAULT_COST_QUALITY_BIAS,
        speed_quality_bias=speed_quality_bias if speed_quality_bias is not None else DEFAULT_SPEED_QUALITY_BIAS,
        cascade_enabled=cascade_enabled if cascade_enabled is not None else DEFAULT_CASCADE_ENABLED,
        max_cascade_attempts=max_cascade_attempts if max_cascade_attempts is not None else DEFAULT_MAX_CASCADE_ATTEMPTS,
    )

    registry = get_model_registry()

    # If explicit model requested and it exists, use it directly (still respect project enable/disable)
    if explicit_model:
        model_info = registry.get_model(explicit_model)
        if model_info:
            enabled_for_project = await registry.get_enabled_models_for_project(project_id)
            if any(m.id == explicit_model for m in enabled_for_project):
                features = extract_request_features(messages, body)
                return RoutingDecision(
                    primary_model=explicit_model,
                    provider=model_info.provider,
                    cascade_chain=[explicit_model],
                    scores={explicit_model: 1.0},
                    reasoning=f"Explicit model requested: {explicit_model}",
                    features=features,
                )
            # Explicit model is disabled for this project - fall through to normal routing

    return await route_request(
        messages=messages,
        body=body,
        preferences=prefs,
        registry=registry,
        exclude_providers=exclude_providers,
        project_id=project_id,
    )


# Alias for backwards compatibility
get_routing_decision_wrapper = get_routing_decision_async
