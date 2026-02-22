import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from gateway.config import (
    ENABLE_SMART_ROUTING,
    OPUS_ROUTING_THRESHOLD,
    DEFAULT_MODEL,
    OPUS_MODEL,
    DATABASE_URL,
    SMART_ROUTING_MODE,
    LOCAL_CONTEXT_CHAR_LIMIT,
)
from gateway.logging_setup import log


@dataclass
class RoutingDecision:
    provider: str        # "local" | "anthropic"
    model: str           # resolved model name
    tier: str            # "local" | "sonnet" | "opus"
    score: float
    reasons: List[str]
    phase: str           # "explicit" | "heuristic" | "llm_classifier"
    
    # Backward compatibility properties
    @property
    def is_opus(self) -> bool:
        return self.tier == "opus"


@dataclass
class RoutingSignals:
    band: str            # "LOCAL" | "CLAUDE" | "AMBIGUOUS"
    confidence: float    # 0.0-1.0
    reasons: List[str]
    fallback_tier: str   # "local" | "sonnet" | "opus" - used if ambiguous and Phase 2 fails


OPUS_KEYWORDS = [
    ("architect", 0.15),
    ("architecture", 0.15),
    ("design", 0.10),
    ("security", 0.12),
    ("migration", 0.12),
    ("refactor", 0.10),
    ("production", 0.10),
    ("incident", 0.12),
    ("postmortem", 0.12),
    ("kubernetes", 0.10),
    ("terraform", 0.10),
    ("infrastructure", 0.10),
    ("scalability", 0.10),
    ("performance", 0.08),
    ("optimization", 0.08),
    ("database schema", 0.12),
    ("api design", 0.10),
    ("system design", 0.12),
    ("distributed", 0.10),
    ("microservice", 0.10),
    ("breaking change", 0.12),
    ("backwards compat", 0.10),
    ("deprecat", 0.08),
]

SONNET_KEYWORDS = [
    ("simple", -0.10),
    ("quick", -0.08),
    ("small change", -0.10),
    ("typo", -0.12),
    ("formatting", -0.10),
    ("comment", -0.08),
    ("rename", -0.08),
]


# =============================================================================
# Phase 1: Fast Heuristics (< 1ms)
# =============================================================================

LOCAL_TASK_PATTERNS = [
    "explain", "summarize", "translate", "comment", "format", "rename",
    "typo", "lint", "type hint", "docstring", "boilerplate", "snippet",
    "what does", "how do i", "what is", "can you explain", "help me understand",
    "add comment", "fix formatting", "add type hints", "generate docstring",
]

CLAUDE_DEEP_REASONING_KEYWORDS = [
    "architect", "architecture", "migrate", "migration", "security audit",
    "incident", "postmortem", "distributed system", "breaking change",
    "backwards compat", "production deployment", "system design",
    "scalability", "infrastructure", "kubernetes", "terraform",
]

# Patterns that indicate the user is asking a simple question (route to local even if tools are in the request)
SIMPLE_QUESTION_PATTERNS = [
    "explain this code", "explain the code", "what does this do", "what does it do",
    "how does this work", "how do i ", "can you explain", "help me understand",
    "what is this", "summarize this", "what is going on", "explain what",
]


def get_last_user_message_text(messages: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """Extract text from the last user message only (for intent detection)."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content[:max_chars] if max_chars else content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text = " ".join(parts)
            return text[:max_chars] if max_chars else text
    return ""


def is_simple_question_last_message(messages: List[Dict[str, Any]]) -> bool:
    """True if the last user message looks like a simple explanation/summary question."""
    last_text = get_last_user_message_text(messages, max_chars=600).strip().lower()
    if not last_text or len(last_text) > 500:
        return False
    return any(p in last_text for p in SIMPLE_QUESTION_PATTERNS) or any(
        p in last_text for p in LOCAL_TASK_PATTERNS
    )


def compute_routing_signals(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    system_prompt: str = "",
) -> RoutingSignals:
    """
    Phase 1: Fast heuristic-based classification.
    Returns LOCAL, CLAUDE, or AMBIGUOUS with confidence and reasons.
    """
    text = extract_text_from_messages(messages)
    text_lower = text.lower()
    total_chars = len(text)
    reasons = []
    
    # ========== FAVOR LOCAL FIRST: simple question in last message ==========
    # Clients (Cursor, etc.) often send tools on every request. If the user's
    # last message is a simple "explain the code" / "what does this do", route
    # to local even when tools are present; the model can answer without using tools.
    # Context size will be handled when sending to local (truncate), not when deciding.
    if is_simple_question_last_message(messages):
        return RoutingSignals(
            band="LOCAL",
            confidence=0.9,
            reasons=["simple_question_last_message", "last_message_suggests_explanation"],
            fallback_tier="local"
        )
    
    # ========== FORCE CLAUDE (high confidence) ==========
    
    # Active tool-use loop: check only last 4 messages for tool_result blocks
    # (old tool_result blocks from history don't affect current routing decision)
    recent_messages = messages[-4:] if len(messages) > 4 else messages
    for msg in recent_messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return RoutingSignals(
                        band="CLAUDE",
                        confidence=1.0,
                        reasons=["active_tool_result_blocks"],
                        fallback_tier="sonnet"
                    )
    
    # Tools/functions present and last message wasn't a simple question - need Claude for tool use
    if tools:
        return RoutingSignals(
            band="CLAUDE",
            confidence=1.0,
            reasons=["tools_present"],
            fallback_tier="sonnet"
        )
    
    # Very long context exceeds local model capacity
    if total_chars > LOCAL_CONTEXT_CHAR_LIMIT:
        return RoutingSignals(
            band="CLAUDE",
            confidence=1.0,
            reasons=[f"long_context:{total_chars}>{LOCAL_CONTEXT_CHAR_LIMIT}"],
            fallback_tier="sonnet"
        )
    
    # Long conversation (> 15 turns) indicates agentic session
    if len(messages) > 15:
        return RoutingSignals(
            band="CLAUDE",
            confidence=0.95,
            reasons=[f"long_conversation:{len(messages)}_turns"],
            fallback_tier="sonnet"
        )
    
    # System prompt contains Cursor/Continue agent markers
    system_lower = system_prompt.lower()
    agent_markers = ["cursor", "continue", "coding agent", "ide assistant", "code editor"]
    if any(marker in system_lower for marker in agent_markers):
        if len(system_prompt) > 1000:
            return RoutingSignals(
                band="CLAUDE",
                confidence=0.9,
                reasons=["agentic_system_prompt"],
                fallback_tier="sonnet"
            )
    
    # Deep reasoning keywords (architecture, security, etc.)
    # Check last user message only, not full conversation (avoids system prompt pollution)
    last_user_text = get_last_user_message_text(messages, max_chars=None).lower()
    deep_keywords_found = []
    for keyword in CLAUDE_DEEP_REASONING_KEYWORDS:
        if keyword in last_user_text:
            deep_keywords_found.append(keyword)
    
    if len(deep_keywords_found) >= 2:
        return RoutingSignals(
            band="CLAUDE",
            confidence=0.9,
            reasons=[f"deep_reasoning:{','.join(deep_keywords_found[:3])}"],
            fallback_tier="opus"
        )
    
    # Multiple file references (> 5 files)
    file_refs = re.findall(r"(?:\.py|\.js|\.ts|\.tsx|\.go|\.rs|\.java|\.cpp|\.c|\.h)\b", text)
    if len(file_refs) > 5:
        return RoutingSignals(
            band="CLAUDE",
            confidence=0.85,
            reasons=[f"many_files:{len(file_refs)}"],
            fallback_tier="sonnet"
        )
    
    # Multi-file change requests
    multi_file_indicators = ["across multiple files", "in all files", "every file", "all the files"]
    if any(indicator in text_lower for indicator in multi_file_indicators):
        return RoutingSignals(
            band="CLAUDE",
            confidence=0.85,
            reasons=["multi_file_request"],
            fallback_tier="sonnet"
        )
    
    # ========== FAVOR LOCAL (high confidence) ==========
    
    # Very short, simple request
    if len(messages) == 1 and total_chars < 3000:
        # Check for simple task patterns
        simple_patterns_found = []
        for pattern in LOCAL_TASK_PATTERNS:
            if pattern in text_lower:
                simple_patterns_found.append(pattern)
        
        if simple_patterns_found:
            # No system prompt or very short system prompt
            if not system_prompt or len(system_prompt) < 500:
                return RoutingSignals(
                    band="LOCAL",
                    confidence=0.9,
                    reasons=[f"simple_task:{simple_patterns_found[0]}", "short_context"],
                    fallback_tier="local"
                )
    
    # Single-file, simple edit request
    single_file_simple = [
        "add a comment", "fix the typo", "rename this", "format this",
        "add type hints", "generate docstring", "explain this code",
        "what does this do", "how does this work",
    ]
    if any(pattern in text_lower for pattern in single_file_simple):
        if len(file_refs) <= 1:
            return RoutingSignals(
                band="LOCAL",
                confidence=0.85,
                reasons=["simple_single_file_edit"],
                fallback_tier="local"
            )
    
    # ========== AMBIGUOUS (route to Phase 2) ==========
    
    # Some signals favor local
    local_score = 0.0
    if total_chars < 5000:
        local_score += 0.3
        reasons.append("moderate_context")
    
    if len(messages) <= 3:
        local_score += 0.2
        reasons.append("short_conversation")
    
    # Some signals favor Claude
    claude_score = 0.0
    if len(deep_keywords_found) == 1:
        claude_score += 0.4
        reasons.append(f"some_complexity:{deep_keywords_found[0]}")
    
    if len(file_refs) > 1:
        claude_score += 0.3
        reasons.append(f"multiple_files:{len(file_refs)}")
    
    code_blocks = text.count("```")
    if code_blocks > 2:
        claude_score += 0.2
        reasons.append(f"code_blocks:{code_blocks}")
    
    # Determine fallback based on score difference
    if local_score > claude_score:
        fallback = "local"
    elif claude_score > local_score + 0.3:
        fallback = "sonnet"
    else:
        fallback = "sonnet"  # Default to sonnet when truly ambiguous
    
    return RoutingSignals(
        band="AMBIGUOUS",
        confidence=abs(local_score - claude_score) / 2.0,  # Lower confidence when scores are close
        reasons=reasons or ["no_strong_signals"],
        fallback_tier=fallback
    )


# =============================================================================
# Explicit vs full model ID
# =============================================================================

def is_explicit_model_alias(model: Optional[str]) -> bool:
    """
    True only when the user sent an intent alias (e.g. "sonnet", "opus", "local"),
    not a full API model ID like "claude-sonnet-4-0". When the client sends a full
    ID we run smart routing (Phase 1/2) so local can be chosen; when they send
    an alias we honor it as explicit.
    """
    if not model or not str(model).strip():
        return False
    m = str(model).strip().lower()
    # Full Anthropic model IDs: do NOT treat as explicit — run Phase 1/2 (local-first)
    if m.startswith("claude-"):
        parts = m.split("-")
        if len(parts) >= 4:  # e.g. claude-sonnet-4-0, claude-opus-4-5
            return False
        if len(parts) == 3 and m[-1].isdigit():  # e.g. claude-sonnet-4
            return False
    # Explicit intent aliases
    if m in ("sonnet", "opus", "local", "ollama", "auto", "smartroute", "fast", "high", "thinking"):
        return True
    if m.startswith("local:") or m.startswith("ollama:"):
        return True
    # Short aliases
    if m in ("sonnet-4", "sonnet4", "claude-sonnet", "opus-4", "opus4", "claude-opus"):
        return True
    return False


# =============================================================================
# Phase 2 Integration & Main Entry Point
# =============================================================================

async def route_request(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    project_id: Optional[int] = None,
    explicit_model: Optional[str] = None,
    system_prompt: str = "",
) -> RoutingDecision:
    """
    Main routing entry point for Smart Routing v2.
    
    Orchestrates:
    1. Explicit model/provider checks
    2. Phase 1: Fast heuristics
    3. Phase 2: LLM classification (if ambiguous)
    4. Final decision with provider + model
    
    Args:
        messages: conversation messages
        tools: tool/function definitions
        project_id: optional project ID for historical scoring
        explicit_model: explicit model request (e.g., "opus", "sonnet", "local:qwen")
        system_prompt: system prompt text
    
    Returns:
        RoutingDecision with provider, model, tier, score, reasons, phase
    """
    from gateway.config import LOCAL_LLM_DEFAULT_MODEL
    
    # ========== Handle explicit requests (only for intent aliases, not full model IDs) ==========
    # When client sends e.g. "claude-sonnet-4-0", we run Phase 1/2 so local can be chosen.
    if explicit_model and is_explicit_model_alias(explicit_model):
        model_lower = explicit_model.lower()
        
        # Explicit local request
        if "local" in model_lower or "ollama" in model_lower:
            return RoutingDecision(
                provider="local",
                model=LOCAL_LLM_DEFAULT_MODEL,
                tier="local",
                score=1.0,
                reasons=["explicit_local_request"],
                phase="explicit"
            )
        
        # Explicit opus request
        if "opus" in model_lower or "high" in model_lower or "thinking" in model_lower:
            return RoutingDecision(
                provider="anthropic",
                model=OPUS_MODEL,
                tier="opus",
                score=1.0,
                reasons=["explicit_opus_request"],
                phase="explicit"
            )
        
        # Explicit sonnet request
        if "sonnet" in model_lower or "fast" in model_lower:
            return RoutingDecision(
                provider="anthropic",
                model=DEFAULT_MODEL,
                tier="sonnet",
                score=0.0,
                reasons=["explicit_sonnet_request"],
                phase="explicit"
            )
    
    # Check routing mode
    if SMART_ROUTING_MODE == "keyword":
        # Legacy mode: use old should_use_opus logic
        decision = await should_use_opus(messages, tools, project_id, explicit_model)
        # Convert to new format
        return RoutingDecision(
            provider="anthropic",
            model=decision.model,
            tier="opus" if decision.is_opus else "sonnet",
            score=decision.score,
            reasons=decision.reasons,
            phase="heuristic"
        )
    
    if not ENABLE_SMART_ROUTING:
        return RoutingDecision(
            provider="anthropic",
            model=DEFAULT_MODEL,
            tier="sonnet",
            score=0.0,
            reasons=["smart_routing_disabled"],
            phase="explicit"
        )
    
    # ========== Phase 1: Fast Heuristics ==========
    signals = compute_routing_signals(messages, tools, system_prompt)
    
    if signals.band == "LOCAL":
        # High confidence: route to local
        log.info(
            "Smart routing v2 -> LOCAL (confidence=%.2f, reasons=%s)",
            signals.confidence, signals.reasons[:3]
        )
        return RoutingDecision(
            provider="local",
            model=LOCAL_LLM_DEFAULT_MODEL,
            tier="local",
            score=signals.confidence,
            reasons=signals.reasons,
            phase="heuristic"
        )
    
    if signals.band == "CLAUDE":
        # High confidence: route to Claude, now decide sonnet vs opus
        # Score based on last user message intent, not full conversation history
        last_user_text = get_last_user_message_text(messages, max_chars=None)
        is_simple_q = is_simple_question_last_message(messages)
        keyword_score, keyword_reasons = compute_keyword_score(last_user_text)
        complexity_score, complexity_reasons = compute_complexity_score(messages, tools, last_user_text, is_simple_q)
        historical_score = await get_historical_score(project_id, OPUS_MODEL)
        
        total_score = keyword_score + complexity_score + historical_score
        all_reasons = signals.reasons + keyword_reasons[:2] + complexity_reasons[:2]
        
        if historical_score != 0:
            all_reasons.append(f"historical:{historical_score:.2f}")
        
        # Opus guard: only allow auto-opus if explicitly enabled
        from gateway.config import ALLOW_AUTO_OPUS
        is_opus = (total_score >= OPUS_ROUTING_THRESHOLD) and ALLOW_AUTO_OPUS
        model = OPUS_MODEL if is_opus else DEFAULT_MODEL
        tier = "opus" if is_opus else "sonnet"
        
        if is_opus:
            log.info(
                "Smart routing v2 -> OPUS (score=%.2f, threshold=%.2f, reasons=%s)",
                total_score, OPUS_ROUTING_THRESHOLD, all_reasons[:5]
            )
        else:
            log.info(
                "Smart routing v2 -> SONNET (score=%.2f, reasons=%s)",
                total_score, all_reasons[:3]
            )
        
        return RoutingDecision(
            provider="anthropic",
            model=model,
            tier=tier,
            score=total_score,
            reasons=all_reasons,
            phase="heuristic"
        )
    
    # ========== Phase 2: LLM Classifier ==========
    log.info("Smart routing v2: AMBIGUOUS, calling LLM classifier (reasons=%s)", signals.reasons[:3])
    
    try:
        from gateway.routing_classifier import classify_with_llm
        tier, reason = await classify_with_llm(messages)
        
        if tier == "local":
            log.info("Smart routing v2 -> LOCAL (phase2, reason=%s)", reason)
            return RoutingDecision(
                provider="local",
                model=LOCAL_LLM_DEFAULT_MODEL,
                tier="local",
                score=0.7,
                reasons=[f"llm_classified:{reason}"] + signals.reasons,
                phase="llm_classifier"
            )
        
        # tier is sonnet or opus - use it directly
        model = OPUS_MODEL if tier == "opus" else DEFAULT_MODEL
        log.info("Smart routing v2 -> %s (phase2, reason=%s)", tier.upper(), reason)
        
        return RoutingDecision(
            provider="anthropic",
            model=model,
            tier=tier,
            score=0.6,
            reasons=[f"llm_classified:{reason}"] + signals.reasons,
            phase="llm_classifier"
        )
    
    except Exception as e:
        log.warning("Smart routing v2: LLM classifier failed: %r, using fallback=%s", e, signals.fallback_tier)
        
        # Fall back to Phase 1's best guess
        if signals.fallback_tier == "local":
            return RoutingDecision(
                provider="local",
                model=LOCAL_LLM_DEFAULT_MODEL,
                tier="local",
                score=0.5,
                reasons=["llm_classifier_failed"] + signals.reasons,
                phase="heuristic"
            )
        
        model = OPUS_MODEL if signals.fallback_tier == "opus" else DEFAULT_MODEL
        return RoutingDecision(
            provider="anthropic",
            model=model,
            tier=signals.fallback_tier,
            score=0.5,
            reasons=["llm_classifier_failed"] + signals.reasons,
            phase="heuristic"
        )


def extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    texts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
    return " ".join(texts)


def compute_keyword_score(text: str) -> Tuple[float, List[str]]:
    text_lower = text.lower()
    score = 0.0
    reasons = []

    for keyword, weight in OPUS_KEYWORDS:
        if keyword in text_lower:
            score += weight
            reasons.append(f"+{keyword}:{weight:.2f}")

    for keyword, weight in SONNET_KEYWORDS:
        if keyword in text_lower:
            score += weight
            reasons.append(f"{keyword}:{weight:.2f}")

    return score, reasons


def compute_complexity_score(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    text: str,
    is_simple_question: bool = False,
) -> Tuple[float, List[str]]:
    """
    Compute complexity score based on last user message text.
    
    Args:
        messages: All messages (for tool_use detection)
        tools: Tool definitions
        text: Last user message text (not full conversation)
        is_simple_question: Whether the last message is a simple question
    """
    score = 0.0
    reasons = []

    # Don't penalize simple questions for having long conversations or many tools
    # These are properties of the client (Cursor), not the task complexity
    if not is_simple_question:
        if len(text) > 5000:
            addition = 0.15
            score += addition
            reasons.append(f"+long_context:{addition:.2f}")
        elif len(text) > 2000:
            addition = 0.08
            score += addition
            reasons.append(f"+medium_context:{addition:.2f}")

        if len(tools) > 8:
            addition = 0.15
            score += addition
            reasons.append(f"+many_tools:{addition:.2f}")
        elif len(tools) > 4:
            addition = 0.08
            score += addition
            reasons.append(f"+some_tools:{addition:.2f}")

    tool_uses = sum(1 for m in messages if m.get("tool_calls"))
    if tool_uses > 5:
        addition = 0.10
        score += addition
        reasons.append(f"+heavy_tool_use:{addition:.2f}")

    file_refs = len(re.findall(r"(?:\.py|\.js|\.ts|\.tsx|\.go|\.rs|\.java)\b", text))
    if file_refs > 10:
        addition = 0.12
        score += addition
        reasons.append(f"+many_files:{addition:.2f}")
    elif file_refs > 5:
        addition = 0.06
        score += addition
        reasons.append(f"+some_files:{addition:.2f}")

    return score, reasons


async def get_historical_score(project_id: Optional[int], model: str) -> float:
    if not DATABASE_URL or not project_id:
        return 0.0

    import asyncio
    try:
        from gateway.db import get_session, ModelSuccessRate
        from sqlalchemy import select

        async def _query():
            async with get_session() as session:
                result = await session.execute(
                    select(ModelSuccessRate).where(
                        ModelSuccessRate.project_id == project_id,
                        ModelSuccessRate.model == model,
                    )
                )
                return result.scalar_one_or_none()

        rate = await asyncio.wait_for(_query(), timeout=2.0)

        if rate:
            total = rate.success_count + rate.failure_count
            if total > 10:
                success_ratio = rate.success_count / total
                return (success_ratio - 0.5) * 0.2

    except asyncio.TimeoutError:
        log.warning("Historical score query timed out")
    except Exception as e:
        log.warning("Failed to get historical score: %r", e)

    return 0.0


async def record_outcome(
    project_id: Optional[int],
    model: str,
    success: bool,
):
    if not DATABASE_URL or not project_id:
        return

    try:
        from gateway.db import get_session, ModelSuccessRate
        from sqlalchemy import select
        from datetime import datetime

        async with get_session() as session:
            result = await session.execute(
                select(ModelSuccessRate).where(
                    ModelSuccessRate.project_id == project_id,
                    ModelSuccessRate.model == model,
                )
            )
            rate = result.scalar_one_or_none()

            if rate:
                if success:
                    rate.success_count += 1
                else:
                    rate.failure_count += 1
                rate.last_updated = datetime.utcnow()
            else:
                rate = ModelSuccessRate(
                    project_id=project_id,
                    model=model,
                    success_count=1 if success else 0,
                    failure_count=0 if success else 1,
                )
                session.add(rate)

    except Exception as e:
        log.warning("Failed to record outcome: %r", e)


async def should_use_opus(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    project_id: Optional[int] = None,
    explicit_model: Optional[str] = None,
) -> RoutingDecision:
    """
    Legacy function for backward compatibility with keyword-based routing.
    Only routes between Sonnet and Opus (not local).
    """
    if explicit_model and is_explicit_model_alias(explicit_model):
        model_lower = explicit_model.lower()
        if "opus" in model_lower or "high" in model_lower or "thinking" in model_lower:
            return RoutingDecision(
                provider="anthropic",
                model=OPUS_MODEL,
                tier="opus",
                score=1.0,
                reasons=["explicit_opus_request"],
                phase="explicit",
            )
        if "sonnet" in model_lower or "fast" in model_lower:
            return RoutingDecision(
                provider="anthropic",
                model=DEFAULT_MODEL,
                tier="sonnet",
                score=0.0,
                reasons=["explicit_sonnet_request"],
                phase="explicit",
            )

    if not ENABLE_SMART_ROUTING:
        return RoutingDecision(
            provider="anthropic",
            model=DEFAULT_MODEL,
            tier="sonnet",
            score=0.0,
            reasons=["smart_routing_disabled"],
            phase="explicit",
        )

    # Use last user message for scoring in legacy mode too
    last_user_text = get_last_user_message_text(messages, max_chars=None)
    is_simple_q = is_simple_question_last_message(messages)

    keyword_score, keyword_reasons = compute_keyword_score(last_user_text)
    complexity_score, complexity_reasons = compute_complexity_score(messages, tools, last_user_text, is_simple_q)

    historical_score = await get_historical_score(project_id, OPUS_MODEL)

    total_score = keyword_score + complexity_score + historical_score
    all_reasons = keyword_reasons + complexity_reasons

    if historical_score != 0:
        all_reasons.append(f"historical:{historical_score:.2f}")

    # Opus guard in legacy mode too
    from gateway.config import ALLOW_AUTO_OPUS
    is_opus = (total_score >= OPUS_ROUTING_THRESHOLD) and ALLOW_AUTO_OPUS
    model = OPUS_MODEL if is_opus else DEFAULT_MODEL
    tier = "opus" if is_opus else "sonnet"

    if is_opus:
        log.info(
            "Smart routing (legacy) -> OPUS (score=%.2f, threshold=%.2f, reasons=%s)",
            total_score,
            OPUS_ROUTING_THRESHOLD,
            all_reasons[:5],
        )

    return RoutingDecision(
        provider="anthropic",
        model=model,
        tier=tier,
        score=total_score,
        reasons=all_reasons,
        phase="heuristic",
    )


async def route_model(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    project_id: Optional[int] = None,
    explicit_model: Optional[str] = None,
) -> str:
    """
    Backward-compatibility wrapper that returns just the model string.
    New code should use route_request() instead.
    """
    decision = await should_use_opus(messages, tools, project_id, explicit_model)
    return decision.model
