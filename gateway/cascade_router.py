"""
Cascade routing: try local first, escalate to Claude if quality is insufficient.

This implements the FrugalGPT-style cascade pattern:
1. Route request (Phase 1 heuristics + Phase 2 LLM classifier)
2. If tier is "local", try local Ollama
3. Check response quality
4. If quality passes, return local response
5. If quality fails, escalate to Claude (Sonnet/Opus)
"""

import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from gateway.config import (
    ENABLE_CASCADE_ROUTING,
    CASCADE_LOG_OUTCOMES,
    LOCAL_LLM_DEFAULT_MODEL,
)
from gateway.logging_setup import log
from gateway.smart_routing import route_request, RoutingDecision
from gateway.quality_check import check_response_quality, compute_quality_metadata


async def route_with_cascade(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    project_id: Optional[int],
    system_prompt: str,
    explicit_model: Optional[str] = None,
) -> Tuple[RoutingDecision, Optional[Dict[str, Any]], Optional[str]]:
    """
    Route with cascade: try local first, escalate if quality check fails.
    
    Args:
        messages: Conversation messages
        tools: Tool definitions
        project_id: Optional project ID
        system_prompt: System prompt text
        explicit_model: Explicit model request (if any)
    
    Returns:
        (decision, local_response, cascade_metadata) where:
        - decision: Final routing decision (may differ from initial)
        - local_response: Local model response if tried and passed, None otherwise
        - cascade_metadata: Dict with cascade info for logging
    """
    if not ENABLE_CASCADE_ROUTING:
        # Cascade disabled, use standard routing
        decision = await route_request(messages, tools, project_id, explicit_model, system_prompt)
        return decision, None, {"cascade_enabled": False}
    
    t0 = time.time()
    
    # Phase 1: Initial routing decision
    initial_decision = await route_request(messages, tools, project_id, explicit_model, system_prompt)
    
    cascade_metadata = {
        "cascade_enabled": True,
        "initial_tier": initial_decision.tier,
        "initial_provider": initial_decision.provider,
        "initial_phase": initial_decision.phase,
        "escalated": False,
        "escalation_reason": None,
        "local_response_time_ms": None,
        "quality_score": None,
        "total_cascade_time_ms": None,
    }
    
    # If initial decision is not local, no cascade needed
    if initial_decision.provider != "local":
        cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
        return initial_decision, None, cascade_metadata
    
    # Phase 2: Try local model
    log.info("Cascade: trying local model (tier=%s, phase=%s)", initial_decision.tier, initial_decision.phase)
    
    try:
        local_t0 = time.time()
        local_response = await _call_local_for_cascade(messages, system_prompt)
        local_time_ms = int((time.time() - local_t0) * 1000)
        cascade_metadata["local_response_time_ms"] = local_time_ms
        
        if not local_response or "error" in local_response:
            # Local call failed
            log.warning("Cascade: local call failed, escalating to Claude")
            escalated_decision = _escalate_decision(initial_decision, "local_call_failed")
            cascade_metadata["escalated"] = True
            cascade_metadata["escalation_reason"] = "local_call_failed"
            cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
            return escalated_decision, None, cascade_metadata
        
        # Phase 3: Quality check
        response_text = local_response.get("content", "")
        query_text = _get_last_user_message_text(messages)
        
        passes, quality_score, fail_reason = check_response_quality(response_text, query_text)
        cascade_metadata["quality_score"] = quality_score
        
        if passes:
            # Quality passed, return local response
            log.info(
                "Cascade: local quality passed (score=%.2f, time=%dms), using local response",
                quality_score, local_time_ms
            )
            cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
            return initial_decision, local_response, cascade_metadata
        
        # Quality failed, escalate to Claude
        log.info(
            "Cascade: local quality failed (score=%.2f, reason=%s), escalating to Claude",
            quality_score, fail_reason
        )
        escalated_decision = _escalate_decision(initial_decision, fail_reason or "quality_check_failed")
        cascade_metadata["escalated"] = True
        cascade_metadata["escalation_reason"] = fail_reason or "quality_check_failed"
        cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
        
        return escalated_decision, None, cascade_metadata
    
    except Exception as e:
        # Exception during cascade, escalate
        log.exception("Cascade: exception during local attempt, escalating: %r", e)
        escalated_decision = _escalate_decision(initial_decision, f"exception:{str(e)[:30]}")
        cascade_metadata["escalated"] = True
        cascade_metadata["escalation_reason"] = f"exception:{str(e)[:30]}"
        cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
        return escalated_decision, None, cascade_metadata


async def _call_local_for_cascade(
    messages: List[Dict[str, Any]],
    system_prompt: str,
) -> Optional[Dict[str, Any]]:
    """
    Call local Ollama for cascade attempt.
    
    Args:
        messages: Conversation messages
        system_prompt: System prompt
    
    Returns:
        Local response dict with "content" key, or None on error
    """
    try:
        from gateway.providers.ollama import call_ollama
        
        # Convert messages to Ollama format
        ollama_messages = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Extract text from blocks if needed
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = " ".join(text_parts)
            
            ollama_messages.append({"role": role, "content": content})
        
        # Call Ollama
        response = await call_ollama(
            messages=ollama_messages,
            model=LOCAL_LLM_DEFAULT_MODEL,
            temperature=0.2,
            max_tokens=2000,
        )
        
        return response
    
    except Exception as e:
        log.exception("Error calling local for cascade: %r", e)
        return {"error": str(e)}


def _escalate_decision(initial_decision: RoutingDecision, reason: str) -> RoutingDecision:
    """
    Create an escalated routing decision (local -> sonnet/opus).
    
    Args:
        initial_decision: Original decision (tier=local)
        reason: Reason for escalation
    
    Returns:
        New routing decision targeting Claude
    """
    from gateway.config import DEFAULT_MODEL, OPUS_MODEL
    
    # Default to Sonnet for escalation
    # TODO: Could use keyword/complexity scoring here to decide sonnet vs opus
    escalated_model = DEFAULT_MODEL
    escalated_tier = "sonnet"
    
    # If initial decision had high complexity signals, escalate to Opus
    if any(r in initial_decision.reasons for r in ["deep_reasoning", "many_files", "architecture"]):
        escalated_model = OPUS_MODEL
        escalated_tier = "opus"
    
    return RoutingDecision(
        provider="anthropic",
        model=escalated_model,
        tier=escalated_tier,
        score=initial_decision.score,
        reasons=initial_decision.reasons + [f"escalated:{reason}"],
        phase="escalated_from_local",
    )


def _get_last_user_message_text(messages: List[Dict[str, Any]]) -> str:
    """Extract text from the last user message."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return " ".join(parts)
    return ""


def compute_query_hash(messages: List[Dict[str, Any]]) -> str:
    """
    Compute a hash of the last user message for routing outcome logging.
    
    Args:
        messages: Conversation messages
    
    Returns:
        SHA256 hash (hex string)
    """
    last_text = _get_last_user_message_text(messages)
    return hashlib.sha256(last_text.encode('utf-8')).hexdigest()


def should_log_routing_outcome() -> bool:
    """Check if routing outcomes should be logged."""
    return CASCADE_LOG_OUTCOMES
