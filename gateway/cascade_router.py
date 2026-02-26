"""
Cascade routing: try local first, escalate to cloud if quality is insufficient.

This implements the FrugalGPT-style cascade pattern:
1. Get routing decision from routing_engine (preference-based scoring)
2. If provider is "ollama", try local first
3. Check response quality
4. If quality passes, return local response
5. If quality fails, get next best model from routing_engine
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


async def route_with_cascade(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    project_id: Optional[int],
    system_prompt: str,
    explicit_model: Optional[str] = None,
    cost_quality_bias: Optional[float] = None,
    speed_quality_bias: Optional[float] = None,
    cascade_enabled: Optional[bool] = None,
    max_cascade_attempts: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Tuple[Any, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Route with cascade: try preferred model first, escalate if quality check fails.
    
    Uses the new preference-based routing engine for model selection.
    
    Args:
        messages: Conversation messages
        tools: Tool definitions
        project_id: Optional project ID
        system_prompt: System prompt text
        explicit_model: Explicit model request (if any)
        cost_quality_bias: Optional override (0=cheapest, 1=highest quality)
        speed_quality_bias: Optional override (0=fastest, 1=highest quality)
        cascade_enabled: Whether cascade (try local first) is enabled for this project
        max_cascade_attempts: Max escalation attempts (passed to routing preferences)
    
    Returns:
        (decision, local_response, cascade_metadata) where:
        - decision: Final routing decision (RoutingDecision from routing_engine)
        - local_response: Local model response if tried and passed, None otherwise
        - cascade_metadata: Dict with cascade info for logging
    """
    from gateway.routing_engine import get_routing_decision_async

    def _routing_kwargs():
        return dict(
            messages=messages,
            tools=tools,
            system_prompt=system_prompt,
            explicit_model=explicit_model,
            cost_quality_bias=cost_quality_bias,
            speed_quality_bias=speed_quality_bias,
            cascade_enabled=cascade_enabled,
            max_cascade_attempts=max_cascade_attempts,
            project_id=project_id,
        )

    if not ENABLE_CASCADE_ROUTING or (cascade_enabled is not None and not cascade_enabled):
        # Cascade disabled globally or per-project, use standard routing directly
        decision = await get_routing_decision_async(**_routing_kwargs())
        return decision, None, {"cascade_enabled": False}
    
    t0 = time.time()
    
    # Phase 1: Get routing decision from new routing engine
    initial_decision = await get_routing_decision_async(**_routing_kwargs())
    
    cascade_metadata = {
        "cascade_enabled": True,
        "session_id": session_id,
        "initial_model": initial_decision.primary_model,
        "initial_provider": initial_decision.provider,
        "initial_score": initial_decision.scores.get(initial_decision.primary_model, 0),
        "escalated": False,
        "escalation_reason": None,
        "local_attempt_content": None,
        "local_response_time_ms": None,
        "quality_score": None,
        "total_cascade_time_ms": None,
    }
    
    # If explicit model requested, skip cascade
    if explicit_model and explicit_model == initial_decision.primary_model:
        cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
        return initial_decision, None, cascade_metadata
    
    # Only cascade if initial decision is for local/Ollama
    if initial_decision.provider != "ollama":
        cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
        return initial_decision, None, cascade_metadata
    
    # Phase 2: Try local model
    log.info("Cascade: trying local model %s", initial_decision.primary_model)
    
    try:
        local_t0 = time.time()
        local_response = await _call_local_for_cascade(
            messages, system_prompt, initial_decision.primary_model
        )
        local_time_ms = int((time.time() - local_t0) * 1000)
        cascade_metadata["local_response_time_ms"] = local_time_ms
        
        if not local_response or "error" in local_response:
            # Local call failed, get next best model
            log.warning("Cascade: local call failed, escalating")
            cascade_metadata["local_attempt_content"] = (local_response or {}).get("content")
            escalated_decision = await _get_escalation_decision(
                messages, tools, system_prompt,
                exclude_provider="ollama",
                cost_quality_bias=cost_quality_bias,
                speed_quality_bias=speed_quality_bias,
                cascade_enabled=cascade_enabled,
                max_cascade_attempts=max_cascade_attempts,
                project_id=project_id,
            )
            cascade_metadata["escalated"] = True
            cascade_metadata["escalation_reason"] = "local_call_failed"
            cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
            return escalated_decision, None, cascade_metadata
        
        # Phase 3: Quality check
        from gateway.quality_check import check_response_quality
        
        response_text = local_response.get("content") or local_response.get("text") or ""
        query_text = _get_last_user_message_text(messages)
        
        passes, quality_score, fail_reason = check_response_quality(response_text, query_text)
        cascade_metadata["quality_score"] = quality_score
        
        if passes:
            # Quality passed, return local response
            log.info(
                "Cascade: local quality passed (score=%.2f, time=%dms)",
                quality_score, local_time_ms
            )
            cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
            return initial_decision, local_response, cascade_metadata
        
        # Quality failed, escalate
        log.info(
            "Cascade: local quality failed (score=%.2f, reason=%s), escalating",
            quality_score, fail_reason
        )
        cascade_metadata["local_attempt_content"] = response_text[:1200] if response_text else None
        escalated_decision = await _get_escalation_decision(
            messages, tools, system_prompt,
            exclude_provider="ollama",
            cost_quality_bias=cost_quality_bias,
            speed_quality_bias=speed_quality_bias,
            cascade_enabled=cascade_enabled,
            max_cascade_attempts=max_cascade_attempts,
            project_id=project_id,
        )
        cascade_metadata["escalated"] = True
        cascade_metadata["escalation_reason"] = fail_reason or "quality_check_failed"
        cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
        
        return escalated_decision, None, cascade_metadata
    
    except Exception as e:
        # Exception during cascade, escalate
        log.exception("Cascade: exception during local attempt: %r", e)
        escalated_decision = await _get_escalation_decision(
            messages, tools, system_prompt,
            exclude_provider="ollama",
            cost_quality_bias=cost_quality_bias,
            speed_quality_bias=speed_quality_bias,
            cascade_enabled=cascade_enabled,
            max_cascade_attempts=max_cascade_attempts,
            project_id=project_id,
        )
        cascade_metadata["escalated"] = True
        cascade_metadata["escalation_reason"] = f"exception:{str(e)[:30]}"
        cascade_metadata["total_cascade_time_ms"] = int((time.time() - t0) * 1000)
        return escalated_decision, None, cascade_metadata


async def _get_escalation_decision(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    system_prompt: str,
    exclude_provider: str,
    cost_quality_bias: Optional[float] = None,
    speed_quality_bias: Optional[float] = None,
    cascade_enabled: Optional[bool] = None,
    max_cascade_attempts: Optional[int] = None,
    project_id: Optional[int] = None,
):
    """
    Get the next best model, excluding the specified provider.
    """
    from gateway.routing_engine import get_routing_decision_async

    return await get_routing_decision_async(
        messages=messages,
        tools=tools,
        system_prompt=system_prompt,
        explicit_model=None,
        cost_quality_bias=cost_quality_bias,
        speed_quality_bias=speed_quality_bias,
        cascade_enabled=cascade_enabled,
        max_cascade_attempts=max_cascade_attempts,
        exclude_providers=[exclude_provider],
        project_id=project_id,
    )


async def _call_local_for_cascade(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    """
    Call local Ollama for cascade attempt with context truncation.
    
    Truncates context to fit local model's window:
    - Last 5-8 messages only
    - Strips all tool_result and tool_use blocks (local can't use tools)
    - Limits system prompt to 2000 chars
    - Ensures total stays under LOCAL_CONTEXT_CHAR_LIMIT
    
    Args:
        messages: Conversation messages
        system_prompt: System prompt
        model: Local model to use
    
    Returns:
        Local response dict with "content" key, or None on error
    """
    try:
        from gateway.providers.ollama_provider import OllamaProvider
        from gateway.config import LOCAL_CONTEXT_CHAR_LIMIT
        
        # Truncate system prompt (keep first 2000 chars)
        truncated_system = system_prompt[:2000] if system_prompt else ""
        
        # Keep only last 8 messages
        recent_messages = messages[-8:] if len(messages) > 8 else messages
        
        # Convert messages, stripping tool blocks
        processed_messages = []
        if truncated_system:
            processed_messages.append({"role": "system", "content": truncated_system})
        
        total_chars = len(truncated_system)
        
        for msg in recent_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Extract text from blocks, skip tool blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type in ("tool_result", "tool_use"):
                            continue
                        if block_type == "text":
                            text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = " ".join(text_parts)
            
            if not content or not content.strip():
                continue
            
            if total_chars + len(content) > LOCAL_CONTEXT_CHAR_LIMIT:
                # Truncate to fit
                remaining = LOCAL_CONTEXT_CHAR_LIMIT - total_chars
                if remaining > 100:
                    content = content[:remaining]
                else:
                    break
            
            processed_messages.append({"role": role, "content": content})
            total_chars += len(content)
        
        log.debug("Cascade: calling local with %d messages, %d chars", 
                  len(processed_messages), total_chars)
        
        # Call Ollama via provider
        provider = OllamaProvider()
        try:
            await provider.ensure_model_loaded(model or LOCAL_LLM_DEFAULT_MODEL)
        except Exception:
            pass
        response = await provider.complete(
            messages=processed_messages,
            model=model or LOCAL_LLM_DEFAULT_MODEL,
            temperature=0.2,
            max_tokens=2000,
        )
        
        return {"content": response.content, "model": response.model}
    
    except Exception as e:
        log.exception("Error calling local for cascade: %r", e)
        return {"error": str(e)}


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
