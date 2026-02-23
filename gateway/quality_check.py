"""
Quality verification for cascade routing.

When the gateway routes to local Ollama first, we need to verify the response
quality before returning it. If quality is insufficient, we escalate to Claude.
"""

import re
from typing import Dict, Any, Optional, Tuple
from gateway.config import CASCADE_QUALITY_CHECK_MODE
from gateway.logging_setup import log


def check_response_quality(
    response_text: str,
    query_text: str,
    check_mode: str = CASCADE_QUALITY_CHECK_MODE,
) -> Tuple[bool, float, Optional[str]]:
    """
    Check if a local model's response meets quality standards.
    
    Args:
        response_text: The response from local model
        query_text: The user's original query
        check_mode: "heuristic", "llm", or "none"
    
    Returns:
        (passes, score, reason) where:
        - passes: True if quality is acceptable
        - score: 0.0-1.0 quality score
        - reason: Human-readable reason if failed, None if passed
    """
    if check_mode == "none":
        return True, 1.0, None
    
    if check_mode == "heuristic":
        return _heuristic_quality_check(response_text, query_text)
    
    if check_mode == "llm":
        # Future: ask local model to self-assess confidence
        # For now, fall back to heuristic
        log.debug("LLM quality check mode not yet implemented, using heuristic")
        return _heuristic_quality_check(response_text, query_text)
    
    # Unknown mode, default to pass
    log.warning("Unknown quality check mode: %s, defaulting to pass", check_mode)
    return True, 1.0, None


def _heuristic_quality_check(response_text: str, query_text: str) -> Tuple[bool, float, Optional[str]]:
    """
    Fast heuristic checks (< 50ms) for response quality.
    
    Returns:
        (passes, score, reason)
    """
    if not response_text or not response_text.strip():
        return False, 0.0, "empty_response"
    
    response_lower = response_text.lower().strip()
    query_lower = query_text.lower().strip()
    
    # Check 1: Refusal patterns
    refusal_patterns = [
        "i don't know",
        "i can't help",
        "i cannot assist",
        "i'm not able to",
        "i don't have access",
        "i apologize, but i",
        "sorry, i can't",
        "i'm unable to",
        "i do not have the ability",
        "i'm not sure",
        "i don't understand",
    ]
    
    for pattern in refusal_patterns:
        if pattern in response_lower[:200]:  # Check first 200 chars for refusal
            return False, 0.1, f"refusal_pattern:{pattern}"
    
    # Don't use length as a quality signal: short correct answers ("6", "Yes") are valid.
    # We only rejected empty above. No minimum-length failure.
    
    # Check 2: Code block presence when code is requested
    # Only require code blocks when the user's actual ask is for code, not when pasted context
    # mentions "code". Short factual questions ("what is 4+2") should not require code blocks.
    code_keywords = ["code", "function", "implement", "write", "create", "build", "fix", "debug"]
    query_asks_for_code = any(keyword in query_lower for keyword in code_keywords)
    # Skip code-block requirement for short simple questions (e.g. "what is 4+2", "explain X")
    simple_factual = (
        len(query_lower) < 120
        and any(p in query_lower for p in ["what is", "what's", "how much", "why is", "explain "])
        and not any(k in query_lower for k in ["code", "implement", "function", "write ", "build ", "fix ", "debug "])
    )
    if query_asks_for_code and not simple_factual:
        # If code was requested, response should have code blocks
        code_block_count = response_text.count("```")
        if code_block_count < 2:  # Should have at least opening and closing
            # Exception: if response is asking clarifying questions (brainstorming mode)
            if not any(q in response_lower for q in ["?", "question", "clarify", "which", "what do you mean"]):
                return False, 0.4, "code_requested_but_no_blocks"
    
    # Check 3: Repetition detection
    if _has_excessive_repetition(response_text):
        return False, 0.2, "excessive_repetition"
    
    # Check 4: Hallucination markers (model talking about itself incorrectly)
    hallucination_markers = [
        "as an ai language model",
        "i was trained by",
        "my knowledge cutoff",
        "i'm a large language model",
    ]
    
    for marker in hallucination_markers:
        if marker in response_lower:
            # Not necessarily bad, but suspicious in a code context
            pass  # Don't fail, just note
    
    # No length-based failure: short correct answers are valid. Skip "very_short_sentences" check.
    
    # All checks passed - compute quality score
    score = 0.7  # Base score for passing
    
    # Bonus for good length
    if len(response_text) > 300:
        score += 0.1
    
    # Bonus for code blocks when appropriate
    if any(keyword in query_lower for keyword in code_keywords):
        if response_text.count("```") >= 2:
            score += 0.1
    
    # Bonus for structured response (numbered lists, bullet points)
    if re.search(r'^\d+\.', response_text, re.MULTILINE) or re.search(r'^[-*]\s', response_text, re.MULTILINE):
        score += 0.1
    
    score = min(score, 1.0)
    
    return True, score, None


def _has_excessive_repetition(text: str, max_repeat: int = 3) -> bool:
    """
    Check if text has excessive repetition (model stuck in a loop).
    
    Args:
        text: Response text
        max_repeat: Maximum allowed consecutive repetitions
    
    Returns:
        True if excessive repetition detected
    """
    # Check for repeated words
    words = text.lower().split()
    if len(words) < 10:
        return False
    
    for i in range(len(words) - max_repeat * 3):
        # Check if same word appears 4+ times in a row
        word = words[i]
        if len(word) > 3:  # Ignore short words like "the", "a"
            consecutive_count = 1
            for j in range(i + 1, min(i + 10, len(words))):
                if words[j] == word:
                    consecutive_count += 1
                    if consecutive_count > max_repeat:
                        return True
                else:
                    break
    
    # Check for repeated phrases (3+ words)
    for i in range(len(words) - 9):
        phrase = " ".join(words[i:i+3])
        rest_text = " ".join(words[i+3:])
        if rest_text.count(phrase) >= 2:
            return True
    
    return False


def compute_quality_metadata(response_text: str, query_text: str) -> Dict[str, Any]:
    """
    Compute quality metadata for logging/analysis.
    
    Args:
        response_text: The response from local model
        query_text: The user's original query
    
    Returns:
        Dict with quality metrics
    """
    return {
        "response_length": len(response_text),
        "query_length": len(query_text),
        "code_block_count": response_text.count("```") // 2,
        "sentence_count": len([s for s in response_text.split('.') if s.strip()]),
        "has_questions": "?" in response_text,
    }
