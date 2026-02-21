import re
import difflib
from typing import Dict, Any, List, Tuple, Optional
from gateway.config import (
    STRIP_IDE_BOILERPLATE,
    TOOL_RESULT_MAX_CHARS,
    USER_MSG_MAX_CHARS,
    SYSTEM_MAX_CHARS,
    ENFORCE_DIFF_FIRST,
)

_SAFE_STRIP_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Only strip known IDE wrappers that don't contain tool schemas or instructions
    (re.compile(r"<\|im_start\|>\s*(?:user|assistant|system)\s*", re.IGNORECASE), ""),
    (re.compile(r"<\|im_end\|>", re.IGNORECASE), ""),
    (re.compile(r"^You are Continue(?:\.ai)?\.\s*", re.MULTILINE | re.IGNORECASE), ""),
]

# Allow-list: patterns that identify critical content that must NEVER be stripped
_CRITICAL_CONTENT_MARKERS: List[re.Pattern] = [
    re.compile(r"\btools?:\s*\[", re.IGNORECASE),  # Tool definitions
    re.compile(r"\bfunction\s+\w+\s*\(", re.IGNORECASE),  # Function signatures
    re.compile(r"input_schema|parameters", re.IGNORECASE),  # Tool schemas
    re.compile(r"##\s*safety|security|constraint", re.IGNORECASE),  # Safety instructions
]

def has_critical_content(s: str) -> bool:
    """Check if text contains tool schemas or safety instructions that must not be stripped."""
    return any(p.search(s) for p in _CRITICAL_CONTENT_MARKERS)

def surgical_strip_boilerplate(text: str) -> Tuple[str, int]:
    """
    Surgically remove only known IDE wrapper markers, preserving tool schemas and instructions.
    Returns (cleaned_text, chars_removed).
    """
    original_len = len(text)
    
    # Apply safe strip patterns
    for pattern, replacement in _SAFE_STRIP_PATTERNS:
        text = pattern.sub(replacement, text)
    
    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    
    chars_removed = original_len - len(text)
    return text, chars_removed

def strip_or_truncate(role: str, text: str, max_chars: int, allow_strip: bool) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"stripped": False, "truncated": False, "before": len(text or ""), "after": len(text or "")}
    if not text:
        return "", meta

    # Surgical stripping: only remove known wrappers, never delete entire messages with critical content
    if allow_strip and STRIP_IDE_BOILERPLATE and len(text) >= 500:
        if not has_critical_content(text):  # Only strip if no tool schemas/safety instructions
            text, chars_removed = surgical_strip_boilerplate(text)
            if chars_removed > 0:
                meta["stripped"] = True
                meta["after"] = len(text)

    if max_chars > 0 and len(text) > max_chars:
        meta["truncated"] = True
        head = text[: int(max_chars * 0.7)]
        tail = text[-int(max_chars * 0.3):]
        text = head + "\n\n[...TRUNCATED...]\n\n" + tail
        meta["after"] = len(text)
        return text, meta

    meta["after"] = len(text)
    return text, meta

def enforce_diff_first(system_text: str) -> str:
    """
    Diff-first rules are now injected via platform_constitution.py when caching is enabled.
    This function is kept for backwards compatibility but does nothing when caching is on.
    """
    if not ENFORCE_DIFF_FIRST:
        return system_text
    # Note: When ENABLE_ANTHROPIC_CACHE_CONTROL is on, diff rules come from platform_constitution
    # so we don't append them here. This prevents duplication.
    from gateway.config import ENABLE_ANTHROPIC_CACHE_CONTROL
    if ENABLE_ANTHROPIC_CACHE_CONTROL:
        return system_text
    # Legacy path: append rules if not using cacheable constitution
    from gateway.platform_constitution import DIFF_FIRST_RULES
    if DIFF_FIRST_RULES.strip() in (system_text or ""):
        return system_text
    return (system_text + "\n\n" + DIFF_FIRST_RULES).strip()

LIMITS = {
    "tool_result_max": TOOL_RESULT_MAX_CHARS,
    "user_msg_max": USER_MSG_MAX_CHARS,
    "system_max": SYSTEM_MAX_CHARS,
}


def validate_unified_diff(diff_text: str, original_content: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate that a unified diff is well-formed and can potentially be applied.
    Returns (is_valid, error_message).
    
    If original_content is provided, attempts to apply the patch to verify it works.
    """
    if not diff_text or not diff_text.strip():
        return False, "Empty diff"
    
    # Check for unified diff markers
    has_diff_header = bool(re.search(r'^(---|\+\+\+|@@)', diff_text, re.MULTILINE))
    if not has_diff_header:
        # Not a diff, might be full file content (which is valid as fallback)
        return True, None
    
    # Basic structure checks
    lines = diff_text.split('\n')
    has_context = any(line.startswith((' ', '-', '+')) for line in lines)
    has_hunks = bool(re.search(r'^@@ -\d+,?\d* \+\d+,?\d* @@', diff_text, re.MULTILINE))
    
    if has_diff_header and not has_hunks:
        return False, "Diff header present but no hunks found"
    
    # If we have original content, try to apply the patch
    if original_content is not None and has_hunks:
        try:
            # Extract the changed lines and see if they make sense
            # This is a basic heuristic; real patch application needs external tools
            added_lines = [l[1:] for l in lines if l.startswith('+') and not l.startswith('+++')]
            removed_lines = [l[1:] for l in lines if l.startswith('-') and not l.startswith('---')]
            
            # Check if removed lines exist in original
            if removed_lines:
                original_lines = original_content.split('\n')
                for rem_line in removed_lines[:5]:  # Check first 5 for efficiency
                    if rem_line.strip() and rem_line.strip() not in [ol.strip() for ol in original_lines]:
                        return False, f"Removed line not found in original: {rem_line[:50]}"
            
            return True, None
        except Exception as e:
            return False, f"Patch validation error: {str(e)}"
    
    # Passed basic structure checks
    return True, None


def suggest_diff_fallback(error_msg: str) -> str:
    """
    Generate a helpful message for the model when a diff fails validation.
    """
    return f"""
The previous diff could not be validated: {error_msg}

Please either:
1. Provide a corrected unified diff with proper context lines
2. Call a file-reading tool to get the current file content, then generate a new diff
3. Provide the complete updated file content (the gateway will accept this as a fallback)
"""

