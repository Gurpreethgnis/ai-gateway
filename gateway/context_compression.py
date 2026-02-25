"""
Context Compression - Compress conversation history for smaller context windows.

Layer 4 of the multi-layer caching strategy.
Summarizes older messages when routing to local models with smaller context windows.
"""

from typing import List, Dict, Any, Optional

from gateway.logging_setup import log
from gateway import config


# Models used for compression (fast, small)
COMPRESSION_MODEL = getattr(config, "COMPRESSION_MODEL", "ollama/llama3.1:8b")


async def compress_context_for_local(
    messages: List[Dict],
    max_tokens: int,
    preserve_recent: int = 4,
) -> List[Dict]:
    """
    Compress conversation history for local models with smaller context windows.
    
    Strategy:
    1. Preserve the most recent N messages
    2. Summarize older messages using a fast local model
    3. Return compressed message list
    
    Args:
        messages: Full conversation messages
        max_tokens: Maximum tokens allowed for target model
        preserve_recent: Number of recent messages to keep verbatim
        
    Returns:
        Compressed message list
    """
    # Estimate current token count
    estimated_tokens = sum(
        len(_get_message_text(m)) // 4 
        for m in messages
    )
    
    if estimated_tokens <= max_tokens:
        return messages  # No compression needed
    
    if len(messages) <= preserve_recent:
        return messages  # Not enough messages to compress
    
    # Split into preserve vs summarize
    recent = messages[-preserve_recent:]
    older = messages[:-preserve_recent]
    
    if not older:
        return messages
    
    log.info(
        "Compressing context: %d messages (%d tokens) -> keeping %d recent",
        len(messages), estimated_tokens, preserve_recent
    )
    
    try:
        # Try to summarize using local model
        summary = await _summarize_messages(older)
        
        if summary:
            # Return compressed history
            return [
                {
                    "role": "user",
                    "content": f"[CONTEXT SUMMARY: {summary}]",
                },
                *recent,
            ]
    except Exception as e:
        log.warning("Context compression failed, using truncation: %r", e)
    
    # Fallback: simple truncation
    return _truncate_messages(messages, max_tokens)


async def _summarize_messages(messages: List[Dict]) -> Optional[str]:
    """
    Summarize a list of messages using a fast local model.
    
    Returns:
        Summary string or None if summarization fails
    """
    from gateway.providers.registry import get_provider_registry
    
    registry = get_provider_registry()
    provider = registry.get("ollama")
    
    if not provider:
        return None
    
    # Format messages for summarization
    formatted = _format_messages_for_summary(messages)
    
    summary_prompt = f"""Summarize this conversation history concisely. Focus on:
- Key decisions made
- Important context that would be needed to continue
- Any code or technical details mentioned

Conversation:
{formatted}

Summary (be brief, 2-3 sentences max):"""
    
    try:
        # Use configured compression model
        model = COMPRESSION_MODEL.split("/")[-1]  # Remove optional provider prefix
        
        response = await provider.complete(
            messages=[{"role": "user", "content": summary_prompt}],
            model=model,
            max_tokens=500,
            temperature=0.3,  # Low temperature for consistent summaries
        )
        
        return response.content.strip()
        
    except Exception as e:
        log.warning("Message summarization failed: %r", e)
        return None


def _format_messages_for_summary(messages: List[Dict]) -> str:
    """Format messages into a readable string for summarization."""
    lines = []
    
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = _get_message_text(msg)
        
        # Truncate very long messages
        if len(content) > 500:
            content = content[:500] + "..."
        
        lines.append(f"{role}: {content}")
    
    return "\n\n".join(lines)


def _get_message_text(message: Dict) -> str:
    """Extract text content from a message."""
    content = message.get("content", "")
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(f"[Tool: {block.get('content', '')}]")
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    
    return str(content)


def _truncate_messages(messages: List[Dict], max_tokens: int) -> List[Dict]:
    """
    Simple truncation fallback - keep most recent messages.
    
    Removes oldest messages until under token limit.
    """
    result = list(messages)
    
    while len(result) > 2:  # Keep at least system + user
        estimated_tokens = sum(
            len(_get_message_text(m)) // 4 
            for m in result
        )
        
        if estimated_tokens <= max_tokens:
            break
        
        # Remove oldest non-system message
        for i, msg in enumerate(result):
            if msg.get("role") != "system":
                result.pop(i)
                break
    
    return result


async def prepare_context_for_provider(
    messages: List[Dict],
    provider: str,
    model: str,
    project_id: Optional[int] = None,
) -> List[Dict]:
    """
    Prepare context optimally for target provider.
    
    Applies:
    1. File deduplication (all providers)
    2. Context compression (if needed for local)
    3. Provider-specific optimizations
    
    Args:
        messages: Original messages
        provider: Target provider name
        model: Target model ID
        project_id: Optional project for file cache lookup
        
    Returns:
        Optimized message list
    """
    from gateway.model_registry import get_model_registry
    from gateway.file_cache import process_tool_result
    
    # Step 1: File deduplication
    processed_messages = []
    for msg in messages:
        content = msg.get("content", "")
        
        # Process tool results for file deduplication
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, str):
                        processed = await process_tool_result(
                            project_id, tool_content, block.get("tool_use_id", "")
                        )
                        new_content.append({**block, "content": processed})
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)
            processed_messages.append({**msg, "content": new_content})
        else:
            processed_messages.append(msg)
    
    # Step 2: Context compression (if needed for local models)
    if provider == "ollama":
        registry = get_model_registry()
        model_info = registry.get_model(model)
        
        if model_info:
            max_ctx = int(model_info.context_window * 0.8)  # Leave room for response
            processed_messages = await compress_context_for_local(
                processed_messages, max_ctx
            )
    
    return processed_messages
