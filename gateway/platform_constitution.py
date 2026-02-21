"""
Platform Constitution - Stable System Rules for Anthropic Prompt Caching

This module contains stable, cacheable system prompts that rarely change.
By separating these from dynamic per-request instructions, we enable
Anthropic's prompt caching to read these blocks at ~10% cost on subsequent requests.

Version the constitution when making changes to invalidate old cache entries.

Usage: Do NOT auto-inject these blocks into user requests. The gateway must act as
a transparent proxy: the client's system prompt is forwarded as-is. This module is
for opt-in use only (e.g. when a client explicitly requests gateway rules via a
header or parameter), so that applications relying on specific model behavior are
not modified or given unrequested instructions.
"""

VERSION = "1.0.0"

PLATFORM_CONSTITUTION = """
# AI Gateway Platform Rules

## Core Principles
- Maintain consistency across all API responses
- Prioritize security and data privacy
- Optimize for token efficiency and cost reduction
- Support OpenAI-compatible interfaces with Anthropic backends

## Code Quality Standards
- Write clean, maintainable code
- Follow language-specific best practices
- Use type hints and documentation
- Prefer composition over inheritance
- Keep functions focused and single-purpose

## Response Guidelines
- Be concise and accurate
- Provide actionable information
- Include relevant context when needed
- Avoid unnecessary verbosity

## Tool Usage
- Only use tools when necessary
- Validate tool inputs before execution
- Handle tool errors gracefully
- Return structured, parseable results

## File Operations
- Prefer reading specific files over entire directories
- Use diffs for modifications when possible
- Validate file paths and permissions
- Handle large files efficiently
"""

DIFF_FIRST_RULES = """
DIFF-FIRST EDITING POLICY:
- When modifying files, respond with unified diffs (git-style) unless user explicitly asks for full file.
- Prefer minimal patches touching the smallest region.
- If you need file context, ask via tool calls for specific files/lines instead of requesting the whole repo.
- Never paste entire large files unless requested; output a patch + brief rationale.
"""

def get_cacheable_system_blocks(include_constitution: bool = True, include_diff_rules: bool = True) -> list:
    """
    Returns system prompt blocks with proper cache control for Anthropic.
    
    Args:
        include_constitution: Include the platform constitution (highly cacheable)
        include_diff_rules: Include diff-first editing policy (highly cacheable)
        
    Returns:
        List of content blocks with cache_control directives
    """
    blocks = []
    
    if include_constitution:
        blocks.append({
            "type": "text",
            "text": PLATFORM_CONSTITUTION.strip(),
            "cache_control": {"type": "ephemeral"}  # Will be changed to "permanent" in next phase
        })
    
    if include_diff_rules:
        blocks.append({
            "type": "text",
            "text": DIFF_FIRST_RULES.strip(),
            "cache_control": {"type": "ephemeral"}  # Will be changed to "permanent" in next phase
        })
    
    return blocks
