"""
Lightweight skill system: curated prompt templates invokable via X-Gateway-Skill header.

Skills are structured prompts that guide the AI through specific methodologies.
Instead of installing 883+ external skills, we maintain 5-10 high-value templates
directly in the gateway.
"""

from typing import Optional, Dict, Any
from gateway.logging_setup import log


SKILLS: Dict[str, Dict[str, Any]] = {
    "brainstorming": {
        "description": "Structured feature planning with requirements gathering",
        "system_prefix": """You are in BRAINSTORMING mode. Follow this process:

1. UNDERSTAND: Ask clarifying questions about the goal (one at a time)
2. REQUIREMENTS: List functional and non-functional requirements
3. CONSTRAINTS: Identify technical constraints and dependencies
4. DESIGN: Propose 2-3 approaches with tradeoffs
5. DECISION: Recommend one approach with rationale
6. SPEC: Output a concise specification document

Do NOT write code until the spec is approved. Ask questions one at a time.
Be thorough but concise.""",
        "force_tier": None,  # Let smart routing decide
    },
    
    "systematic-debugging": {
        "description": "Methodical debugging with hypothesis testing",
        "system_prefix": """You are in DEBUGGING mode. Follow this process:

1. SYMPTOMS: What exactly is happening? Get specifics.
2. EXPECTED: What should happen instead?
3. REPRODUCE: Can we isolate a minimal reproduction?
4. HYPOTHESES: List 3 possible causes, ranked by likelihood
5. TEST: For each hypothesis, what would confirm/deny it?
6. INVESTIGATE: Check the most likely cause first
7. FIX: Apply minimal fix, explain why it works
8. VERIFY: Confirm the fix and check for regressions

Be methodical. Don't guess randomly. Follow the scientific method.""",
        "force_tier": None,
    },
    
    "code-review": {
        "description": "Thorough code review checklist",
        "system_prefix": """You are in CODE REVIEW mode. Check the following:

1. CORRECTNESS: Does the code do what it's supposed to?
2. EDGE CASES: Are error cases and boundaries handled?
3. READABILITY: Is the code clear? Are names meaningful?
4. PATTERNS: Does it follow project conventions and best practices?
5. SECURITY: Any injection risks, auth bypasses, or data leaks?
6. PERFORMANCE: Any obvious inefficiencies or bottlenecks?
7. TESTS: Are there tests? Do they cover important cases?
8. DOCUMENTATION: Is anything non-obvious explained?

Provide specific, actionable feedback. Praise what's done well.
Prioritize critical issues over nitpicks.""",
        "force_tier": None,
    },
    
    "git-pushing": {
        "description": "Safe commit workflow with best practices",
        "system_prefix": """You are in GIT COMMIT mode. Follow this workflow:

1. CHANGES: Review what's changed (git status, git diff)
2. STAGE: Add relevant files (exclude temp/generated/secrets)
3. MESSAGE: Write a clear commit message:
   - First line: concise summary (imperative mood, <60 chars)
   - Body: why this change, not what (if non-obvious)
   - Reference issues if applicable
4. VERIFY: Run linters/tests before committing
5. COMMIT: Create the commit
6. PUSH: Push to remote (after verifying no force-push to main)

Do NOT commit secrets, credentials, or temp files.
Do NOT force-push to main/master without explicit user approval.""",
        "force_tier": None,
    },
    
    "concise-planning": {
        "description": "Break down large tasks into actionable steps",
        "system_prefix": """You are in PLANNING mode. Break down the task:

1. GOAL: Restate the objective in one sentence
2. SCOPE: What's included? What's explicitly excluded?
3. DEPENDENCIES: What needs to exist first?
4. STEPS: List 5-10 concrete, actionable steps
5. VALIDATION: How will we know each step is done?
6. RISKS: What could go wrong? Mitigations?

Keep it concise. Each step should be small enough to complete in one session.
Order steps by dependency, not difficulty.""",
        "force_tier": None,
    },
    
    "architecture-review": {
        "description": "System design and architecture analysis",
        "system_prefix": """You are in ARCHITECTURE REVIEW mode. Evaluate:

1. REQUIREMENTS: What are the system's functional and non-functional requirements?
2. DESIGN: How is the system structured? (components, data flow, APIs)
3. SCALABILITY: Will it handle growth? Bottlenecks?
4. RELIABILITY: Single points of failure? Error handling? Monitoring?
5. SECURITY: Attack surface? Auth/authz? Data protection?
6. MAINTAINABILITY: Is it understandable? Testable? Extensible?
7. TRADEOFFS: What's the rationale for key decisions?
8. ALTERNATIVES: What other approaches were considered?

Focus on high-level structure, not implementation details.
Recommend improvements with clear rationale.""",
        "force_tier": "opus",  # Architecture reviews need deep reasoning
    },
    
    "quick-fix": {
        "description": "Fast, minimal changes for simple issues",
        "system_prefix": """You are in QUICK FIX mode. Keep it minimal:

1. Identify the exact line(s) causing the issue
2. Apply the smallest possible fix
3. Verify it doesn't break anything nearby
4. Done. No refactoring, no "while we're here" changes.

This is for typos, off-by-one errors, missing imports, etc.
If the fix requires >5 lines changed, suggest switching to debugging mode.""",
        "force_tier": "local",  # Quick fixes can use local model
    },
    
    "security-audit": {
        "description": "Security vulnerability scan and hardening",
        "system_prefix": """You are in SECURITY AUDIT mode. Check for:

1. INJECTION: SQL, NoSQL, command, XSS, template injection
2. AUTH/AUTHZ: Broken authentication, authorization bypasses, session issues
3. DATA EXPOSURE: Sensitive data leaks, insufficient encryption
4. DEPENDENCIES: Known CVEs in packages
5. CONFIG: Hardcoded secrets, debug mode in prod, exposed admin panels
6. INPUT VALIDATION: Unchecked user input, file uploads, rate limiting
7. LOGIC FLAWS: Race conditions, insecure defaults, business logic bypasses

Prioritize by severity (critical > high > medium > low).
Provide remediation steps, not just identification.""",
        "force_tier": "opus",  # Security needs thorough analysis
    },
}


def get_skill(skill_id: str) -> Optional[Dict[str, Any]]:
    """
    Look up a skill by ID.
    
    Args:
        skill_id: The skill identifier (e.g., "brainstorming")
    
    Returns:
        Skill dict with description, system_prefix, force_tier, or None if not found
    """
    skill_id = skill_id.strip().lower()
    return SKILLS.get(skill_id)


def apply_skill_to_system_prompt(system_prompt: str, skill_id: str) -> str:
    """
    Prepend skill prompt to existing system prompt.
    
    Args:
        system_prompt: Existing system prompt
        skill_id: The skill identifier
    
    Returns:
        Enhanced system prompt with skill prefix, or original if skill not found
    """
    skill = get_skill(skill_id)
    if not skill:
        log.warning("Skill not found: %s", skill_id)
        return system_prompt
    
    skill_prefix = skill.get("system_prefix", "")
    if not skill_prefix:
        return system_prompt
    
    # Prepend skill prompt with separator
    separator = "\n\n" + ("=" * 80) + "\n\n"
    if system_prompt:
        return skill_prefix + separator + system_prompt
    return skill_prefix


def list_skills() -> Dict[str, str]:
    """
    Return a dict of skill_id -> description for all available skills.
    
    Returns:
        Dict mapping skill IDs to their descriptions
    """
    return {skill_id: skill["description"] for skill_id, skill in SKILLS.items()}


def get_skill_forced_tier(skill_id: str) -> Optional[str]:
    """
    Check if a skill forces a specific routing tier.
    
    Args:
        skill_id: The skill identifier
    
    Returns:
        "local", "sonnet", "opus", or None if skill doesn't force a tier
    """
    skill = get_skill(skill_id)
    if skill:
        return skill.get("force_tier")
    return None
