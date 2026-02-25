import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from gateway.config import (
    ENABLE_MULTI_PROJECT,
    DATABASE_URL,
    DEFAULT_MODEL,
    OPUS_MODEL,
    DEFAULT_MAX_TOKENS,
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    STRIP_IDE_BOILERPLATE,
    ENFORCE_DIFF_FIRST,
    DEFAULT_COST_QUALITY_BIAS,
    DEFAULT_SPEED_QUALITY_BIAS,
)
from gateway.logging_setup import log
from gateway.db import hash_api_key


@dataclass
class ProjectConfig:
    default_model: str = DEFAULT_MODEL
    opus_model: str = OPUS_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    rate_limit_rpm: int = RATE_LIMIT_REQUESTS_PER_MINUTE
    enforce_diff_first: bool = ENFORCE_DIFF_FIRST
    strip_ide_boilerplate: bool = STRIP_IDE_BOILERPLATE
    custom_system_prompt: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    blocked_tools: List[str] = field(default_factory=list)
    enable_memory: bool = False
    enable_repo_map: bool = False
    enable_smart_routing: bool = True
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectInfo:
    id: int
    name: str
    config: ProjectConfig
    is_active: bool


DEFAULT_CONFIG = ProjectConfig()


async def get_project_by_api_key(api_key: str) -> Optional[ProjectInfo]:
    if not ENABLE_MULTI_PROJECT or not DATABASE_URL:
        return None

    try:
        from gateway.db import get_session, Project
        from sqlalchemy import select

        key_hash = hash_api_key(api_key)

        async with get_session() as session:
            result = await session.execute(
                select(Project).where(
                    Project.api_key_hash == key_hash,
                    Project.is_active == True,
                )
            )
            project = result.scalar_one_or_none()

            if not project:
                return None

            config_data = json.loads(project.config_json or "{}")
            config = ProjectConfig(
                default_model=config_data.get("default_model", DEFAULT_MODEL),
                opus_model=config_data.get("opus_model", OPUS_MODEL),
                max_tokens=config_data.get("max_tokens", DEFAULT_MAX_TOKENS),
                rate_limit_rpm=project.rate_limit_rpm,
                enforce_diff_first=config_data.get("enforce_diff_first", ENFORCE_DIFF_FIRST),
                strip_ide_boilerplate=config_data.get("strip_ide_boilerplate", STRIP_IDE_BOILERPLATE),
                custom_system_prompt=config_data.get("custom_system_prompt"),
                allowed_tools=config_data.get("allowed_tools", []),
                blocked_tools=config_data.get("blocked_tools", []),
                enable_memory=config_data.get("enable_memory", False),
                enable_repo_map=config_data.get("enable_repo_map", False),
                enable_smart_routing=config_data.get("enable_smart_routing", True),
                custom_settings=config_data.get("custom_settings", {}),
            )

            return ProjectInfo(
                id=project.id,
                name=project.name,
                config=config,
                is_active=project.is_active,
            )

    except Exception as e:
        log.warning("Failed to get project by API key: %r", e)
        return None


async def get_project_by_id(project_id: int) -> Optional[ProjectInfo]:
    if not DATABASE_URL:
        return None

    try:
        from gateway.db import get_session, Project
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(Project).where(Project.id == project_id)
            )
            project = result.scalar_one_or_none()

            if not project:
                return None

            config_data = json.loads(project.config_json or "{}")
            config = ProjectConfig(
                default_model=config_data.get("default_model", DEFAULT_MODEL),
                opus_model=config_data.get("opus_model", OPUS_MODEL),
                max_tokens=config_data.get("max_tokens", DEFAULT_MAX_TOKENS),
                rate_limit_rpm=project.rate_limit_rpm,
                enforce_diff_first=config_data.get("enforce_diff_first", ENFORCE_DIFF_FIRST),
                strip_ide_boilerplate=config_data.get("strip_ide_boilerplate", STRIP_IDE_BOILERPLATE),
                custom_system_prompt=config_data.get("custom_system_prompt"),
                allowed_tools=config_data.get("allowed_tools", []),
                blocked_tools=config_data.get("blocked_tools", []),
                enable_memory=config_data.get("enable_memory", False),
                enable_repo_map=config_data.get("enable_repo_map", False),
                enable_smart_routing=config_data.get("enable_smart_routing", True),
                custom_settings=config_data.get("custom_settings", {}),
            )

            return ProjectInfo(
                id=project.id,
                name=project.name,
                config=config,
                is_active=project.is_active,
            )

    except Exception as e:
        log.warning("Failed to get project by ID: %r", e)
        return None


async def get_routing_preferences(project_id: Optional[int]) -> tuple:
    """
    Load saved routing preferences (cost/speed bias) for the project.
    Returns (cost_quality_bias, speed_quality_bias) for use in routing.
    When project_id is None or not found, uses first project by id; falls back to config defaults.
    """
    if not DATABASE_URL:
        return (DEFAULT_COST_QUALITY_BIAS, DEFAULT_SPEED_QUALITY_BIAS)
    try:
        from gateway.db import get_session
        from sqlalchemy import text

        pid = project_id
        if pid is None:
            async with get_session() as session:
                r = await session.execute(text("SELECT id FROM projects ORDER BY id ASC LIMIT 1"))
                row = r.fetchone()
                pid = row[0] if row else None
        if pid is None:
            return (DEFAULT_COST_QUALITY_BIAS, DEFAULT_SPEED_QUALITY_BIAS)

        async with get_session() as session:
            r = await session.execute(
                text("""
                    SELECT cost_quality_bias, speed_quality_bias
                    FROM projects WHERE id = :pid
                """),
                {"pid": pid},
            )
            row = r.fetchone()
        if row:
            cost = row[0] if row[0] is not None else DEFAULT_COST_QUALITY_BIAS
            speed = row[1] if row[1] is not None else DEFAULT_SPEED_QUALITY_BIAS
            return (float(cost), float(speed))
    except Exception as e:
        log.debug("Could not load routing preferences: %r", e)
    return (DEFAULT_COST_QUALITY_BIAS, DEFAULT_SPEED_QUALITY_BIAS)


async def update_project_config(project_id: int, config_updates: Dict[str, Any]) -> bool:
    if not DATABASE_URL:
        return False

    try:
        from gateway.db import get_session, Project
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(Project).where(Project.id == project_id)
            )
            project = result.scalar_one_or_none()

            if not project:
                return False

            existing_config = json.loads(project.config_json or "{}")
            existing_config.update(config_updates)
            project.config_json = json.dumps(existing_config)

            return True

    except Exception as e:
        log.warning("Failed to update project config: %r", e)
        return False


def filter_tools_for_project(
    tools: List[Dict[str, Any]],
    config: ProjectConfig,
) -> List[Dict[str, Any]]:
    if not config.allowed_tools and not config.blocked_tools:
        return tools

    filtered = []
    for tool in tools:
        tool_name = ""
        if tool.get("type") == "function":
            tool_name = tool.get("function", {}).get("name", "")
        else:
            tool_name = tool.get("name", "")

        if config.allowed_tools and tool_name not in config.allowed_tools:
            continue

        if config.blocked_tools and tool_name in config.blocked_tools:
            continue

        filtered.append(tool)

    return filtered


def apply_project_config(
    request_body: Dict[str, Any],
    config: ProjectConfig,
) -> Dict[str, Any]:
    result = request_body.copy()

    if not result.get("model"):
        result["model"] = config.default_model

    if not result.get("max_tokens"):
        result["max_tokens"] = config.max_tokens

    if config.custom_system_prompt:
        messages = result.get("messages", [])
        has_system = any(m.get("role") == "system" for m in messages)

        if not has_system:
            messages.insert(0, {
                "role": "system",
                "content": config.custom_system_prompt,
            })
            result["messages"] = messages

    if "tools" in result:
        result["tools"] = filter_tools_for_project(result["tools"], config)

    return result


async def get_or_create_default_project(api_key: str) -> Optional[ProjectInfo]:
    project = await get_project_by_api_key(api_key)
    if project:
        return project

    return ProjectInfo(
        id=0,
        name="default",
        config=DEFAULT_CONFIG,
        is_active=True,
    )
