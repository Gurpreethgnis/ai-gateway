import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import httpx

from gateway.config import ENABLE_PLUGIN_TOOLS, DATABASE_URL
from gateway.logging_setup import log


@dataclass
class PluginToolDefinition:
    id: int
    project_id: int
    name: str
    description: str
    input_schema: Dict[str, Any]
    endpoint_url: str
    is_active: bool


async def get_project_plugin_tools(project_id: int) -> List[PluginToolDefinition]:
    if not ENABLE_PLUGIN_TOOLS or not DATABASE_URL:
        return []

    try:
        from gateway.db import get_session, PluginTool
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(PluginTool).where(
                    PluginTool.project_id == project_id,
                    PluginTool.is_active == True,
                )
            )
            rows = result.scalars().all()

            return [
                PluginToolDefinition(
                    id=row.id,
                    project_id=row.project_id,
                    name=row.name,
                    description=row.description,
                    input_schema=json.loads(row.input_schema_json or "{}"),
                    endpoint_url=row.endpoint_url,
                    is_active=row.is_active,
                )
                for row in rows
            ]

    except Exception as e:
        log.warning("Failed to get plugin tools: %r", e)
        return []


async def register_plugin_tool(
    project_id: int,
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    endpoint_url: str,
) -> Optional[int]:
    if not ENABLE_PLUGIN_TOOLS or not DATABASE_URL:
        return None

    try:
        from gateway.db import get_session, PluginTool
        from sqlalchemy import select

        async with get_session() as session:
            existing = await session.execute(
                select(PluginTool).where(
                    PluginTool.project_id == project_id,
                    PluginTool.name == name,
                )
            )
            existing_tool = existing.scalar_one_or_none()

            if existing_tool:
                existing_tool.description = description
                existing_tool.input_schema_json = json.dumps(input_schema)
                existing_tool.endpoint_url = endpoint_url
                existing_tool.is_active = True
                return existing_tool.id

            tool = PluginTool(
                project_id=project_id,
                name=name,
                description=description,
                input_schema_json=json.dumps(input_schema),
                endpoint_url=endpoint_url,
                is_active=True,
            )
            session.add(tool)
            await session.flush()
            return tool.id

    except Exception as e:
        log.warning("Failed to register plugin tool: %r", e)
        return None


async def unregister_plugin_tool(project_id: int, name: str) -> bool:
    if not DATABASE_URL:
        return False

    try:
        from gateway.db import get_session, PluginTool
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(PluginTool).where(
                    PluginTool.project_id == project_id,
                    PluginTool.name == name,
                )
            )
            tool = result.scalar_one_or_none()

            if tool:
                tool.is_active = False
                return True

            return False

    except Exception as e:
        log.warning("Failed to unregister plugin tool: %r", e)
        return False


def build_tools_with_plugins(
    original_tools: List[Dict[str, Any]],
    plugins: List[PluginToolDefinition],
) -> List[Dict[str, Any]]:
    tools = list(original_tools)

    for plugin in plugins:
        tool_def = {
            "type": "function",
            "function": {
                "name": f"plugin_{plugin.name}",
                "description": plugin.description,
                "parameters": plugin.input_schema,
            },
        }
        tools.append(tool_def)

    return tools


def is_plugin_tool_call(tool_name: str) -> bool:
    return tool_name.startswith("plugin_")


def get_plugin_name(tool_name: str) -> str:
    if tool_name.startswith("plugin_"):
        return tool_name[7:]
    return tool_name


async def execute_plugin_tool(
    project_id: int,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: float = 30.0,
) -> Dict[str, Any]:
    if not ENABLE_PLUGIN_TOOLS:
        return {"error": "Plugin tools not enabled"}

    actual_name = get_plugin_name(tool_name)

    tools = await get_project_plugin_tools(project_id)
    matching = [t for t in tools if t.name == actual_name]

    if not matching:
        return {"error": f"Plugin tool '{actual_name}' not found"}

    plugin = matching[0]

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                plugin.endpoint_url,
                json={
                    "tool_name": actual_name,
                    "arguments": arguments,
                    "project_id": project_id,
                },
                headers={"Content-Type": "application/json"},
            )

            if response.status_code >= 400:
                return {
                    "error": f"Plugin endpoint returned {response.status_code}",
                    "details": response.text[:500],
                }

            return response.json()

    except httpx.TimeoutException:
        return {"error": f"Plugin tool '{actual_name}' timed out after {timeout}s"}
    except Exception as e:
        log.error("Plugin tool execution failed: %r", e)
        return {"error": str(e)}


async def handle_plugin_tool_calls(
    project_id: int,
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results = []

    for tc in tool_calls:
        tool_name = tc.get("function", {}).get("name", "")

        if not is_plugin_tool_call(tool_name):
            continue

        try:
            arguments = json.loads(tc.get("function", {}).get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}

        result = await execute_plugin_tool(project_id, tool_name, arguments)

        results.append({
            "tool_call_id": tc.get("id", ""),
            "name": tool_name,
            "result": result,
        })

    return results


def separate_plugin_tool_calls(
    tool_calls: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    regular = []
    plugin = []

    for tc in tool_calls:
        name = tc.get("function", {}).get("name", "")
        if is_plugin_tool_call(name):
            plugin.append(tc)
        else:
            regular.append(tc)

    return regular, plugin
