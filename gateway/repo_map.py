import re
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from gateway.config import ENABLE_REPO_MAP, DATABASE_URL
from gateway.logging_setup import log


@dataclass
class RepoNode:
    path: str
    node_type: str
    symbols: List[str] = field(default_factory=list)
    children: List["RepoNode"] = field(default_factory=list)


PYTHON_SYMBOL_PATTERNS = [
    (r"^class\s+(\w+)", "class"),
    (r"^def\s+(\w+)", "function"),
    (r"^async\s+def\s+(\w+)", "async_function"),
]

JS_TS_SYMBOL_PATTERNS = [
    (r"^(?:export\s+)?class\s+(\w+)", "class"),
    (r"^(?:export\s+)?function\s+(\w+)", "function"),
    (r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(", "arrow_function"),
    (r"^(?:export\s+)?(?:async\s+)?function\s*\*?\s*(\w+)", "function"),
    (r"interface\s+(\w+)", "interface"),
    (r"type\s+(\w+)\s*=", "type"),
]


def extract_symbols(content: str, file_path: str) -> List[str]:
    symbols = []
    ext = PurePosixPath(file_path).suffix.lower()

    if ext in (".py",):
        patterns = PYTHON_SYMBOL_PATTERNS
    elif ext in (".js", ".ts", ".jsx", ".tsx", ".mjs"):
        patterns = JS_TS_SYMBOL_PATTERNS
    else:
        return symbols

    for line in content.split("\n"):
        line = line.strip()
        for pattern, symbol_type in patterns:
            match = re.match(pattern, line)
            if match:
                symbol_name = match.group(1)
                if not symbol_name.startswith("_"):
                    symbols.append(f"{symbol_type}:{symbol_name}")

    return symbols[:50]


async def update_repo_map(
    project_id: Optional[int],
    file_path: str,
    content: str,
):
    if not ENABLE_REPO_MAP or not DATABASE_URL or not project_id:
        return

    try:
        from gateway.db import get_session, RepoNode as RepoNodeModel
        from sqlalchemy import select

        symbols = extract_symbols(content, file_path)

        normalized_path = file_path.replace("\\", "/")
        if normalized_path.startswith("./"):
            normalized_path = normalized_path[2:]

        async with get_session() as session:
            result = await session.execute(
                select(RepoNodeModel).where(
                    RepoNodeModel.project_id == project_id,
                    RepoNodeModel.file_path == normalized_path,
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.symbols_json = json.dumps(symbols)
                existing.last_updated = datetime.utcnow()
            else:
                node = RepoNodeModel(
                    project_id=project_id,
                    file_path=normalized_path,
                    node_type="file",
                    symbols_json=json.dumps(symbols),
                    last_updated=datetime.utcnow(),
                )
                session.add(node)

    except Exception as e:
        log.warning("Failed to update repo map: %r", e)


async def get_repo_map(project_id: Optional[int]) -> Dict[str, Any]:
    if not DATABASE_URL or not project_id:
        return {"enabled": False, "nodes": []}

    try:
        from gateway.db import get_session, RepoNode as RepoNodeModel
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(RepoNodeModel)
                .where(RepoNodeModel.project_id == project_id)
                .order_by(RepoNodeModel.file_path)
            )
            rows = result.scalars().all()

            nodes = []
            for row in rows:
                symbols = json.loads(row.symbols_json or "[]")
                nodes.append({
                    "path": row.file_path,
                    "type": row.node_type,
                    "symbols": symbols,
                    "last_updated": row.last_updated.isoformat(),
                })

            return {"enabled": True, "nodes": nodes, "count": len(nodes)}

    except Exception as e:
        log.warning("Failed to get repo map: %r", e)
        return {"enabled": True, "nodes": [], "error": str(e)}


def build_tree(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}

    for node in nodes:
        parts = node["path"].split("/")
        current = tree

        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current[part] = {
                    "_type": "file",
                    "_symbols": node.get("symbols", []),
                }
            else:
                if part not in current:
                    current[part] = {"_type": "dir"}
                current = current[part]

    return tree


def format_tree_text(tree: Dict[str, Any], prefix: str = "", is_last: bool = True) -> str:
    lines = []

    items = [(k, v) for k, v in tree.items() if not k.startswith("_")]
    items.sort(key=lambda x: (x[1].get("_type") != "dir", x[0]))

    for i, (name, value) in enumerate(items):
        is_last_item = i == len(items) - 1
        connector = "└── " if is_last_item else "├── "

        node_type = value.get("_type", "dir")
        symbols = value.get("_symbols", [])

        if node_type == "file":
            symbol_str = ""
            if symbols:
                short_symbols = [s.split(":")[-1] for s in symbols[:5]]
                if len(symbols) > 5:
                    short_symbols.append(f"+{len(symbols) - 5}")
                symbol_str = f" ({', '.join(short_symbols)})"
            lines.append(f"{prefix}{connector}{name}{symbol_str}")
        else:
            lines.append(f"{prefix}{connector}{name}/")
            new_prefix = prefix + ("    " if is_last_item else "│   ")
            subtree = {k: v for k, v in value.items() if not k.startswith("_")}
            if subtree:
                lines.append(format_tree_text(subtree, new_prefix, is_last_item))

    return "\n".join(lines)


async def get_repo_map_summary(project_id: Optional[int], max_lines: int = 100) -> str:
    if not ENABLE_REPO_MAP:
        return ""

    repo_data = await get_repo_map(project_id)
    if not repo_data.get("nodes"):
        return ""

    tree = build_tree(repo_data["nodes"])
    tree_text = format_tree_text(tree)

    lines = tree_text.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"... and {len(tree_text.split(chr(10))) - max_lines} more files")

    summary = "Repository structure:\n```\n" + "\n".join(lines) + "\n```"
    return summary


async def clear_repo_map(project_id: int):
    if not DATABASE_URL:
        return

    try:
        from gateway.db import get_session, RepoNode as RepoNodeModel
        from sqlalchemy import delete

        async with get_session() as session:
            await session.execute(
                delete(RepoNodeModel).where(RepoNodeModel.project_id == project_id)
            )

    except Exception as e:
        log.warning("Failed to clear repo map: %r", e)
