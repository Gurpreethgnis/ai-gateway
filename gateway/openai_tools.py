import json
from typing import Any, Dict, List, Optional, Union

def oa_tools_from_body(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = []
    if isinstance(parsed.get("tools"), list):
        tools = parsed.get("tools") or []
    elif isinstance(parsed.get("functions"), list):
        for fn in parsed.get("functions") or []:
            if isinstance(fn, dict) and fn.get("name"):
                tools.append({"type": "function", "function": fn})
    return [t for t in tools if isinstance(t, dict)]

def anthropic_tools_from_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        desc = fn.get("description") or ""
        params = fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        out.append({"name": name, "description": desc, "input_schema": params})
    return out

def anthropic_tool_choice_from_openai(tool_choice: Any) -> Optional[Dict[str, Any]]:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        low = tool_choice.lower().strip()
        if low in ("auto",):
            return {"type": "auto"}
        if low in ("none", "no", "disabled"):
            return {"type": "none"}
        return {"type": "auto"}
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn = (tool_choice.get("function") or {})
            name = fn.get("name")
            if name:
                return {"type": "tool", "name": name}
        return {"type": "auto"}
    return None

def ensure_json_args_str(args: Any) -> str:
    if args is None:
        return "{}"
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False)
    except Exception:
        return "{}"

def parse_json_maybe(s: Any) -> Any:
    if s is None:
        return {}
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        ss = s.strip()
        if not ss:
            return {}
        try:
            return json.loads(ss)
        except Exception:
            return {"_raw": s}
    return {"_raw": s}

def anthropic_tool_result_block(tool_call_id: str, tool_text: str) -> List[Dict[str, Any]]:
    return [{
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": [{"type": "text", "text": tool_text}],
    }]

# ---- Pydantic extras safe getters ----
def get_extra(m: Any, key: str, default=None):
    if hasattr(m, key):
        v = getattr(m, key)
        if v is not None:
            return v
    extra = getattr(m, "__pydantic_extra__", None)
    if isinstance(extra, dict) and key in extra:
        return extra.get(key, default)
    extra2 = getattr(m, "model_extra", None)
    if isinstance(extra2, dict) and key in extra2:
        return extra2.get(key, default)
    d = getattr(m, "__dict__", None)
    if isinstance(d, dict) and key in d:
        return d.get(key, default)
    return default

def oai_tool_calls_from_assistant_msg(m: Any) -> List[Dict[str, Any]]:
    tc = get_extra(m, "tool_calls", None)
    if not isinstance(tc, list):
        return []
    out = []
    for item in tc:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function":
            continue
        fn = item.get("function") or {}
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        out.append(item)
    return out

def assistant_blocks_from_oai(content_text: str, tool_calls: List[Dict[str, Any]]) -> Union[str, List[Dict[str, Any]]]:
    blocks: List[Dict[str, Any]] = []
    if content_text:
        blocks.append({"type": "text", "text": content_text})
    for tc in tool_calls:
        tc_id = tc.get("id") or "call_missing_id"
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args = parse_json_maybe(fn.get("arguments"))
        if name:
            blocks.append({"type": "tool_use", "id": tc_id, "name": name, "input": args})
    if blocks and all(b.get("type") == "text" for b in blocks):
        return content_text
    return blocks if blocks else content_text
