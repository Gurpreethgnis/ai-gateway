import json
import time
import traceback
import hashlib
import asyncio
from typing import Any, Dict, List

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.models import OAChatReq
from gateway.config import MAX_BODY_BYTES, DEFAULT_MAX_TOKENS, CACHE_TTL_SECONDS, DEFAULT_MODEL, OPUS_MODEL
from gateway.logging_setup import log
from gateway.routing import route_model_from_messages, with_model_prefix
from gateway.cache import cache_key, cache_get, cache_set
from gateway.anthropic_client import client, call_anthropic_with_timeout, extract_usage, anthropic_to_openai_usage

from gateway.openai_tools import (
    oa_tools_from_body,
    anthropic_tools_from_openai,
    anthropic_tool_choice_from_openai,
    ensure_json_args_str,
    anthropic_tool_result_block,
    get_extra,
    oai_tool_calls_from_assistant_msg,
    assistant_blocks_from_oai,
)
from gateway.token_reduction import (
    strip_or_truncate,
    maybe_prefix_cache,
    tool_result_dedup,
    enforce_diff_first,
    LIMITS,
)

router = APIRouter()

@router.post("/v1/chat/completions")
async def openai_chat_completions(req: Request, body: OAChatReq):
    raw = await req.body()

    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
        log.info("OA INCOMING KEYS: %s", list(parsed.keys()))
        log.info("OA HAS tools: %s", "tools" in parsed)
        log.info("OA HAS functions: %s", "functions" in parsed)
        log.info("OA HAS tool_choice: %s", "tool_choice" in parsed)
        log.info("OA HAS function_call: %s", "function_call" in parsed)
    except Exception:
        parsed = {}

    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")

    ray = req.headers.get("cf-ray") or ""
    t0 = time.time()

    oa_tools = oa_tools_from_body(parsed)
    aa_tools = anthropic_tools_from_openai(oa_tools)
    aa_tool_choice = anthropic_tool_choice_from_openai(parsed.get("tool_choice"))

    if aa_tool_choice is None and "function_call" in parsed:
        fc = parsed.get("function_call")
        if isinstance(fc, str):
            aa_tool_choice = anthropic_tool_choice_from_openai(fc)
        elif isinstance(fc, dict) and fc.get("name"):
            aa_tool_choice = {"type": "tool", "name": fc["name"]}

    system_parts: List[str] = []
    aa_messages: List[Dict[str, Any]] = []
    user_join: List[str] = []

    for m in body.messages:
        role = (m.role or "").lower()
        content_text = m.content
        content_text = content_text if isinstance(content_text, str) else (json.dumps(content_text, ensure_ascii=False) if content_text is not None else "")
        content_text = content_text or ""

        if role in ("system", "developer"):
            new_text, _meta = strip_or_truncate(role, content_text, LIMITS["system_max"], allow_strip=True)
            if new_text.strip():
                system_parts.append(new_text.strip())
            continue

        if role == "tool":
            tool_call_id = get_extra(m, "tool_call_id", "") or ""
            tool_text, _tmeta = strip_or_truncate("tool", content_text, LIMITS["tool_result_max"], allow_strip=False)
            tool_text, _dmeta = tool_result_dedup(tool_call_id, tool_text)

            if tool_call_id:
                aa_messages.append({"role": "user", "content": anthropic_tool_result_block(tool_call_id, tool_text)})
            else:
                aa_messages.append({"role": "user", "content": tool_text})
            continue

        if role == "user":
            new_text, _meta = strip_or_truncate("user", content_text, LIMITS["user_msg_max"], allow_strip=False)
            if new_text:
                user_join.append(new_text)
            aa_messages.append({"role": "user", "content": new_text})
            continue

        if role == "assistant":
            # IMPORTANT: translate assistant tool_calls history into Anthropic tool_use blocks
            tcs = oai_tool_calls_from_assistant_msg(m)
            aa_content = assistant_blocks_from_oai(content_text, tcs)
            aa_messages.append({"role": "assistant", "content": aa_content})
            continue

    system_text = "\n\n".join([p for p in system_parts if p]).strip()
    system_text = enforce_diff_first(system_text)
    system_text, _ = strip_or_truncate("system", system_text, LIMITS["system_max"], allow_strip=False)
    system_text, _ = maybe_prefix_cache(system_text)

    joined_user = "\n".join(user_join)
    model = route_model_from_messages(joined_user, body.model)
    max_tokens = int(body.max_tokens or DEFAULT_MAX_TOKENS)
    temperature = body.temperature if body.temperature is not None else 0.2

    payload: Dict[str, Any] = {
        "model": model,
        "system": system_text,
        "messages": aa_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if aa_tools:
        payload["tools"] = aa_tools
        if aa_tool_choice:
            payload["tool_choice"] = aa_tool_choice

    do_cache = temperature <= 0.3
    key = cache_key(payload) if do_cache else None

    # cache hit only for text-only non-tool responses (same as your current logic)
    if key:
        cached = cache_get(key)
        if cached and isinstance(cached, dict) and "text" in cached and cached.get("tool_calls") is None:
            out_text = cached.get("text", "")
            usage_cached = cached.get("usage")
            resp_json = {
                "id": f"chatcmpl_cached_{key[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": with_model_prefix(str(cached.get("model", model))),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": out_text}, "finish_reason": "stop"}],
                "usage": anthropic_to_openai_usage(usage_cached),
            }
            response = JSONResponse(content=resp_json)
            response.headers["X-Gateway"] = "gursimanoor-gateway"
            response.headers["X-Model-Source"] = "custom"
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Reduction"] = "1"
            return response

    # STREAMING: keep your existing implementation style (thread worker + queue)
    if body.stream:
        if client is None:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

        async def sse_stream():
            chunk_id = f"chatcmpl_{hashlib.sha1((ray + str(time.time())).encode()).hexdigest()[:16]}"
            created = int(time.time())
            q: asyncio.Queue = asyncio.Queue()

            def _worker_stream():
                try:
                    with client.messages.stream(**payload) as stream:
                        for ev in stream:
                            etype = getattr(ev, "type", None) or (ev.get("type") if isinstance(ev, dict) else None)
                            if etype in ("content_block_start", "content_block_delta", "content_block_stop"):
                                block = getattr(ev, "content_block", None) or (ev.get("content_block") if isinstance(ev, dict) else None)
                                delta = getattr(ev, "delta", None) or (ev.get("delta") if isinstance(ev, dict) else None)

                                btype = getattr(block, "type", None) if block is not None else None
                                if btype is None and isinstance(block, dict):
                                    btype = block.get("type")

                                if btype == "text" and delta is not None:
                                    txt = getattr(delta, "text", None) if not isinstance(delta, dict) else delta.get("text")
                                    if txt:
                                        asyncio.run_coroutine_threadsafe(q.put(("text", txt)), loop)

                                if btype == "tool_use":
                                    if isinstance(block, dict):
                                        tool_use_id = block.get("id") or ""
                                        tool_name = block.get("name") or ""
                                        tool_input = block.get("input") or {}
                                    else:
                                        tool_use_id = getattr(block, "id", "") or ""
                                        tool_name = getattr(block, "name", "") or ""
                                        tool_input = getattr(block, "input", {}) or {}

                                    if tool_use_id and tool_name:
                                        tc = {
                                            "id": tool_use_id,
                                            "type": "function",
                                            "function": {"name": tool_name, "arguments": ensure_json_args_str(tool_input)},
                                        }
                                        asyncio.run_coroutine_threadsafe(q.put(("tool_call", tc)), loop)

                            if etype in ("message_stop", "message_end"):
                                break

                        final = stream.get_final_message()
                        usage = extract_usage(final)
                        asyncio.run_coroutine_threadsafe(q.put(("done", usage)), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(q.put(("error", str(e))), loop)

            loop = asyncio.get_running_loop()
            await asyncio.to_thread(_worker_stream)

            finished = False
            while not finished:
                kind, payload_item = await q.get()

                if kind == "error":
                    final = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": with_model_prefix(model),
                             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    finished = True
                    continue

                if kind == "text":
                    event = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": with_model_prefix(model),
                             "choices": [{"index": 0, "delta": {"content": payload_item}}]}
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue

                if kind == "tool_call":
                    event = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": with_model_prefix(model),
                             "choices": [{"index": 0, "delta": {"tool_calls": [payload_item]}}]}
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue

                if kind == "done":
                    final = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": with_model_prefix(model),
                             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    finished = True

        response = StreamingResponse(sse_stream(), media_type="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        re
