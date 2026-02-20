import json
import time
import traceback
import hashlib
import asyncio
from typing import Any, Dict, List

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.models import OAChatReq
from gateway.config import (
    MAX_BODY_BYTES,
    DEFAULT_MAX_TOKENS,
    CACHE_TTL_SECONDS,
    DEFAULT_MODEL,
    OPUS_MODEL,
)
from gateway.logging_setup import log
from gateway.routing import route_model_from_messages, with_model_prefix
from gateway.cache import cache_key, cache_get, cache_set
from gateway.anthropic_client import (
    client,
    call_anthropic_with_timeout,
    extract_usage,
    anthropic_to_openai_usage,
)

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

    # -------------------------
    # Tools / tool_choice
    # -------------------------
    oa_tools = oa_tools_from_body(parsed)
    aa_tools = anthropic_tools_from_openai(oa_tools)
    aa_tool_choice = anthropic_tool_choice_from_openai(parsed.get("tool_choice"))

    if aa_tool_choice is None and "function_call" in parsed:
        fc = parsed.get("function_call")
        if isinstance(fc, str):
            aa_tool_choice = anthropic_tool_choice_from_openai(fc)
        elif isinstance(fc, dict) and fc.get("name"):
            aa_tool_choice = {"type": "tool", "name": fc["name"]}

    # -------------------------
    # Convert OpenAI messages -> Anthropic
    # -------------------------
    system_parts: List[str] = []
    aa_messages: List[Dict[str, Any]] = []
    user_join: List[str] = []

    for m in body.messages:
        role = (m.role or "").lower()

        content_text = m.content
        if isinstance(content_text, str):
            content_text = content_text
        elif content_text is None:
            content_text = ""
        else:
            content_text = json.dumps(content_text, ensure_ascii=False)
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

        # ignore unknown roles safely

    # -------------------------
    # System prompt assembly + diff-first policy
    # -------------------------
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

    # -------------------------
    # Cache hit: text-only responses only
    # -------------------------
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
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": out_text}, "finish_reason": "stop"}
                ],
                "usage": anthropic_to_openai_usage(usage_cached),
            }
            response = JSONResponse(content=resp_json)
            response.headers["X-Gateway"] = "gursimanoor-gateway"
            response.headers["X-Model-Source"] = "custom"
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Reduction"] = "1"
            return response

    # -------------------------
    # STREAMING: tool-call compatible SSE
    # -------------------------
    if body.stream:
        if client is None:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

        async def sse_stream():
            chunk_id = f"chatcmpl_{hashlib.sha1((ray + str(time.time())).encode()).hexdigest()[:16]}"
            created = int(time.time())
            q: asyncio.Queue = asyncio.Queue()

            # IMPORTANT: define loop BEFORE worker uses it
            loop = asyncio.get_running_loop()

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

                                # TEXT
                                if btype == "text" and delta is not None:
                                    txt = getattr(delta, "text", None) if not isinstance(delta, dict) else delta.get("text")
                                    if txt:
                                        asyncio.run_coroutine_threadsafe(q.put(("text", txt)), loop)

                                # TOOL USE
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
                                            "function": {
                                                "name": tool_name,
                                                "arguments": ensure_json_args_str(tool_input),
                                            },
                                        }
                                        asyncio.run_coroutine_threadsafe(q.put(("tool_call", tc)), loop)

                            if etype in ("message_stop", "message_end"):
                                break

                        final = stream.get_final_message()
                        usage = extract_usage(final)
                        asyncio.run_coroutine_threadsafe(q.put(("done", usage)), loop)

                except Exception as e:
                    # Log full Anthropic error + payload summary
                    try:
                        from gateway.anthropic_client import log_anthropic_error
                        log_anthropic_error("ANTHROPIC stream failed", payload, e)
                    except Exception:
                        pass

                    asyncio.run_coroutine_threadsafe(q.put(("error", str(e))), loop)

            await asyncio.to_thread(_worker_stream)

            finished = False
            while not finished:
                kind, payload_item = await q.get()

                if kind == "error":
                    final = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    finished = True
                    continue

                if kind == "text":
                    event = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {"content": payload_item}}],
                    }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue

                if kind == "tool_call":
                    event = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {"tool_calls": [payload_item]}}],
                    }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue

                if kind == "done":
                    final = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    finished = True

        response = StreamingResponse(sse_stream(), media_type="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["X-Gateway"] = "gursimanoor-gateway"
        response.headers["X-Model-Source"] = "custom"
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Reduction"] = "1"
        dt_ms = int((time.time() - t0) * 1000)
        log.info("OA OK STREAM (cf-ray=%s) model=%s ms=%s tools=%s", ray, model, dt_ms, bool(aa_tools))
        return response

    # -------------------------
    # Non-stream upstream call
    # -------------------------
    try:
        resp = await call_anthropic_with_timeout(payload)
    except HTTPException:
        raise
    except Exception as e:
        log.error("OA UPSTREAM ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream model error")

    # -------------------------
    # Parse response: text + tool_use -> tool_calls
    # -------------------------
    out_text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    try:
        for block in getattr(resp, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                out_text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_use_id = getattr(block, "id", "") or ""
                name = getattr(block, "name", "") or ""
                tool_input = getattr(block, "input", {}) or {}
                if tool_use_id and name:
                    tool_calls.append({
                        "id": tool_use_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": ensure_json_args_str(tool_input),
                        },
                    })
    except Exception as e:
        log.error("OA PARSE ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream response parse error")

    out_text = "".join(out_text_parts)
    usage = extract_usage(resp)

    cache_blob = {
        "cached": False,
        "model": model,
        "text": out_text,
        "usage": usage,
        "tool_calls": (tool_calls or None),
    }
    if key and not tool_calls:
        cache_set(key, cache_blob, CACHE_TTL_SECONDS)

    finish_reason = "stop"
    message_obj: Dict[str, Any] = {"role": "assistant", "content": out_text}

    if tool_calls:
        message_obj["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    resp_json = {
        "id": f"chatcmpl_{hashlib.sha1((ray + str(time.time())).encode()).hexdigest()[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": with_model_prefix(model),
        "choices": [{"index": 0, "message": message_obj, "finish_reason": finish_reason}],
        "usage": anthropic_to_openai_usage(usage),
    }

    response = JSONResponse(content=resp_json)
    response.headers["X-Gateway"] = "gursimanoor-gateway"
    response.headers["X-Model-Source"] = "custom"
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Reduction"] = "1"
    return response


@router.post("/chat/completions")
async def chat_completions_alias(req: Request, body: OAChatReq):
    return await openai_chat_completions(req, body)


@router.get("/v1/models")
async def openai_models(req: Request):
    ray = req.headers.get("cf-ray") or ""
    ua = req.headers.get("user-agent") or ""
    log.info("MODELS HIT cf-ray=%s ua=%s", ray, ua[:120])

    payload = {
        "object": "list",
        "data": [
            {"id": with_model_prefix("sonnet"), "object": "model"},
            {"id": with_model_prefix("opus"), "object": "model"},
            {"id": with_model_prefix(DEFAULT_MODEL), "object": "model"},
            {"id": with_model_prefix(OPUS_MODEL), "object": "model"},
            {"id": "sonnet", "object": "model"},
            {"id": "opus", "object": "model"},
            {"id": DEFAULT_MODEL, "object": "model"},
            {"id": OPUS_MODEL, "object": "model"},
        ],
    }
    resp = JSONResponse(content=payload)
    resp.headers["X-Gateway"] = "gursimanoor-gateway"
    resp.headers["X-Model-Source"] = "custom"
    return resp


@router.get("/models")
async def models_alias(req: Request):
    return await openai_models(req)
