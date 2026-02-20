import json
import time
import traceback
import hashlib
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

# Note: OAChatReq no longer used directly - we normalize request body ourselves
from gateway.config import (
    MAX_BODY_BYTES,
    DEFAULT_MAX_TOKENS,
    CACHE_TTL_SECONDS,
    DEFAULT_MODEL,
    OPUS_MODEL,
    ENABLE_ANTHROPIC_CACHE_CONTROL,
    ENABLE_SMART_ROUTING,
    ENABLE_CONTEXT_PRUNING,
    ENABLE_FILE_HASH_CACHE,
    ENABLE_MEMORY_LAYER,
    ENABLE_MULTI_PROJECT,
    ENABLE_PLUGIN_TOOLS,
    DATABASE_URL,
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
    enforce_diff_first,
    LIMITS,
)

from gateway.rate_limit import check_rate_limit, get_rate_limit_headers
from gateway.circuit_breaker import with_circuit_breaker, CircuitOpenError
from gateway.retry import with_retry, RetryExhaustedError
from gateway.metrics import (
    record_request,
    record_stream_duration,
    record_upstream_error,
    increment_active_requests,
    decrement_active_requests,
)
from gateway.telemetry import emit_error, emit_request
from gateway.db import calculate_cost

router = APIRouter()


async def get_project_context(request: Request) -> tuple[Optional[int], Optional[str], int]:
    if not ENABLE_MULTI_PROJECT or not DATABASE_URL:
        return None, None, 60

    auth = request.headers.get("authorization") or ""
    api_key = None
    if auth.lower().startswith("bearer "):
        api_key = auth.split(" ", 1)[1].strip()

    if not api_key:
        return None, None, 60

    try:
        from gateway.projects import get_project_by_api_key
        project = await get_project_by_api_key(api_key)
        if project:
            return project.id, project.name, project.config.rate_limit_rpm
    except Exception as e:
        log.warning("Project lookup failed: %r", e)

    return None, None, 60


async def record_usage_to_db(
    project_id: Optional[int],
    model: str,
    input_tokens: int,
    output_tokens: int,
    cf_ray: str,
    cached: bool,
):
    if not DATABASE_URL or not project_id:
        return

    try:
        from gateway.db import get_session, UsageRecord, calculate_cost as calc_cost

        cost = calc_cost(model, input_tokens, output_tokens)

        async with get_session() as session:
            record = UsageRecord(
                project_id=project_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                cached=cached,
                cf_ray=cf_ray,
            )
            session.add(record)

    except Exception as e:
        log.warning("Failed to record usage: %r", e)


def normalize_request_body(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Convert various OpenAI API formats to Chat Completions format."""
    if "messages" in parsed:
        return parsed

    if "input" in parsed:
        input_data = parsed["input"]
        messages = []

        if isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, dict) and "role" in item:
                    messages.append(item)
                elif isinstance(item, str):
                    messages.append({"role": "user", "content": item})
        elif isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})

        result = {**parsed, "messages": messages}
        del result["input"]
        return result

    return parsed


@router.post("/v1/chat/completions")
async def openai_chat_completions(req: Request):
    raw = await req.body()

    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}

    parsed = normalize_request_body(parsed)

    if "messages" not in parsed or not parsed["messages"]:
        raise HTTPException(
            status_code=422,
            detail="Request must include 'messages' or 'input' field with content"
        )

    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")

    ray = req.headers.get("cf-ray") or ""
    t0 = time.time()

    project_id, project_name, rate_limit_rpm = await get_project_context(req)

    rate_result = await check_rate_limit(
        project_id=str(project_id) if project_id else "default",
        limit_override=rate_limit_rpm,
    )
    if not rate_result.allowed:
        response = JSONResponse(
            status_code=429,
            content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
        )
        for k, v in get_rate_limit_headers(rate_result).items():
            response.headers[k] = v
        return response

    oa_tools = oa_tools_from_body(parsed)
    aa_tools = anthropic_tools_from_openai(oa_tools)
    aa_tool_choice = anthropic_tool_choice_from_openai(parsed.get("tool_choice"))

    if aa_tool_choice is None and "function_call" in parsed:
        fc = parsed.get("function_call")
        if isinstance(fc, str):
            aa_tool_choice = anthropic_tool_choice_from_openai(fc)
        elif isinstance(fc, dict) and fc.get("name"):
            aa_tool_choice = {"type": "tool", "name": fc["name"]}

    if ENABLE_PLUGIN_TOOLS and project_id:
        try:
            from gateway.plugins import get_project_plugin_tools, build_tools_with_plugins
            plugins = await get_project_plugin_tools(project_id)
            if plugins:
                oa_tools = build_tools_with_plugins(oa_tools, plugins)
                aa_tools = anthropic_tools_from_openai(oa_tools)
        except Exception as e:
            log.warning("Plugin tools load failed: %r", e)

    system_parts: List[str] = []
    aa_messages: List[Dict[str, Any]] = []
    user_join: List[str] = []

    messages_list = parsed.get("messages", [])
    for m in messages_list:
        if isinstance(m, dict):
            role = (m.get("role") or "").lower()
            raw_content = m.get("content")
        else:
            role = getattr(m, "role", "")
            raw_content = getattr(m, "content", None)

        content_text = raw_content
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
            tool_call_id = m.get("tool_call_id", "") if isinstance(m, dict) else get_extra(m, "tool_call_id", "")
            tool_call_id = tool_call_id or ""
            tool_text, _tmeta = strip_or_truncate("tool", content_text, LIMITS["tool_result_max"], allow_strip=False)

            if ENABLE_FILE_HASH_CACHE:
                try:
                    from gateway.file_cache import process_tool_result
                    tool_text = await process_tool_result(project_id, tool_text, tool_name=None)
                except Exception as e:
                    log.warning("File cache processing failed: %r", e)

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
            tcs = oai_tool_calls_from_assistant_msg(m)
            aa_content = assistant_blocks_from_oai(content_text, tcs)
            aa_messages.append({"role": "assistant", "content": aa_content})
            continue

    system_text = "\n\n".join([p for p in system_parts if p]).strip()
    system_text = enforce_diff_first(system_text)
    system_text, _ = strip_or_truncate("system", system_text, LIMITS["system_max"], allow_strip=False)

    if ENABLE_MEMORY_LAYER and project_id:
        try:
            from gateway.memory import build_memory_context
            joined_query = "\n".join(user_join[-3:])
            memory_context = await build_memory_context(project_id, joined_query, max_chars=1500)
            if memory_context:
                system_text = memory_context + "\n\n" + system_text
        except Exception as e:
            log.warning("Memory context build failed: %r", e)

    joined_user = "\n".join(user_join)

    request_model = parsed.get("model")
    if ENABLE_SMART_ROUTING:
        try:
            from gateway.smart_routing import route_model
            model = await route_model(aa_messages, aa_tools, project_id, request_model)
        except Exception as e:
            log.warning("Smart routing failed: %r", e)
            model = route_model_from_messages(joined_user, request_model)
    else:
        model = route_model_from_messages(joined_user, request_model)

    max_tokens = int(parsed.get("max_tokens") or DEFAULT_MAX_TOKENS)
    temperature = parsed.get("temperature") if parsed.get("temperature") is not None else 0.2

    if ENABLE_CONTEXT_PRUNING:
        try:
            from gateway.context_pruner import prune_context
            aa_messages, prune_meta = await prune_context(aa_messages, system_text=system_text)
            if prune_meta.get("pruned"):
                log.debug("Context pruned: %s", prune_meta)
        except Exception as e:
            log.warning("Context pruning failed: %r", e)

    if ENABLE_ANTHROPIC_CACHE_CONTROL and system_text and len(system_text) >= 1024:
        system_param = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
    else:
        system_param = system_text

    payload: Dict[str, Any] = {
        "model": model,
        "system": system_param,
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

            dt_ms = int((time.time() - t0) * 1000)
            record_request(model, project_name or "default", 200, "/v1/chat/completions", dt_ms / 1000.0, cached=True)

            response = JSONResponse(content=resp_json)
            response.headers["X-Gateway"] = "gursimanoor-gateway"
            response.headers["X-Model-Source"] = "custom"
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Reduction"] = "1"
            return response

    is_stream = parsed.get("stream", False)
    if is_stream:
        if client is None:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

        include_usage = False
        stream_opts = parsed.get("stream_options")
        if isinstance(stream_opts, dict) and stream_opts.get("include_usage"):
            include_usage = True

        async def sse_stream():
            stream_t0 = time.time()
            chunk_id = f"chatcmpl_{hashlib.sha1((ray + str(time.time())).encode()).hexdigest()[:16]}"
            created = int(time.time())
            q: asyncio.Queue = asyncio.Queue()

            loop = asyncio.get_running_loop()
            increment_active_requests(model)

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

                                if etype == "content_block_delta" and delta is not None:
                                    delta_type = getattr(delta, "type", None) if not isinstance(delta, dict) else delta.get("type")
                                    if delta_type == "text_delta":
                                        txt = getattr(delta, "text", None) if not isinstance(delta, dict) else delta.get("text")
                                        if txt:
                                            asyncio.run_coroutine_threadsafe(q.put(("text", txt)), loop)
                                    elif delta_type == "input_json_delta":
                                        partial_json = getattr(delta, "partial_json", None) if not isinstance(delta, dict) else delta.get("partial_json")
                                        if partial_json:
                                            asyncio.run_coroutine_threadsafe(q.put(("tool_args_delta", partial_json)), loop)

                                if etype == "content_block_start" and btype == "tool_use":
                                    if isinstance(block, dict):
                                        tool_use_id = block.get("id") or ""
                                        tool_name = block.get("name") or ""
                                    else:
                                        tool_use_id = getattr(block, "id", "") or ""
                                        tool_name = getattr(block, "name", "") or ""

                                    if tool_use_id and tool_name:
                                        tc = {
                                            "id": tool_use_id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": "",
                                            },
                                        }
                                        asyncio.run_coroutine_threadsafe(q.put(("tool_call_start", tc)), loop)

                            if etype in ("message_stop", "message_end"):
                                break

                        final = stream.get_final_message()
                        usage = extract_usage(final)
                        asyncio.run_coroutine_threadsafe(q.put(("done", usage)), loop)

                except Exception as e:
                    try:
                        from gateway.anthropic_client import log_anthropic_error
                        log_anthropic_error("ANTHROPIC stream failed", payload, e)
                    except Exception:
                        pass

                    asyncio.run_coroutine_threadsafe(q.put(("error", str(e))), loop)

            loop.run_in_executor(None, _worker_stream)

            finished = False
            current_tool_call_idx = -1
            final_usage = None

            while not finished:
                kind, payload_item = await q.get()

                if kind == "error":
                    record_upstream_error(model, "stream_error", 500)
                    final_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
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

                if kind == "tool_call_start":
                    current_tool_call_idx += 1
                    tc_with_index = {**payload_item, "index": current_tool_call_idx}
                    event = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {"tool_calls": [tc_with_index]}}],
                    }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue

                if kind == "tool_args_delta":
                    if current_tool_call_idx >= 0:
                        event = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": with_model_prefix(model),
                            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": current_tool_call_idx, "function": {"arguments": payload_item}}]}}],
                        }
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue

                if kind == "done":
                    final_usage = payload_item
                    final_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    if include_usage and payload_item:
                        final_chunk["usage"] = anthropic_to_openai_usage(payload_item)
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

                    dt_ms = int((time.time() - stream_t0) * 1000)
                    log.info("OA STREAM DONE (cf-ray=%s) model=%s ms=%s tools=%s", ray, model, dt_ms, bool(aa_tools))

                    record_stream_duration(model, project_name or "default", dt_ms / 1000.0)
                    record_request(model, project_name or "default", 200, "/v1/chat/completions", dt_ms / 1000.0,
                                   input_tokens=final_usage.get("input_tokens", 0) if final_usage else 0,
                                   output_tokens=final_usage.get("output_tokens", 0) if final_usage else 0,
                                   cost_usd=calculate_cost(model,
                                                           final_usage.get("input_tokens", 0) if final_usage else 0,
                                                           final_usage.get("output_tokens", 0) if final_usage else 0))

                    if final_usage:
                        await record_usage_to_db(
                            project_id, model,
                            final_usage.get("input_tokens", 0),
                            final_usage.get("output_tokens", 0),
                            ray, False
                        )

                    finished = True

            decrement_active_requests(model)

        log.debug("OA INCOMING KEYS: %s tools=%s functions=%s tool_choice=%s function_call=%s",
                  list(parsed.keys()), "tools" in parsed, "functions" in parsed,
                  "tool_choice" in parsed, "function_call" in parsed)

        response = StreamingResponse(sse_stream(), media_type="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["X-Gateway"] = "gursimanoor-gateway"
        response.headers["X-Model-Source"] = "custom"
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Reduction"] = "1"
        return response

    increment_active_requests(model)

    try:
        async def _call_upstream():
            return await call_anthropic_with_timeout(payload)

        try:
            resp = await with_circuit_breaker(
                lambda: with_retry(_call_upstream)
            )
        except CircuitOpenError as e:
            await emit_error(
                "circuit_open", str(e), cf_ray=ray, model=model,
                project_id=str(project_id) if project_id else None,
            )
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable",
                headers={"Retry-After": str(e.retry_after)},
            )
        except RetryExhaustedError as e:
            record_upstream_error(model, "retry_exhausted", 502)
            await emit_error(
                "retry_exhausted", str(e), cf_ray=ray, model=model,
                project_id=str(project_id) if project_id else None,
                exception=e.last_error,
            )
            raise HTTPException(status_code=502, detail="Upstream model error after retries")

    except HTTPException:
        decrement_active_requests(model)
        raise
    except Exception as e:
        decrement_active_requests(model)
        log.error("OA UPSTREAM ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        record_upstream_error(model, type(e).__name__, 502)
        await emit_error(
            "upstream_error", str(e), cf_ray=ray, model=model,
            project_id=str(project_id) if project_id else None,
            exception=e,
        )
        raise HTTPException(status_code=502, detail="Upstream model error")

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
        decrement_active_requests(model)
        log.error("OA PARSE ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream response parse error")

    decrement_active_requests(model)

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

    dt_ms = int((time.time() - t0) * 1000)
    input_tokens = usage.get("input_tokens", 0) if usage else 0
    output_tokens = usage.get("output_tokens", 0) if usage else 0
    cost = calculate_cost(model, input_tokens, output_tokens)

    record_request(model, project_name or "default", 200, "/v1/chat/completions", dt_ms / 1000.0,
                   input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

    await emit_request(
        cf_ray=ray,
        project_id=str(project_id) if project_id else None,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_ms=dt_ms,
        cached=False,
        status_code=200,
        has_tools=bool(aa_tools),
        stream=False,
    )

    await record_usage_to_db(project_id, model, input_tokens, output_tokens, ray, False)

    if ENABLE_MEMORY_LAYER and project_id and out_text and len(out_text) > 200:
        try:
            from gateway.memory import store_memory
            asyncio.create_task(store_memory(
                project_id, out_text[:3000],
                {"role": "assistant", "model": model, "cf_ray": ray}
            ))
        except Exception:
            pass

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
async def chat_completions_alias(req: Request):
    return await openai_chat_completions(req)


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
