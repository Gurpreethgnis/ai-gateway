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
    RETRY_ENABLED,
    RETRY_MAX_ATTEMPTS,
    LOCAL_LLM_DEFAULT_MODEL,
    LOCAL_LLM_MODEL_ALLOWLIST,
)
from gateway.logging_setup import log
from gateway.routing import route_model_from_messages, with_model_prefix, strip_model_prefix, VALID_ANTHROPIC_MODELS, get_fallback_model
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
from gateway.concurrency import acquire_concurrency_slot, release_concurrency_slot
from gateway.retry import with_retry, RetryExhaustedError
from gateway.metrics import (
    record_request,
    record_stream_duration,
    record_upstream_error,
    increment_active_requests,
    decrement_active_requests,
)
from gateway.telemetry import emit_error, emit_request
from gateway.db import calculate_cost, record_usage_to_db

router = APIRouter()


def is_local_provider_request(parsed: Dict[str, Any]) -> bool:
    """Check if request should be routed to local Ollama provider."""
    provider = parsed.get("provider")
    if provider == "local":
        return True
    
    model = parsed.get("model") or ""
    if model.startswith("local:") or model.startswith("ollama:"):
        return True
    
    return False


def extract_local_model(parsed: Dict[str, Any]) -> Optional[str]:
    """Extract model name for local provider, stripping any prefix."""
    model = parsed.get("model") or ""
    
    if model.startswith("local:"):
        return model[6:] or None
    if model.startswith("ollama:"):
        return model[7:] or None
    
    return model if model else None


async def handle_local_provider(
    parsed: Dict[str, Any],
    ray: str,
    t0: float,
) -> JSONResponse:
    """Handle request via local Ollama provider and return OpenAI-formatted response."""
    from gateway.providers.ollama import call_ollama_openai_format, validate_local_model
    import hashlib
    
    messages = parsed.get("messages", [])
    local_model = extract_local_model(parsed)
    resolved_model = validate_local_model(local_model)
    
    oa_messages = []
    for m in messages:
        role = m.get("role", "user") if isinstance(m, dict) else getattr(m, "role", "user")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        
        oa_messages.append({"role": role, "content": content or ""})
    
    max_tokens = int(parsed.get("max_tokens") or DEFAULT_MAX_TOKENS)
    temperature = parsed.get("temperature") if parsed.get("temperature") is not None else 0.2
    
    request_id = f"chatcmpl_local_{hashlib.sha1((ray + str(time.time())).encode()).hexdigest()[:12]}"
    
    result = await call_ollama_openai_format(
        messages=oa_messages,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        request_id=request_id,
    )
    
    dt_ms = int((time.time() - t0) * 1000)
    log.info(
        "OA LOCAL OK (cf-ray=%s) model=%s ms=%d",
        ray, resolved_model, dt_ms
    )
    
    response = JSONResponse(content=result)
    response.headers["X-Gateway"] = "claude-gateway"
    response.headers["X-Model-Source"] = "local"
    response.headers["X-Provider"] = "ollama"
    return response


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

    if is_local_provider_request(parsed):
        return await handle_local_provider(parsed, ray, t0)

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
    last_assistant_tool_use_ids: List[str] = []
    tool_result_index_since_assistant: int = 0
    pending_tool_result_blocks: List[Dict[str, Any]] = []
    gateway_tokens_saved = 0

    def track_reduction(meta: Dict[str, Any]):
        nonlocal gateway_tokens_saved
        if meta.get("stripped") or meta.get("truncated"):
            chars_saved = meta.get("before", 0) - meta.get("after", 0)
            tokens_saved = chars_saved // 4
            gateway_tokens_saved += tokens_saved
            
            # Record mechanism-specific savings
            mechanism = "boilerplate_strip" if meta.get("stripped") else "truncation"
            from gateway.metrics import record_tokens_saved
            record_tokens_saved(mechanism, project_name or "default", tokens_saved)
            
            log.debug("Gateway reduction: %d chars -> %d tokens (total: %d)",
                     chars_saved, tokens_saved, gateway_tokens_saved)

    def flush_tool_results():
        nonlocal tool_result_index_since_assistant, pending_tool_result_blocks
        if pending_tool_result_blocks:
            aa_messages.append({"role": "user", "content": list(pending_tool_result_blocks)})
            pending_tool_result_blocks = []
        tool_result_index_since_assistant = 0

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
        elif isinstance(content_text, list):
            # Extract text from OpenAI/Anthropic content blocks
            parts = []
            for b in content_text:
                if isinstance(b, str):
                    parts.append(b)
                elif isinstance(b, dict):
                    parts.append(b.get("text") or "")
            content_text = "\n".join([p for p in parts if p])
        elif content_text is None:
            content_text = ""
        else:
            content_text = str(content_text)
        content_text = content_text or ""

        if role in ("system", "developer"):
            new_text, meta = strip_or_truncate(role, content_text, LIMITS["system_max"], allow_strip=True)
            track_reduction(meta)
            if new_text.strip():
                system_parts.append(new_text.strip())
            continue

        if role == "tool":
            tool_call_id = m.get("tool_call_id", "") if isinstance(m, dict) else get_extra(m, "tool_call_id", "")
            tool_call_id = tool_call_id or ""
            tool_text, tmeta = strip_or_truncate("tool", content_text, LIMITS["tool_result_max"], allow_strip=False)
            track_reduction(tmeta)

            if ENABLE_FILE_HASH_CACHE:
                try:
                    from gateway.file_cache import process_tool_result
                    tool_text = await process_tool_result(project_id, tool_text, tool_name=None)
                except Exception as e:
                    log.warning("File cache processing failed: %r", e)

            # If client sent an Anthropic ID (toolu_), use it directly
            # Otherwise use index-based mapping
            use_id = tool_call_id
            if tool_call_id and tool_call_id.startswith("toolu_"):
                use_id = tool_call_id
            elif last_assistant_tool_use_ids and tool_result_index_since_assistant < len(last_assistant_tool_use_ids):
                use_id = last_assistant_tool_use_ids[tool_result_index_since_assistant]
            elif tool_call_id:
                use_id = tool_call_id
            tool_result_index_since_assistant += 1

            if use_id:
                block = anthropic_tool_result_block(use_id, tool_text)
                pending_tool_result_blocks.extend(block if isinstance(block, list) else [block])
            else:
                if pending_tool_result_blocks:
                    aa_messages.append({"role": "user", "content": pending_tool_result_blocks})
                    pending_tool_result_blocks = []
                aa_messages.append({"role": "user", "content": tool_text})
            continue

        if role == "user":
            # User message may be blocks containing tool results (e.g. tool_result_for / tool_call_id)
            raw_content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if isinstance(raw_content, list) and raw_content and all(isinstance(b, dict) for b in raw_content):
                has_tool_result_key = any(
                    (b.get("tool_result_for") or b.get("tool_call_id")) for b in raw_content
                )
                if has_tool_result_key:
                    blocks_as_tool_results: List[Dict[str, Any]] = []
                    idx = 0
                    for b in raw_content:
                        tid = b.get("tool_result_for") or b.get("tool_call_id") or ""
                        part = b.get("content") or b.get("text") or ""
                        if isinstance(part, list):
                            part = " ".join(
                                (x.get("text") or str(x)) for x in part if isinstance(x, dict)
                            )
                        elif not isinstance(part, str):
                            part = str(part) if part else ""
                        
                        # If client already sent an Anthropic ID (toolu_), use it directly
                        # Otherwise use index-based mapping
                        use_id = tid
                        if tid and tid.startswith("toolu_"):
                            use_id = tid
                        elif last_assistant_tool_use_ids and idx < len(last_assistant_tool_use_ids):
                            use_id = last_assistant_tool_use_ids[idx]
                        elif tid:
                            use_id = tid
                        idx += 1
                        
                        if use_id:
                            tool_text, tmeta = strip_or_truncate("tool", part, LIMITS["tool_result_max"], allow_strip=False)
                            track_reduction(tmeta)
                            blocks_as_tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": use_id,
                                "content": [{"type": "text", "text": tool_text}],
                            })
                    if blocks_as_tool_results:
                        flush_tool_results()
                        aa_messages.append({"role": "user", "content": blocks_as_tool_results})
                        tool_result_index_since_assistant = 0
                        continue
                else:
                    is_mixed = any(b.get("type") in ("text", "image_url") for b in raw_content)
                    if is_mixed:
                        flush_tool_results()
                        anthropic_content = []
                        text_parts = []
                        for b in raw_content:
                            if b.get("type") == "text":
                                t = b.get("text") or ""
                                anthropic_content.append({"type": "text", "text": t})
                                text_parts.append(t)
                            elif b.get("type") == "image_url":
                                iu_obj = b.get("image_url") or {}
                                url = iu_obj.get("url") or ""
                                if url.startswith("data:"):
                                    try:
                                        header, b64 = url.split(",", 1)
                                        media_type = header.replace("data:", "").split(";")[0]
                                        anthropic_content.append({
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": b64
                                            }
                                        })
                                        text_parts.append("[Image attached]")
                                    except Exception:
                                        pass
                                else:
                                    text_parts.append(f"[Image URL: {url}]")
                        if anthropic_content:
                            user_join.append("\n".join(text_parts))
                            aa_messages.append({"role": "user", "content": anthropic_content})
                            continue
            flush_tool_results()
            new_text, meta = strip_or_truncate("user", content_text, LIMITS["user_msg_max"], allow_strip=True)
            track_reduction(meta)
            if new_text:
                user_join.append(new_text)
                aa_messages.append({"role": "user", "content": new_text})
            continue

        if role == "assistant":
            flush_tool_results()
            tcs = oai_tool_calls_from_assistant_msg(m)
            aa_content = assistant_blocks_from_oai(content_text, tcs)
            aa_messages.append({"role": "assistant", "content": aa_content})
            if isinstance(aa_content, list):
                last_assistant_tool_use_ids = [b.get("id") or "" for b in aa_content if isinstance(b, dict) and b.get("type") == "tool_use"]
            else:
                last_assistant_tool_use_ids = []
            continue

    flush_tool_results()

    # Normalize messages: filter empty, ensure first is user, merge consecutive same-role
    def normalize_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Filter out empty content
        filtered = []
        for msg in msgs:
            c = msg.get("content")
            if c is None:
                continue
            if isinstance(c, str) and not c.strip():
                continue
            if isinstance(c, list) and len(c) == 0:
                continue
            filtered.append(msg)
        
        if not filtered:
            return []
        
        # Ensure first message is user
        if filtered[0].get("role") != "user":
            filtered.insert(0, {"role": "user", "content": "(conversation continues)"})
        
        # Merge consecutive same-role messages
        merged = []
        for msg in filtered:
            if merged and merged[-1].get("role") == msg.get("role"):
                prev_content = merged[-1].get("content")
                curr_content = msg.get("content")
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    merged[-1]["content"] = prev_content + "\n\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    merged[-1]["content"] = prev_content + curr_content
                elif isinstance(prev_content, str) and isinstance(curr_content, list):
                    merged[-1]["content"] = [{"type": "text", "text": prev_content}] + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, str):
                    merged[-1]["content"] = prev_content + [{"type": "text", "text": curr_content}]
                else:
                    merged.append(msg)
            else:
                merged.append(msg)
        
        return merged

    aa_messages = normalize_messages(aa_messages)

    system_text = "\n\n".join([p for p in system_parts if p]).strip()
    system_text = enforce_diff_first(system_text)
    system_text, meta = strip_or_truncate("system", system_text, LIMITS["system_max"], allow_strip=False)
    track_reduction(meta)

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

    # Per-request smart routing: header, body, or model "auto" (no redeploy needed)
    use_smart_routing = ENABLE_SMART_ROUTING
    h = req.headers.get("x-gateway-smart-routing")
    if h is not None:
        use_smart_routing = h.strip().lower() in ("1", "true", "yes")
    if "smart_routing" in parsed:
        use_smart_routing = bool(parsed.get("smart_routing"))
    is_auto_model = request_model and (strip_model_prefix(request_model) or "").strip().lower() in ("auto", "smartroute")
    if is_auto_model:
        use_smart_routing = True
        request_model = None  # let gateway pick from content

    if use_smart_routing:
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
                gateway_tokens_saved += prune_meta.get("tokens_saved", 0)
                log.debug("Context pruned: %s", prune_meta)
        except Exception as e:
            log.warning("Context pruning failed: %r", e)

    # Anthropic requires every tool_result to follow an assistant message with tool_use.
    # Pruning can leave the first message as user+tool_result only; strip such leading messages.
    def _is_only_tool_result_blocks(content: Any) -> bool:
        if not isinstance(content, list) or len(content) == 0:
            return False
        for block in content:
            if not isinstance(block, dict):
                return False
            if block.get("type") == "text" and (block.get("text") or "").strip():
                return False
            is_tool_result = (
                block.get("type") == "tool_result"
                or "tool_use_id" in block
                or "tool_result_for" in block
            )
            if not is_tool_result:
                return False
        return True

    def drop_leading_tool_result_only_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        msgs = list(msgs)
        while msgs and msgs[0].get("role") == "user" and _is_only_tool_result_blocks(msgs[0].get("content")):
            msgs.pop(0)
        # After stripping, first message must still be user (Anthropic expects user first)
        if msgs and msgs[0].get("role") != "user":
            msgs.insert(0, {"role": "user", "content": "(conversation continues)"})
        return msgs

    aa_messages = drop_leading_tool_result_only_messages(aa_messages)

    # When caching is enabled and system prompt is long (>= 1024 chars), prepend cacheable constitution
    # + diff rules for 60–80% prompt-cache savings, then append the client's system text.
    if ENABLE_ANTHROPIC_CACHE_CONTROL and system_text and len(system_text) >= 1024:
        from gateway.platform_constitution import get_cacheable_system_blocks
        system_blocks = get_cacheable_system_blocks(include_constitution=True, include_diff_rules=True)
        system_blocks.append({"type": "text", "text": system_text})
        system_param = system_blocks
    else:
        system_param = system_text

    if ENABLE_ANTHROPIC_CACHE_CONTROL and aa_messages:
        # Anthropic allows up to 4 checkpoints. 1-2 used for system blocks.
        # Use remaining checkpoints for recent user messages to capture growing context.
        user_msgs_indices = [i for i, m in enumerate(aa_messages) if m.get("role") == "user"]
        for idx in user_msgs_indices[-2:]:
            msg = aa_messages[idx]
            content = msg.get("content")
            if isinstance(content, str):
                if len(content) > 2048:  # Only cache substantial messages
                    msg["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
            elif isinstance(content, list) and content:
                # Add cache control to the last block of the message
                last_block = content[-1]
                if isinstance(last_block, dict) and last_block.get("type") == "text":
                    text_len = len(last_block.get("text", ""))
                    if text_len > 2048:  # Only cache substantial content
                        last_block["cache_control"] = {"type": "ephemeral"}

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

    do_cache = temperature <= 0.3 and not req.headers.get("X-No-Cache")
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
            response.headers["X-Gateway"] = "claude-gateway"
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
            nonlocal model
            stream_t0 = time.time()
            chunk_id = f"chatcmpl_{hashlib.sha1((ray + str(time.time())).encode()).hexdigest()[:16]}"
            created = int(time.time())
            q: asyncio.Queue = asyncio.Queue()

            loop = asyncio.get_running_loop()
            increment_active_requests(model)

            def _worker_stream():
                import time
                from gateway.retry import is_retryable_error
                from gateway.anthropic_client import log_anthropic_error
                from gateway.concurrency import acquire_concurrency_slot, release_concurrency_slot
                max_attempts = 3

                for attempt in range(max_attempts):
                    yielded_any = False
                    
                    req_id = acquire_concurrency_slot(payload["model"])
                    acquired_model = payload["model"]  # save before any failover changes it
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
                                            if isinstance(delta, dict):
                                                txt = delta.get("text")
                                            else:
                                                txt = getattr(delta, "text", None)
                                            if txt:
                                                yielded_any = True
                                                asyncio.run_coroutine_threadsafe(q.put(("text", txt)), loop)
                                        elif delta_type == "input_json_delta":
                                            partial_json = getattr(delta, "partial_json", None) if not isinstance(delta, dict) else delta.get("partial_json")
                                            if partial_json:
                                                yielded_any = True
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
                                            yielded_any = True
                                            asyncio.run_coroutine_threadsafe(q.put(("tool_call_start", tc)), loop)

                                if etype in ("message_stop", "message_end"):
                                    break

                            final = stream.get_final_message()
                            usage = extract_usage(final)
                            asyncio.run_coroutine_threadsafe(q.put(("done", usage)), loop)
                            return

                    except Exception as e:
                        if not yielded_any and attempt < max_attempts - 1 and is_retryable_error(e):
                            log.warning("STREAM Retry attempt %d/%d after error: %r", attempt + 1, max_attempts, e)
                            
                            # Auto-failover to Haiku if we hit concurrent connection rate limits (429)
                            status = getattr(e, "status_code", None)
                            if status is None:
                                resp = getattr(e, "response", None)
                                status = getattr(resp, "status_code", None) if resp else None
                            
                            if status == 429:
                                fallback_model = get_fallback_model(payload["model"], attempt + 1)
                                log.warning("STREAM RateLimit hit (429) for %s, failing over to %s for attempt %d", payload["model"], fallback_model, attempt + 1)
                                payload["model"] = fallback_model
                                # Signal model switch back to sse_stream so it can update its local variables/metrics
                                asyncio.run_coroutine_threadsafe(q.put(("model_switch", fallback_model)), loop)
                            
                            time.sleep(1.0 * (2 ** attempt))
                            continue

                        try:
                            log_anthropic_error("ANTHROPIC stream failed", payload, e)
                        except Exception:
                            pass

                        err_msg = str(e)
                        if getattr(e, "response", None) is not None:
                            try:
                                j = e.response.json()
                                err_msg = j.get("error", {}).get("message", err_msg)
                            except Exception:
                                pass
                        asyncio.run_coroutine_threadsafe(q.put(("error", err_msg)), loop)
                        break
                    
                    finally:
                        release_concurrency_slot(acquired_model, req_id)

            loop.run_in_executor(None, _worker_stream)

            finished = False
            current_tool_call_idx = -1
            final_usage = None

            while not finished:
                kind, payload_item = await q.get()

                if kind == "model_switch":
                    decrement_active_requests(model)
                    model = payload_item
                    increment_active_requests(model)
                    continue

                if kind == "error":
                    record_upstream_error(model, "stream_error", 500)
                    final_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {"content": f"\n\n[Upstream Error: {payload_item}]"}, "finish_reason": "stop"}],
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
                    finish_reason = "stop" if current_tool_call_idx < 0 else "tool_calls"
                    final_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": with_model_prefix(model),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
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
                        cache_read_tokens = final_usage.get("cache_read_input_tokens", 0)
                        cache_write_tokens = final_usage.get("cache_creation_input_tokens", 0)
                        
                        await record_usage_to_db(
                            project_id, model,
                            final_usage.get("input_tokens", 0),
                            final_usage.get("output_tokens", 0),
                            ray, False,
                            cache_read_tokens,
                            cache_write_tokens,
                            gateway_tokens_saved
                        )
                        
                        # Record Anthropic prompt cache metrics
                        if cache_read_tokens > 0:
                            from gateway.metrics import record_prompt_cache_tokens
                            record_prompt_cache_tokens("read", model, project_name or "default", cache_read_tokens)
                        if cache_write_tokens > 0:
                            from gateway.metrics import record_prompt_cache_tokens
                            record_prompt_cache_tokens("write", model, project_name or "default", cache_write_tokens)

                    finished = True

            decrement_active_requests(model)

        log.debug("OA INCOMING KEYS: %s tools=%s functions=%s tool_choice=%s function_call=%s",
                  list(parsed.keys()), "tools" in parsed, "functions" in parsed,
                  "tool_choice" in parsed, "function_call" in parsed)

        response = StreamingResponse(sse_stream(), media_type="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["X-Gateway"] = "claude-gateway"
        response.headers["X-Model-Source"] = "custom"
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Reduction"] = "1"
        return response

    increment_active_requests(model)

    max_attempts = RETRY_MAX_ATTEMPTS if RETRY_ENABLED else 1
    resp = None
    last_err = None

    for attempt in range(max_attempts):
        req_id = acquire_concurrency_slot(model)
        acquired_model = model
        try:
            # Note: with_circuit_breaker uses a global breaker by default.
            # We pass call_anthropic_with_timeout and its payload.
            resp = await with_circuit_breaker(
                call_anthropic_with_timeout,
                payload
            )
            break
        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status is None:
                rproto = getattr(e, "response", None)
                status = getattr(rproto, "status_code", None) if rproto else None
            
            if status == 429 and attempt < max_attempts - 1:
                fallback = get_fallback_model(model, attempt + 1)
                log.warning("OA RateLimit hit (429) for %s, rotating to %s", model, fallback)
                
                # Update metrics for the model switch
                decrement_active_requests(acquired_model)
                model = fallback
                increment_active_requests(model)
                
                payload["model"] = model
                release_concurrency_slot(acquired_model, req_id)
                await asyncio.sleep(1.0 * (2 ** attempt))
                continue
            
            release_concurrency_slot(acquired_model, req_id)
            if attempt == max_attempts - 1:
                decrement_active_requests(model)
                if isinstance(e, CircuitOpenError):
                    record_upstream_error(model, "circuit_open", 503)
                    raise HTTPException(status_code=503, detail="Upstream model circuit open")
                
                log_anthropic_error("OA create failed", payload, e)
                await emit_error(
                    "upstream_error", str(e), cf_ray=ray, model=model,
                    project_id=str(project_id) if project_id else None,
                    exception=e,
                )
                raise HTTPException(status_code=502, detail="Upstream model error")
        finally:
            if resp: # Success: release the slot for the model that actually worked
                release_concurrency_slot(acquired_model, req_id)

    if not resp:
        raise last_err or HTTPException(status_code=502, detail="Upstream model error after retries")
    tool_calls: List[Dict[str, Any]] = []
    out_text_parts: List[str] = []

    try:
        content_blocks = getattr(resp, "content", []) or (resp.get("content") if isinstance(resp, dict) else [])
        for block in content_blocks:
            if isinstance(block, dict):
                btype = block.get("type")
                btext = block.get("text", "")
            else:
                btype = getattr(block, "type", None)
                btext = getattr(block, "text", "")
            
            if btype == "text":
                out_text_parts.append(btext or "")
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

    cache_read_tokens = usage.get("cache_read_input_tokens", 0) if usage else 0
    cache_write_tokens = usage.get("cache_creation_input_tokens", 0) if usage else 0
    
    await record_usage_to_db(
        project_id, model, input_tokens, output_tokens, ray, False,
        cache_read_tokens,
        cache_write_tokens,
        gateway_tokens_saved
    )
    
    # Record Anthropic prompt cache metrics
    if cache_read_tokens > 0:
        from gateway.metrics import record_prompt_cache_tokens
        record_prompt_cache_tokens("read", model, project_name or "default", cache_read_tokens)
    if cache_write_tokens > 0:
        from gateway.metrics import record_prompt_cache_tokens
        record_prompt_cache_tokens("write", model, project_name or "default", cache_write_tokens)

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
    response.headers["X-Gateway"] = "claude-gateway"
    response.headers["X-Model-Source"] = "custom"
    
    # Set cache status based on actual cache usage
    cache_hit = usage and usage.get("cache_read_input_tokens", 0) > 0
    response.headers["X-Cache"] = "HIT" if cache_hit else "MISS"
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

    base_models = [
        {"id": "smartroute", "object": "model"},
        {"id": with_model_prefix("sonnet"), "object": "model"},
        {"id": with_model_prefix("opus"), "object": "model"},
        {"id": with_model_prefix(DEFAULT_MODEL), "object": "model"},
        {"id": with_model_prefix(OPUS_MODEL), "object": "model"},
        {"id": "sonnet", "object": "model"},
        {"id": "opus", "object": "model"},
        {"id": DEFAULT_MODEL, "object": "model"},
        {"id": OPUS_MODEL, "object": "model"},
    ]

    for m in VALID_ANTHROPIC_MODELS:
        base_models.append({"id": m, "object": "model"})
        base_models.append({"id": with_model_prefix(m), "object": "model"})

    for local_model in LOCAL_LLM_MODEL_ALLOWLIST:
        base_models.append({"id": f"local:{local_model}", "object": "model"})
        base_models.append({"id": f"ollama:{local_model}", "object": "model"})
    
    if LOCAL_LLM_DEFAULT_MODEL:
        base_models.append({"id": f"local:{LOCAL_LLM_DEFAULT_MODEL}", "object": "model"})

    seen = set()
    data = []
    for item in base_models:
        if item["id"] not in seen:
            seen.add(item["id"])
            data.append(item)

    payload = {
        "object": "list",
        "data": data,
    }
    resp = JSONResponse(content=payload)
    resp.headers["X-Gateway"] = "claude-gateway"
    resp.headers["X-Model-Source"] = "custom"
    return resp


@router.get("/models")
async def models_alias(req: Request):
    return await openai_models(req)
