import time
import traceback
from fastapi import APIRouter, Request, HTTPException

from gateway.models import ChatReq
from gateway.routing import route_model_from_messages
from gateway.anthropic_client import call_anthropic_with_timeout, extract_text_from_anthropic, extract_usage
from gateway.cache import cache_key, cache_get, cache_set
from gateway.config import MAX_BODY_BYTES, CACHE_TTL_SECONDS
from gateway.logging_setup import log

router = APIRouter()


async def _handle_local_provider(body: ChatReq, ray: str) -> dict:
    """Handle chat request via local Ollama provider."""
    from gateway.providers.ollama import call_ollama
    
    t0 = time.time()
    
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    
    if body.system:
        messages.insert(0, {"role": "system", "content": body.system})
    
    result = await call_ollama(
        messages=messages,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    
    dt_ms = int((time.time() - t0) * 1000)
    log.info(
        "CHAT OK (cf-ray=%s) provider=local model=%s ms=%d",
        ray, result.get("model"), dt_ms
    )
    
    return result


@router.post("/chat")
async def chat(req: Request, body: ChatReq):
    raw = await req.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")

    ray = req.headers.get("cf-ray") or ""
    t0 = time.time()
    
    # Skills system: check for X-Gateway-Skill header
    skill_header = req.headers.get("x-gateway-skill")
    if skill_header:
        from gateway.config import ENABLE_SKILLS
        if ENABLE_SKILLS:
            from gateway.skills import apply_skill_to_system_prompt
            original_system = body.system or ""
            enhanced_system = apply_skill_to_system_prompt(original_system, skill_header)
            if enhanced_system != original_system:
                log.info("CHAT skill applied: %s", skill_header)
                body.system = enhanced_system

    # Explicit provider request
    if body.provider == "local":
        return await _handle_local_provider(body, ray)
    
    # Smart routing when no explicit provider
    if not body.provider and body.model in (None, "auto", "smartroute"):
        try:
            from gateway.config import ENABLE_SMART_ROUTING, ENABLE_CASCADE_ROUTING
            
            if ENABLE_SMART_ROUTING:
                messages = [{"role": m.role, "content": m.content} for m in body.messages]
                
                if ENABLE_CASCADE_ROUTING:
                    # Use cascade routing
                    from gateway.cascade_router import route_with_cascade
                    decision, local_response, cascade_metadata = await route_with_cascade(
                        messages=messages,
                        tools=[],  # Simple chat endpoint doesn't support tools
                        project_id=None,
                        system_prompt=body.system or "",
                        explicit_model=body.model,
                    )
                    
                    if local_response:
                        # Return local response immediately
                        log.info("CHAT cascade: returning local response")
                        return local_response
                    
                    model = decision.model
                    log.info("CHAT cascade -> %s (tier=%s, escalated=%s)", 
                            model, decision.tier, cascade_metadata.get("escalated"))
                else:
                    # Standard smart routing
                    from gateway.smart_routing import route_request
                    decision = await route_request(
                        messages=messages,
                        tools=[],
                        project_id=None,
                        explicit_model=body.model,
                        system_prompt=body.system or ""
                    )
                    
                    if decision.provider == "local":
                        log.info("CHAT smart routing -> LOCAL (tier=%s, phase=%s)", decision.tier, decision.phase)
                        return await _handle_local_provider(body, ray)
                    
                    model = decision.model
                    log.info("CHAT smart routing -> %s (tier=%s, phase=%s)", model, decision.tier, decision.phase)
            else:
                joined_user = "\n".join(m.content for m in body.messages if m.role == "user")
                model = route_model_from_messages(joined_user, body.model)
        except Exception as e:
            log.warning("CHAT smart routing failed: %r, using fallback", e)
            joined_user = "\n".join(m.content for m in body.messages if m.role == "user")
            model = route_model_from_messages(joined_user, body.model)
    else:
        # Explicit model or provider specified
        joined_user = "\n".join(m.content for m in body.messages if m.role == "user")
        model = route_model_from_messages(joined_user, body.model)

    payload = {
        "model": model,
        "system": body.system or "",
        "messages": [{"role": m.role, "content": m.content} for m in body.messages],
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
    }

    do_cache = body.temperature is None or body.temperature <= 0.3
    key = cache_key(payload) if do_cache else None

    if key:
        cached = cache_get(key)
        if cached:
            cached["cached"] = True
            return cached

    try:
        resp = await call_anthropic_with_timeout(payload)
    except HTTPException:
        raise
    except Exception as e:
        log.error("CHAT UPSTREAM ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream model error")

    try:
        text = extract_text_from_anthropic(resp)
    except Exception as e:
        log.error("CHAT PARSE ERROR (cf-ray=%s): %r", ray, e)
        log.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail="Upstream response parse error")

    usage = extract_usage(resp)
    data = {"cached": False, "model": model, "text": text, "usage": usage}

    if key:
        cache_set(key, data, CACHE_TTL_SECONDS)

    dt_ms = int((time.time() - t0) * 1000)
    log.info("CHAT OK (cf-ray=%s) provider=anthropic model=%s ms=%s cached=%s", ray, model, dt_ms, False)
    return data
