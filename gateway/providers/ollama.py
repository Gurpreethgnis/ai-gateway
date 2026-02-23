"""
Ollama provider client for local LLM via Cloudflare Access tunnel.

This module handles:
- Building requests to Ollama's /api/chat endpoint
- Adding Cloudflare Access service token headers
- Normalizing responses to match the gateway's expected format
- Model allowlist enforcement
- Error handling with proper status codes
"""

import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from gateway.config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_DEFAULT_MODEL,
    LOCAL_LLM_TIMEOUT_SECONDS,
    LOCAL_CF_ACCESS_CLIENT_ID,
    LOCAL_CF_ACCESS_CLIENT_SECRET,
    LOCAL_LLM_MODEL_ALLOWLIST,
)
from gateway.logging_setup import log


def _ollama_error_detail(status_code: int, response_text: str, model: str, preflight_ok: bool = True) -> str:
    """Build a clear error message for Ollama upstream errors."""
    if status_code == 404:
        if preflight_ok:
            return (
                f"POST /api/chat returned 404 (GET /api/tags succeeded, so URL and CF Access are OK). "
                f"Model '{model}' may not exist on the server - run 'ollama pull {model}' and ensure the name matches 'ollama list'. "
                "Or your tunnel/proxy may not allow POST to /api/chat."
            )
        return (
            f"Ollama returned 404 for model '{model}'. "
            f"Run: ollama pull {model} and ensure LOCAL_LLM_BASE_URL and CF Access credentials in Railway match your working curl."
        )
    return f"Local LLM error (status={status_code}): {response_text}"


class OllamaConfigError(Exception):
    """Raised when Ollama provider is misconfigured."""
    pass


class OllamaUpstreamError(Exception):
    """Raised when Ollama returns a non-2xx response."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Ollama upstream error {status_code}: {detail}")


def validate_local_config() -> None:
    """
    Validate that all required environment variables for local LLM are set.
    Raises HTTPException(500) if any are missing.
    """
    missing = []
    if not LOCAL_LLM_BASE_URL:
        missing.append("LOCAL_LLM_BASE_URL")
    if not LOCAL_CF_ACCESS_CLIENT_ID:
        missing.append("LOCAL_CF_ACCESS_CLIENT_ID")
    if not LOCAL_CF_ACCESS_CLIENT_SECRET:
        missing.append("LOCAL_CF_ACCESS_CLIENT_SECRET")
    
    if missing:
        msg = f"Local LLM provider misconfigured. Missing env vars: {', '.join(missing)}"
        log.error(msg)
        raise HTTPException(status_code=500, detail=msg)


def validate_local_model(model: Optional[str]) -> str:
    """
    Validate and return the model to use.
    - If model is None/empty, returns default model
    - If model is not in allowlist, raises HTTPException(400)
    """
    if not model:
        return LOCAL_LLM_DEFAULT_MODEL
    
    if model not in LOCAL_LLM_MODEL_ALLOWLIST:
        log.warning("Rejected local model request for: %s", model)
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not in local allowlist. Allowed models: {LOCAL_LLM_MODEL_ALLOWLIST}"
        )
    
    return model


def build_ollama_headers() -> Dict[str, str]:
    """Build headers for Ollama request including CF Access tokens."""
    return {
        "CF-Access-Client-Id": LOCAL_CF_ACCESS_CLIENT_ID,
        "CF-Access-Client-Secret": LOCAL_CF_ACCESS_CLIENT_SECRET,
        "Content-Type": "application/json",
    }


async def _ollama_preflight(base_url: str, headers: Dict[str, str], timeout: float = 10.0) -> tuple[bool, str]:
    """
    GET /api/tags to verify Ollama base URL and CF Access work from this environment.
    Returns (success, message). If False, message describes the failure.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            return True, ""
        body = (resp.text or resp.content.decode("utf-8", errors="replace"))[:200] or "(empty)"
        return False, f"GET /api/tags returned {resp.status_code}: {body}"
    except Exception as e:
        return False, str(e)[:200]


def build_ollama_payload(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build Ollama API payload.
    
    Ollama expects:
    {
        "model": "<model>",
        "messages": [...],
        "stream": false,
        "options": { "temperature": <float>, "num_predict": <int> }
    }
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    
    options: Dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    
    if options:
        payload["options"] = options
    
    return payload


def normalize_ollama_response(
    ollama_resp: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """
    Normalize Ollama response to match gateway's /chat response format.
    
    Gateway /chat returns:
    {
        "cached": False,
        "model": str,
        "text": str,
        "usage": { "input_tokens": int, "output_tokens": int, ... } | None
    }
    
    Ollama returns:
    {
        "model": str,
        "message": { "role": "assistant", "content": str },
        "done": bool,
        "prompt_eval_count": int,  # input tokens
        "eval_count": int,         # output tokens
        ...
    }
    """
    message = ollama_resp.get("message", {})
    text = message.get("content", "")
    
    prompt_tokens = ollama_resp.get("prompt_eval_count", 0) or 0
    completion_tokens = ollama_resp.get("eval_count", 0) or 0
    
    usage = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    
    return {
        "cached": False,
        "model": model,
        "text": text,
        "usage": usage,
    }


def normalize_ollama_response_openai(
    ollama_resp: Dict[str, Any],
    model: str,
    request_id: str,
) -> Dict[str, Any]:
    """
    Normalize Ollama response to OpenAI chat.completion format.
    
    OpenAI format:
    {
        "id": str,
        "object": "chat.completion",
        "created": int,
        "model": str,
        "choices": [{ "index": 0, "message": {...}, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": int, "completion_tokens": int, "total_tokens": int }
    }
    """
    message = ollama_resp.get("message", {})
    text = message.get("content", "")
    
    prompt_tokens = ollama_resp.get("prompt_eval_count", 0) or 0
    completion_tokens = ollama_resp.get("eval_count", 0) or 0
    
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"local:{model}",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def call_ollama(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Call Ollama via Cloudflare Access tunnel.
    
    Returns normalized response in gateway /chat format.
    
    Raises:
        HTTPException(500): If configuration is missing
        HTTPException(400): If model not in allowlist
        HTTPException(502): If Ollama returns error or is unreachable
    """
    validate_local_config()
    resolved_model = validate_local_model(model)
    
    base_url = LOCAL_LLM_BASE_URL.rstrip("/")
    url = f"{base_url}/api/chat"
    headers = build_ollama_headers()
    payload = build_ollama_payload(resolved_model, messages, temperature, max_tokens)
    
    # Preflight: verify base URL and CF Access work (GET /api/tags)
    preflight_ok, preflight_msg = await _ollama_preflight(base_url, headers, timeout=8.0)
    if not preflight_ok:
        host = base_url.split("//", 1)[-1].split("/")[0] if "//" in base_url else base_url[:50]
        log.error("OLLAMA PREFLIGHT FAILED host=%s msg=%s", host, preflight_msg)
        raise HTTPException(
            status_code=502,
            detail=(
                "Ollama unreachable from gateway. GET /api/tags failed: " + preflight_msg + ". "
                "Check LOCAL_LLM_BASE_URL and LOCAL_CF_ACCESS_CLIENT_ID / LOCAL_CF_ACCESS_CLIENT_SECRET in Railway match your curl; same host and CF Access headers."
            )
        )
    
    t0 = time.time()
    host = base_url.split("//", 1)[-1].split("/")[0] if "//" in base_url else base_url[:40]
    log.info("OLLAMA REQUEST host=%s model=%s messages=%d", host, resolved_model, len(messages))
    
    try:
        async with httpx.AsyncClient(timeout=LOCAL_LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException:
        elapsed_ms = int((time.time() - t0) * 1000)
        log.error("OLLAMA TIMEOUT model=%s ms=%d", resolved_model, elapsed_ms)
        raise HTTPException(
            status_code=504,
            detail=f"Local LLM timed out after {LOCAL_LLM_TIMEOUT_SECONDS}s"
        )
    except httpx.RequestError as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        log.error("OLLAMA CONNECTION ERROR model=%s ms=%d error=%r", resolved_model, elapsed_ms, e)
        raise HTTPException(
            status_code=502,
            detail=f"Local LLM unreachable: {str(e)[:200]}"
        )
    
    elapsed_ms = int((time.time() - t0) * 1000)
    
    if resp.status_code >= 400:
        response_text = (resp.text or resp.content.decode("utf-8", errors="replace"))[:500] or "(empty)"
        log.error(
            "OLLAMA UPSTREAM ERROR model=%s status=%d ms=%d response=%s",
            resolved_model, resp.status_code, elapsed_ms, response_text
        )
        detail = _ollama_error_detail(resp.status_code, response_text, resolved_model, preflight_ok=True)
        raise HTTPException(status_code=502, detail=detail)
    
    try:
        ollama_data = resp.json()
    except Exception as e:
        log.error("OLLAMA PARSE ERROR model=%s ms=%d error=%r", resolved_model, elapsed_ms, e)
        raise HTTPException(
            status_code=502,
            detail="Local LLM returned invalid JSON"
        )
    
    result = normalize_ollama_response(ollama_data, resolved_model)
    
    input_tokens = result.get("usage", {}).get("input_tokens", 0)
    output_tokens = result.get("usage", {}).get("output_tokens", 0)
    log.info(
        "OLLAMA OK model=%s ms=%d input_tokens=%d output_tokens=%d",
        resolved_model, elapsed_ms, input_tokens, output_tokens
    )
    
    return result


async def call_ollama_openai_format(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    request_id: str = "chatcmpl_local",
) -> Dict[str, Any]:
    """
    Call Ollama and return response in OpenAI chat.completion format.
    
    This is used by the /v1/chat/completions endpoint.
    """
    validate_local_config()
    resolved_model = validate_local_model(model)
    
    base_url = LOCAL_LLM_BASE_URL.rstrip("/")
    url = f"{base_url}/api/chat"
    headers = build_ollama_headers()
    payload = build_ollama_payload(resolved_model, messages, temperature, max_tokens)
    
    preflight_ok, preflight_msg = await _ollama_preflight(base_url, headers, timeout=8.0)
    if not preflight_ok:
        log.error("OLLAMA PREFLIGHT FAILED (OpenAI) msg=%s", preflight_msg)
        raise HTTPException(
            status_code=502,
            detail="Ollama unreachable from gateway. GET /api/tags failed: " + preflight_msg + ". Check LOCAL_LLM_BASE_URL and CF Access credentials in Railway."
        )
    
    t0 = time.time()
    log.info("OLLAMA REQUEST (OpenAI) model=%s messages=%d", resolved_model, len(messages))
    
    try:
        async with httpx.AsyncClient(timeout=LOCAL_LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException:
        elapsed_ms = int((time.time() - t0) * 1000)
        log.error("OLLAMA TIMEOUT (OpenAI) model=%s ms=%d", resolved_model, elapsed_ms)
        raise HTTPException(
            status_code=504,
            detail=f"Local LLM timed out after {LOCAL_LLM_TIMEOUT_SECONDS}s"
        )
    except httpx.RequestError as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        log.error("OLLAMA CONNECTION ERROR (OpenAI) model=%s ms=%d error=%r", resolved_model, elapsed_ms, e)
        raise HTTPException(
            status_code=502,
            detail=f"Local LLM unreachable: {str(e)[:200]}"
        )
    
    elapsed_ms = int((time.time() - t0) * 1000)
    
    if resp.status_code >= 400:
        response_text = (resp.text or resp.content.decode("utf-8", errors="replace"))[:500] or "(empty)"
        log.error(
            "OLLAMA UPSTREAM ERROR (OpenAI) model=%s status=%d ms=%d response=%s",
            resolved_model, resp.status_code, elapsed_ms, response_text
        )
        detail = _ollama_error_detail(resp.status_code, response_text, resolved_model, preflight_ok=True)
        raise HTTPException(status_code=502, detail=detail)
    
    try:
        ollama_data = resp.json()
    except Exception as e:
        log.error("OLLAMA PARSE ERROR (OpenAI) model=%s ms=%d error=%r", resolved_model, elapsed_ms, e)
        raise HTTPException(
            status_code=502,
            detail="Local LLM returned invalid JSON"
        )
    
    result = normalize_ollama_response_openai(ollama_data, resolved_model, request_id)
    
    usage = result.get("usage", {})
    log.info(
        "OLLAMA OK (OpenAI) model=%s ms=%d input_tokens=%d output_tokens=%d",
        resolved_model, elapsed_ms, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    )
    
    return result
