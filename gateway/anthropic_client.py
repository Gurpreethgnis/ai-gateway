import asyncio
from typing import Dict, Any, Optional

from fastapi import HTTPException
from anthropic import Anthropic

from gateway.config import ANTHROPIC_API_KEY, UPSTREAM_TIMEOUT_SECONDS
from gateway.logging_setup import log

client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

async def call_anthropic_with_timeout(payload: Dict[str, Any]):
    if client is None:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    async def _run():
        return await asyncio.to_thread(lambda: client.messages.create(**payload))

    try:
        return await asyncio.wait_for(_run(), timeout=UPSTREAM_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Upstream model timed out")

def extract_text_from_anthropic(resp) -> str:
    return "".join(
        block.text for block in getattr(resp, "content", [])
        if getattr(block, "type", None) == "text"
    )

def extract_usage(resp) -> Optional[Dict[str, Any]]:
    usage_obj = getattr(resp, "usage", None)
    if usage_obj is None:
        return None
    try:
        return usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj
    except Exception:
        return None

def anthropic_to_openai_usage(usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not usage:
        return None
    try:
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    except Exception:
        return None
