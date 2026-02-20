import asyncio
import json
import traceback
from typing import Dict, Any, Optional

from fastapi import HTTPException
from anthropic import Anthropic

from gateway.config import ANTHROPIC_API_KEY, UPSTREAM_TIMEOUT_SECONDS
from gateway.logging_setup import log


# ---------------------------------------------------------
# Client
# ---------------------------------------------------------

client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# ---------------------------------------------------------
# Main Create Call (non-stream)
# ---------------------------------------------------------

async def call_anthropic_with_timeout(payload: Dict[str, Any]):
    """
    Anthropic SDK call is synchronous.
    Run it in a thread + enforce timeout.
    Logs full upstream error body on failure.
    """
    if client is None:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    async def _run():
        return await asyncio.to_thread(lambda: client.messages.create(**payload))

    try:
        return await asyncio.wait_for(_run(), timeout=UPSTREAM_TIMEOUT_SECONDS)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Upstream model timed out")

    except Exception as e:
        log_anthropic_error("ANTHROPIC create failed", payload, e)
        raise


# ---------------------------------------------------------
# Extraction Helpers
# ---------------------------------------------------------

def extract_text_from_anthropic(resp) -> str:
    return "".join(
        block.text
        for block in getattr(resp, "content", [])
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


def anthropic_to_openai_usage(usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not usage:
        return None
    try:
        # Anthropic 'input_tokens' only counts the *un-cached* mapped tokens.
        # So we want total prompt tokens to be billed + cached to match OpenAI's total input expectation.
        billed_prompt = int(usage.get("input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        
        total_prompt = billed_prompt + cache_creation + cache_read
        completion = int(usage.get("output_tokens", 0) or 0)
        
        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": completion,
            "total_tokens": total_prompt + completion,
            "prompt_tokens_details": {
                "cached_tokens": cache_read
            },
            # Map native Anthropic fields too just so client can see them
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        }
    except Exception:
        return None


# ---------------------------------------------------------
# Centralized Error Logger
# ---------------------------------------------------------

def log_anthropic_error(prefix: str, payload: Dict[str, Any], err: Exception):
    """
    Logs:
      - error repr
      - upstream response JSON/text (if present)
      - compact payload summary (roles + tool ids)
      - traceback
    Never throws.
    """
    try:
        log.error("%s: %r", prefix, err)

        # ---- Try to extract HTTP response body ----
        resp = getattr(err, "response", None)
        if resp is not None:
            try:
                j = resp.json()
                log.error(
                    "%s response.json: %s",
                    prefix,
                    json.dumps(j, ensure_ascii=False)[:8000],
                )
            except Exception:
                try:
                    txt = getattr(resp, "text", "")
                    log.error(
                        "%s response.text: %s",
                        prefix,
                        (txt or "")[:8000],
                    )
                except Exception:
                    pass

        # ---- Compact payload summary ----
        msgs = payload.get("messages") or []
        summary = []

        for m in msgs[-12:]:  # last 12 only
            role = m.get("role")
            content = m.get("content")
            item = {"role": role}

            if isinstance(content, str):
                item["content_type"] = "text"
                item["content_len"] = len(content)

            elif isinstance(content, list):
                item["content_type"] = "blocks"
                item["blocks"] = []

                for b in content:
                    if not isinstance(b, dict):
                        continue

                    btype = b.get("type")

                    if btype == "tool_use":
                        item["blocks"].append({
                            "tool_use_id": b.get("id"),
                            "name": b.get("name"),
                        })

                    elif btype == "tool_result":
                        item["blocks"].append({
                            "tool_result_for": b.get("tool_use_id"),
                        })

                    elif btype == "text":
                        item["blocks"].append({
                            "text_len": len(b.get("text") or "")
                        })

                    else:
                        item["blocks"].append({"type": btype})

            summary.append(item)

        log.error(
            "%s payload_summary: %s",
            prefix,
            json.dumps({
                "model": payload.get("model"),
                "has_tools": bool(payload.get("tools")),
                "tool_choice": payload.get("tool_choice"),
                "system_len": len(payload.get("system") or ""),
                "messages": summary,
            }, ensure_ascii=False)[:8000],
        )

        log.error("%s traceback:\n%s", prefix, traceback.format_exc())

    except Exception:
        # Logging should never crash your app
        pass
