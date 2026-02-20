import json
import asyncio
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from gateway.config import ENABLE_BATCH_API
from gateway.cache import rds
from gateway.logging_setup import log


router = APIRouter(prefix="/v1/batch", tags=["batch"])


@dataclass
class BatchItem:
    custom_id: str
    method: str
    url: str
    body: Dict[str, Any]


@dataclass
class BatchResult:
    custom_id: str
    status: str
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchRequest(BaseModel):
    requests: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class BatchStatus(BaseModel):
    batch_id: str
    status: str
    total: int
    completed: int
    failed: int
    created_at: str
    completed_at: Optional[str] = None


def _batch_key(batch_id: str) -> str:
    return f"batch:{batch_id}"


def _batch_results_key(batch_id: str) -> str:
    return f"batch:{batch_id}:results"


def _batch_meta_key(batch_id: str) -> str:
    return f"batch:{batch_id}:meta"


async def process_single_request(item: BatchItem) -> BatchResult:
    try:
        from gateway.anthropic_client import call_anthropic_with_timeout, extract_text_from_anthropic, extract_usage
        from gateway.routing import route_model_from_messages

        body = item.body
        messages = body.get("messages", [])
        model = body.get("model")

        user_text = "\n".join(
            m.get("content", "") for m in messages
            if isinstance(m.get("content"), str) and m.get("role") == "user"
        )
        selected_model = route_model_from_messages(user_text, model)

        payload = {
            "model": selected_model,
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if m.get("role") in ("user", "assistant")
            ],
            "max_tokens": body.get("max_tokens", 1024),
            "temperature": body.get("temperature", 0.2),
        }

        if body.get("system"):
            payload["system"] = body["system"]

        resp = await call_anthropic_with_timeout(payload)
        text = extract_text_from_anthropic(resp)
        usage = extract_usage(resp)

        return BatchResult(
            custom_id=item.custom_id,
            status="success",
            response={
                "id": f"batch_{item.custom_id}",
                "model": selected_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
        )

    except Exception as e:
        log.error("Batch item %s failed: %r", item.custom_id, e)
        return BatchResult(
            custom_id=item.custom_id,
            status="failed",
            error=str(e),
        )


async def process_batch(batch_id: str):
    if rds is None:
        log.error("Redis not available for batch processing")
        return

    log.info("Starting batch processing: %s", batch_id)

    try:
        while True:
            item_json = rds.rpop(_batch_key(batch_id))
            if not item_json:
                break

            if isinstance(item_json, bytes):
                item_json = item_json.decode()

            item_data = json.loads(item_json)
            item = BatchItem(**item_data)

            result = await process_single_request(item)

            rds.lpush(_batch_results_key(batch_id), json.dumps(asdict(result)))

            await asyncio.sleep(0.1)

        rds.hset(_batch_meta_key(batch_id), "status", "completed")
        rds.hset(_batch_meta_key(batch_id), "completed_at", datetime.utcnow().isoformat())

        log.info("Batch completed: %s", batch_id)

    except Exception as e:
        log.error("Batch processing failed: %s - %r", batch_id, e)
        if rds:
            rds.hset(_batch_meta_key(batch_id), "status", "failed")
            rds.hset(_batch_meta_key(batch_id), "error", str(e))


@router.post("")
async def create_batch(request: Request, body: BatchRequest):
    if not ENABLE_BATCH_API:
        raise HTTPException(status_code=404, detail="Batch API not enabled")

    if rds is None:
        raise HTTPException(status_code=500, detail="Redis required for batch processing")

    if len(body.requests) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 requests per batch")

    batch_id = str(uuid4())

    for i, req in enumerate(body.requests):
        item = BatchItem(
            custom_id=req.get("custom_id", f"request_{i}"),
            method=req.get("method", "POST"),
            url=req.get("url", "/v1/chat/completions"),
            body=req.get("body", req),
        )
        rds.lpush(_batch_key(batch_id), json.dumps(asdict(item)))

    rds.hset(_batch_meta_key(batch_id), mapping={
        "status": "processing",
        "total": str(len(body.requests)),
        "created_at": datetime.utcnow().isoformat(),
    })

    rds.expire(_batch_key(batch_id), 86400)
    rds.expire(_batch_results_key(batch_id), 86400)
    rds.expire(_batch_meta_key(batch_id), 86400)

    asyncio.create_task(process_batch(batch_id))

    return {
        "batch_id": batch_id,
        "status": "processing",
        "total": len(body.requests),
        "created_at": datetime.utcnow().isoformat(),
    }


@router.get("/{batch_id}")
async def get_batch_status(batch_id: str):
    if not ENABLE_BATCH_API:
        raise HTTPException(status_code=404, detail="Batch API not enabled")

    if rds is None:
        raise HTTPException(status_code=500, detail="Redis required")

    meta = rds.hgetall(_batch_meta_key(batch_id))
    if not meta:
        raise HTTPException(status_code=404, detail="Batch not found")

    def decode_val(v):
        return v.decode() if isinstance(v, bytes) else v

    meta = {decode_val(k): decode_val(v) for k, v in meta.items()}

    pending = rds.llen(_batch_key(batch_id))
    completed = rds.llen(_batch_results_key(batch_id))
    total = int(meta.get("total", 0))

    failed = 0
    results = rds.lrange(_batch_results_key(batch_id), 0, -1)
    for r in results:
        try:
            data = json.loads(r.decode() if isinstance(r, bytes) else r)
            if data.get("status") == "failed":
                failed += 1
        except Exception:
            pass

    return {
        "batch_id": batch_id,
        "status": meta.get("status", "unknown"),
        "total": total,
        "completed": completed,
        "pending": pending,
        "failed": failed,
        "created_at": meta.get("created_at"),
        "completed_at": meta.get("completed_at"),
    }


@router.get("/{batch_id}/results")
async def get_batch_results(batch_id: str, offset: int = 0, limit: int = 100):
    if not ENABLE_BATCH_API:
        raise HTTPException(status_code=404, detail="Batch API not enabled")

    if rds is None:
        raise HTTPException(status_code=500, detail="Redis required")

    results_raw = rds.lrange(_batch_results_key(batch_id), offset, offset + limit - 1)

    results = []
    for r in results_raw:
        try:
            data = json.loads(r.decode() if isinstance(r, bytes) else r)
            results.append(data)
        except Exception:
            continue

    total = rds.llen(_batch_results_key(batch_id))

    return {
        "batch_id": batch_id,
        "results": results,
        "offset": offset,
        "limit": limit,
        "total": total,
    }


@router.delete("/{batch_id}")
async def cancel_batch(batch_id: str):
    if not ENABLE_BATCH_API:
        raise HTTPException(status_code=404, detail="Batch API not enabled")

    if rds is None:
        raise HTTPException(status_code=500, detail="Redis required")

    rds.delete(_batch_key(batch_id))
    rds.hset(_batch_meta_key(batch_id), "status", "cancelled")

    return {"batch_id": batch_id, "status": "cancelled"}
