import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from gateway.config import ADMIN_API_KEY, DATABASE_URL, PROMETHEUS_ENABLED
from gateway.logging_setup import log
from gateway.metrics import get_metrics_output, get_metrics_content_type
from gateway.telemetry import get_recent_errors, get_error_stats

router = APIRouter(prefix="/admin", tags=["admin"])


def verify_admin_key(request: Request):
    auth = request.headers.get("authorization") or ""
    api_key = request.headers.get("x-api-key") or request.headers.get("api-key")

    if auth.lower().startswith("bearer "):
        api_key = auth.split(" ", 1)[1].strip()

    if not api_key or api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin API key required")


class ProjectCreate(BaseModel):
    name: str
    api_key: str
    rate_limit_rpm: int = 60
    config: Optional[Dict[str, Any]] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    rate_limit_rpm: Optional[int] = None
    config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


@router.get("/metrics")
async def prometheus_metrics(request: Request):
    if not PROMETHEUS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    return Response(
        content=get_metrics_output(),
        media_type=get_metrics_content_type(),
    )


@router.get("/errors")
async def get_errors(request: Request, count: int = Query(default=100, le=1000)):
    verify_admin_key(request)
    errors = await get_recent_errors(count)
    return {"errors": errors, "count": len(errors)}


@router.get("/errors/stats")
async def get_errors_stats(request: Request, hours: int = Query(default=24, le=168)):
    verify_admin_key(request)
    stats = await get_error_stats(hours)
    return stats


@router.get("/usage")
async def get_usage(
    request: Request,
    project_id: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    verify_admin_key(request)

    if not DATABASE_URL:
        return {"error": "Database not configured", "usage": []}

    try:
        from gateway.db import get_session, UsageRecord
        from sqlalchemy import select, func

        async with get_session() as session:
            query = select(
                UsageRecord.model,
                func.sum(UsageRecord.input_tokens).label("total_input"),
                func.sum(UsageRecord.output_tokens).label("total_output"),
                func.sum(UsageRecord.cost_usd).label("total_cost"),
                func.count(UsageRecord.id).label("request_count"),
            )

            if project_id:
                query = query.where(UsageRecord.project_id == project_id)

            if start:
                try:
                    start_dt = datetime.fromisoformat(start)
                    query = query.where(UsageRecord.timestamp >= start_dt)
                except ValueError:
                    pass

            if end:
                try:
                    end_dt = datetime.fromisoformat(end)
                    query = query.where(UsageRecord.timestamp <= end_dt)
                except ValueError:
                    pass

            query = query.group_by(UsageRecord.model)
            result = await session.execute(query)
            rows = result.all()

            usage = [
                {
                    "model": row.model,
                    "input_tokens": int(row.total_input or 0),
                    "output_tokens": int(row.total_output or 0),
                    "cost_usd": float(row.total_cost or 0),
                    "request_count": int(row.request_count or 0),
                }
                for row in rows
            ]

            return {"usage": usage, "project_id": project_id}

    except Exception as e:
        log.error("Failed to get usage: %r", e)
        return {"error": str(e), "usage": []}


@router.get("/usage/daily")
async def get_daily_usage(
    request: Request,
    project_id: Optional[int] = None,
    days: int = Query(default=7, le=90),
):
    verify_admin_key(request)

    if not DATABASE_URL:
        return {"error": "Database not configured", "daily": []}

    try:
        from gateway.db import get_session, UsageRecord
        from sqlalchemy import select, func, cast, Date

        start_dt = datetime.utcnow() - timedelta(days=days)

        async with get_session() as session:
            query = select(
                func.date(UsageRecord.timestamp).label("date"),
                UsageRecord.model,
                func.sum(UsageRecord.input_tokens).label("input_tokens"),
                func.sum(UsageRecord.output_tokens).label("output_tokens"),
                func.sum(UsageRecord.cost_usd).label("cost_usd"),
                func.count(UsageRecord.id).label("requests"),
            ).where(UsageRecord.timestamp >= start_dt)

            if project_id:
                query = query.where(UsageRecord.project_id == project_id)

            query = query.group_by(func.date(UsageRecord.timestamp), UsageRecord.model)
            query = query.order_by(func.date(UsageRecord.timestamp).desc())

            result = await session.execute(query)
            rows = result.all()

            daily = [
                {
                    "date": str(row.date),
                    "model": row.model,
                    "input_tokens": int(row.input_tokens or 0),
                    "output_tokens": int(row.output_tokens or 0),
                    "cost_usd": float(row.cost_usd or 0),
                    "requests": int(row.requests or 0),
                }
                for row in rows
            ]

            return {"daily": daily, "days": days, "project_id": project_id}

    except Exception as e:
        log.error("Failed to get daily usage: %r", e)
        return {"error": str(e), "daily": []}


@router.get("/costs")
async def get_costs(
    request: Request,
    project_id: Optional[int] = None,
):
    verify_admin_key(request)

    if not DATABASE_URL:
        return {"error": "Database not configured"}

    try:
        from gateway.db import get_session, UsageRecord
        from sqlalchemy import select, func

        async with get_session() as session:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today - timedelta(days=today.weekday())
            month_start = today.replace(day=1)

            periods = {
                "today": today,
                "this_week": week_start,
                "this_month": month_start,
                "all_time": None,
            }

            costs = {}
            for period_name, start_dt in periods.items():
                query = select(func.sum(UsageRecord.cost_usd))
                if project_id:
                    query = query.where(UsageRecord.project_id == project_id)
                if start_dt:
                    query = query.where(UsageRecord.timestamp >= start_dt)

                result = await session.execute(query)
                total = result.scalar() or 0
                costs[period_name] = round(float(total), 4)

            return {"costs": costs, "project_id": project_id}

    except Exception as e:
        log.error("Failed to get costs: %r", e)
        return {"error": str(e)}


@router.get("/projects")
async def list_projects(request: Request):
    verify_admin_key(request)

    if not DATABASE_URL:
        return {"error": "Database not configured", "projects": []}

    try:
        from gateway.db import get_session, Project, UsageRecord
        from sqlalchemy import select, func

        async with get_session() as session:
            query = select(
                Project,
                func.sum(UsageRecord.cost_usd).label("total_cost"),
                func.count(UsageRecord.id).label("request_count"),
            ).outerjoin(UsageRecord).group_by(Project.id)

            result = await session.execute(query)
            rows = result.all()

            projects = [
                {
                    "id": row.Project.id,
                    "name": row.Project.name,
                    "rate_limit_rpm": row.Project.rate_limit_rpm,
                    "is_active": row.Project.is_active,
                    "created_at": row.Project.created_at.isoformat(),
                    "total_cost": round(float(row.total_cost or 0), 4),
                    "request_count": int(row.request_count or 0),
                }
                for row in rows
            ]

            return {"projects": projects}

    except Exception as e:
        log.error("Failed to list projects: %r", e)
        return {"error": str(e), "projects": []}


@router.post("/projects")
async def create_project(request: Request, body: ProjectCreate):
    verify_admin_key(request)

    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        from gateway.db import get_session, Project, hash_api_key

        async with get_session() as session:
            project = Project(
                name=body.name,
                api_key_hash=hash_api_key(body.api_key),
                rate_limit_rpm=body.rate_limit_rpm,
                config_json=json.dumps(body.config or {}),
            )
            session.add(project)
            await session.flush()

            return {
                "id": project.id,
                "name": project.name,
                "rate_limit_rpm": project.rate_limit_rpm,
                "created_at": project.created_at.isoformat(),
            }

    except Exception as e:
        log.error("Failed to create project: %r", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/projects/{project_id}")
async def update_project(request: Request, project_id: int, body: ProjectUpdate):
    verify_admin_key(request)

    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        from gateway.db import get_session, Project
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(Project).where(Project.id == project_id)
            )
            project = result.scalar_one_or_none()

            if not project:
                raise HTTPException(status_code=404, detail="Project not found")

            if body.name is not None:
                project.name = body.name
            if body.rate_limit_rpm is not None:
                project.rate_limit_rpm = body.rate_limit_rpm
            if body.is_active is not None:
                project.is_active = body.is_active
            if body.config is not None:
                existing = json.loads(project.config_json or "{}")
                existing.update(body.config)
                project.config_json = json.dumps(existing)

            return {"id": project.id, "name": project.name, "updated": True}

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to update project: %r", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projects/{project_id}")
async def delete_project(request: Request, project_id: int):
    verify_admin_key(request)

    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        from gateway.db import get_session, Project
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(Project).where(Project.id == project_id)
            )
            project = result.scalar_one_or_none()

            if not project:
                raise HTTPException(status_code=404, detail="Project not found")

            project.is_active = False
            return {"id": project_id, "deleted": True}

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to delete project: %r", e)
        raise HTTPException(status_code=500, detail=str(e))
