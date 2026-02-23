"""
Dashboard API Router - Preferences, model management, Ollama control.

Provides:
- GET/PUT /api/preferences - routing preferences (sliders)
- GET /api/models - all available models
- PUT /api/models/{id}/enabled - toggle model
- GET /api/ollama/models - local Ollama models
- POST /api/ollama/pull - trigger model download
- DELETE /api/ollama/models/{name} - delete model
- GET /api/stats - usage statistics
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from gateway.logging_setup import log
from gateway.routers.auth import require_auth
from gateway import config


router = APIRouter(prefix="/api", tags=["dashboard-api"])


# =============================================================================
# Pydantic Models
# =============================================================================

class RoutingPreferences(BaseModel):
    cost_quality_bias: float = Field(ge=0.0, le=1.0, description="0=cheapest, 1=highest quality")
    speed_quality_bias: float = Field(ge=0.0, le=1.0, description="0=fastest, 1=highest quality")
    cascade_enabled: bool = Field(description="Enable automatic fallback on errors")
    max_cascade_attempts: int = Field(ge=1, le=5, default=3)
    preferred_providers: Optional[List[str]] = Field(default=None, description="Optional provider preference order")


class ModelSettingsUpdate(BaseModel):
    is_enabled: bool
    custom_quality_rating: Optional[float] = Field(ge=0.0, le=1.0, default=None)


class OllamaPullRequest(BaseModel):
    model_name: str


class ModelInfo(BaseModel):
    id: str
    provider: str
    display_name: str
    is_enabled: bool
    quality_rating: float
    cost_per_1k_input: float
    cost_per_1k_output: float
    latency_ms: int
    context_window: int
    capabilities: List[str]


class OllamaModelInfo(BaseModel):
    name: str
    size_gb: float
    modified_at: str
    digest: str


class UsageStats(BaseModel):
    total_requests: int
    total_tokens: int
    estimated_cost: float
    cache_hits: int
    cache_hit_rate: float
    requests_by_model: dict
    requests_by_provider: dict


class PullJobStatus(BaseModel):
    id: int
    model_name: str
    status: str  # pending, pulling, completed, failed
    progress: Optional[float]
    error: Optional[str]
    created_at: datetime


# =============================================================================
# Preferences Endpoints
# =============================================================================

@router.get("/preferences")
async def get_preferences(
    request: Request,
    project_id: Optional[str] = None,
    user: dict = Depends(require_auth),
) -> RoutingPreferences:
    """Get routing preferences for the current user/project."""
    from gateway.db import get_session
    from sqlalchemy import text
    
    if project_id:
        async with get_session() as session:
            result = await session.execute(
                text("""
                    SELECT cost_quality_bias, speed_quality_bias, 
                           cascade_enabled, max_cascade_attempts
                    FROM projects
                    WHERE api_key = :project_id
                """),
                {"project_id": project_id},
            )
            row = result.fetchone()
            
            if row:
                return RoutingPreferences(
                    cost_quality_bias=row[0] or config.DEFAULT_COST_QUALITY_BIAS,
                    speed_quality_bias=row[1] or config.DEFAULT_SPEED_QUALITY_BIAS,
                    cascade_enabled=row[2] if row[2] is not None else config.DEFAULT_CASCADE_ENABLED,
                    max_cascade_attempts=row[3] or 3,
                )
    
    # Return defaults
    return RoutingPreferences(
        cost_quality_bias=config.DEFAULT_COST_QUALITY_BIAS,
        speed_quality_bias=config.DEFAULT_SPEED_QUALITY_BIAS,
        cascade_enabled=config.DEFAULT_CASCADE_ENABLED,
        max_cascade_attempts=3,
    )


@router.put("/preferences")
async def update_preferences(
    request: Request,
    prefs: RoutingPreferences,
    project_id: Optional[str] = None,
    user: dict = Depends(require_auth),
):
    """Update routing preferences."""
    from gateway.db import get_session
    from sqlalchemy import text
    
    if project_id:
        async with get_session() as session:
            result = await session.execute(
                text("""
                    UPDATE projects
                    SET cost_quality_bias = :cost_quality_bias,
                        speed_quality_bias = :speed_quality_bias,
                        cascade_enabled = :cascade_enabled,
                        max_cascade_attempts = :max_cascade_attempts
                    WHERE api_key = :project_id
                    RETURNING id
                """),
                {
                    "project_id": project_id,
                    "cost_quality_bias": prefs.cost_quality_bias,
                    "speed_quality_bias": prefs.speed_quality_bias,
                    "cascade_enabled": prefs.cascade_enabled,
                    "max_cascade_attempts": prefs.max_cascade_attempts,
                },
            )
            
            if not result.scalar():
                raise HTTPException(status_code=404, detail="Project not found")
    
    log.info("Preferences updated for project %s: cost_bias=%.2f, speed_bias=%.2f",
             project_id, prefs.cost_quality_bias, prefs.speed_quality_bias)
    
    return {"success": True, "preferences": prefs}


# =============================================================================
# Model Management Endpoints
# =============================================================================

@router.get("/models")
async def get_models(
    request: Request,
    project_id: Optional[str] = None,
    user: dict = Depends(require_auth),
) -> List[ModelInfo]:
    """Get all available models with their settings."""
    from gateway.model_registry import get_model_registry
    
    registry = get_model_registry()
    models = registry.get_all_models()
    
    # Get project-specific settings if project_id provided
    custom_settings = {}
    if project_id:
        from gateway.db import get_session
        from sqlalchemy import text
        
        async with get_session() as session:
            result = await session.execute(
                text("""
                    SELECT ms.model_id, ms.is_enabled, ms.custom_quality_rating
                    FROM model_settings ms
                    JOIN projects p ON ms.project_id = p.id
                    WHERE p.api_key = :project_id
                """),
                {"project_id": project_id},
            )
            
            for row in result.fetchall():
                custom_settings[row[0]] = {
                    "is_enabled": row[1],
                    "custom_quality_rating": row[2],
                }
    
    result = []
    for model in models:
        custom = custom_settings.get(model.id, {})
        result.append(ModelInfo(
            id=model.id,
            provider=model.provider,
            display_name=model.id.replace("-", " ").title(),
            is_enabled=custom.get("is_enabled", True),
            quality_rating=custom.get("custom_quality_rating") or model.quality_rating,
            cost_per_1k_input=model.cost_per_1k_input,
            cost_per_1k_output=model.cost_per_1k_output,
            latency_ms=model.latency_ms_first_token,
            context_window=model.context_window,
            capabilities=model.capabilities,
        ))
    
    return result


@router.put("/models/{model_id}/enabled")
async def set_model_enabled(
    request: Request,
    model_id: str,
    settings: ModelSettingsUpdate,
    project_id: str,
    user: dict = Depends(require_auth),
):
    """Enable or disable a model for a project."""
    from gateway.db import get_session
    from sqlalchemy import text
    
    async with get_session() as session:
        # Get project ID
        result = await session.execute(
            text("SELECT id FROM projects WHERE api_key = :api_key"),
            {"api_key": project_id},
        )
        project_row = result.fetchone()
        
        if not project_row:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_db_id = project_row[0]
        
        # Upsert model settings
        await session.execute(
            text("""
                INSERT INTO model_settings (project_id, model_id, is_enabled, custom_quality_rating)
                VALUES (:project_id, :model_id, :is_enabled, :custom_quality_rating)
                ON CONFLICT (project_id, model_id) 
                DO UPDATE SET 
                    is_enabled = :is_enabled,
                    custom_quality_rating = :custom_quality_rating
            """),
            {
                "project_id": project_db_id,
                "model_id": model_id,
                "is_enabled": settings.is_enabled,
                "custom_quality_rating": settings.custom_quality_rating,
            },
        )
    
    log.info("Model %s %s for project %s", 
             model_id, "enabled" if settings.is_enabled else "disabled", project_id)
    
    return {"success": True, "model_id": model_id, "is_enabled": settings.is_enabled}


# =============================================================================
# Ollama Management Endpoints
# =============================================================================

@router.get("/ollama/models")
async def get_ollama_models(
    request: Request,
    user: dict = Depends(require_auth),
) -> List[OllamaModelInfo]:
    """Get list of locally available Ollama models."""
    from gateway.providers.ollama_provider import OllamaProvider
    
    provider = OllamaProvider()
    models = await provider.discover_models()
    
    return models


@router.post("/ollama/pull")
async def pull_ollama_model(
    request: Request,
    body: OllamaPullRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_auth),
):
    """
    Start pulling an Ollama model.
    Returns immediately with a job ID to track progress.
    """
    from gateway.db import get_session
    from sqlalchemy import text
    
    # Create pull job record
    async with get_session() as session:
        result = await session.execute(
            text("""
                INSERT INTO ollama_pull_jobs (model_name, status, started_by)
                VALUES (:model_name, 'pending', :user_id)
                RETURNING id
            """),
            {"model_name": body.model_name, "user_id": user["id"]},
        )
        job_id = result.scalar()
    
    # Start background pull
    background_tasks.add_task(_pull_model_task, job_id, body.model_name)
    
    log.info("Started Ollama pull job %d for model %s", job_id, body.model_name)
    
    return {"success": True, "job_id": job_id, "model_name": body.model_name}


async def _pull_model_task(job_id: int, model_name: str):
    """Background task to pull Ollama model."""
    from gateway.db import get_session
    from gateway.providers.ollama_provider import OllamaProvider
    from sqlalchemy import text
    
    try:
        # Update status to pulling
        async with get_session() as session:
            await session.execute(
                text("UPDATE ollama_pull_jobs SET status = 'pulling' WHERE id = :id"),
                {"id": job_id},
            )
        
        # Pull model
        provider = OllamaProvider()
        success = await provider.pull_model(model_name)
        
        # Update final status
        async with get_session() as session:
            await session.execute(
                text("""
                    UPDATE ollama_pull_jobs 
                    SET status = :status, 
                        completed_at = NOW(),
                        progress = 1.0
                    WHERE id = :id
                """),
                {"id": job_id, "status": "completed" if success else "failed"},
            )
        
        log.info("Ollama pull job %d completed: %s", job_id, "success" if success else "failed")
        
    except Exception as e:
        log.error("Ollama pull job %d failed: %s", job_id, e)
        
        async with get_session() as session:
            await session.execute(
                text("""
                    UPDATE ollama_pull_jobs 
                    SET status = 'failed', 
                        error = :error,
                        completed_at = NOW()
                    WHERE id = :id
                """),
                {"id": job_id, "error": str(e)},
            )


@router.get("/ollama/pull/{job_id}")
async def get_pull_status(
    request: Request,
    job_id: int,
    user: dict = Depends(require_auth),
) -> PullJobStatus:
    """Get status of an Ollama pull job."""
    from gateway.db import get_session
    from sqlalchemy import text
    
    async with get_session() as session:
        result = await session.execute(
            text("""
                SELECT id, model_name, status, progress, error, created_at
                FROM ollama_pull_jobs
                WHERE id = :id
            """),
            {"id": job_id},
        )
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Pull job not found")
        
        return PullJobStatus(
            id=row[0],
            model_name=row[1],
            status=row[2],
            progress=row[3],
            error=row[4],
            created_at=row[5],
        )


@router.delete("/ollama/models/{model_name:path}")
async def delete_ollama_model(
    request: Request,
    model_name: str,
    user: dict = Depends(require_auth),
):
    """Delete an Ollama model."""
    from gateway.providers.ollama_provider import OllamaProvider
    
    provider = OllamaProvider()
    success = await provider.delete_model(model_name)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete model")
    
    log.info("Deleted Ollama model: %s", model_name)
    
    return {"success": True, "model_name": model_name}


# =============================================================================
# Statistics Endpoints
# =============================================================================

@router.get("/stats")
async def get_stats(
    request: Request,
    project_id: Optional[str] = None,
    days: int = 7,
    user: dict = Depends(require_auth),
) -> UsageStats:
    """Get usage statistics."""
    from gateway.db import get_session
    from sqlalchemy import text
    
    since = datetime.utcnow() - timedelta(days=days)
    
    async with get_session() as session:
        # Get request counts by model
        result = await session.execute(
            text("""
                SELECT model, COUNT(*) as count, 
                       SUM(prompt_tokens + completion_tokens) as tokens,
                       SUM(COALESCE(estimated_cost, 0)) as cost
                FROM metrics
                WHERE created_at > :since
                AND (:project_id IS NULL OR project_id = (
                    SELECT id FROM projects WHERE api_key = :project_id
                ))
                GROUP BY model
            """),
            {"since": since, "project_id": project_id},
        )
        
        requests_by_model = {}
        total_requests = 0
        total_tokens = 0
        total_cost = 0.0
        
        for row in result.fetchall():
            requests_by_model[row[0] or "unknown"] = row[1]
            total_requests += row[1]
            total_tokens += row[2] or 0
            total_cost += float(row[3] or 0)
        
        # Group by provider
        requests_by_provider = {}
        for model, count in requests_by_model.items():
            provider = _extract_provider(model)
            requests_by_provider[provider] = requests_by_provider.get(provider, 0) + count
        
        # Get cache hits
        cache_result = await session.execute(
            text("""
                SELECT 
                    COUNT(*) FILTER (WHERE cache_hit = true) as hits,
                    COUNT(*) as total
                FROM metrics
                WHERE created_at > :since
                AND (:project_id IS NULL OR project_id = (
                    SELECT id FROM projects WHERE api_key = :project_id
                ))
            """),
            {"since": since, "project_id": project_id},
        )
        cache_row = cache_result.fetchone()
        cache_hits = cache_row[0] or 0
        total_cacheable = cache_row[1] or 1  # Avoid division by zero
        
        return UsageStats(
            total_requests=total_requests,
            total_tokens=total_tokens,
            estimated_cost=round(total_cost, 4),
            cache_hits=cache_hits,
            cache_hit_rate=round(cache_hits / total_cacheable, 4) if total_cacheable > 0 else 0.0,
            requests_by_model=requests_by_model,
            requests_by_provider=requests_by_provider,
        )


def _extract_provider(model: str) -> str:
    """Extract provider name from model ID."""
    model_lower = model.lower()
    
    if "claude" in model_lower:
        return "anthropic"
    elif "gpt" in model_lower or "o1" in model_lower:
        return "openai"
    elif "gemini" in model_lower:
        return "gemini"
    elif "llama" in model_lower or "mixtral" in model_lower:
        return "groq"
    elif ":" in model_lower:  # Ollama models often have format "model:tag"
        return "ollama"
    else:
        return "unknown"


# =============================================================================
# Health Check for Providers
# =============================================================================

@router.get("/providers/status")
async def get_provider_status(
    request: Request,
    user: dict = Depends(require_auth),
):
    """Check status of all configured providers."""
    from gateway.providers.registry import get_provider_registry
    
    registry = get_provider_registry()
    providers = registry.get_available_providers()
    
    status = {}
    for name, provider in providers.items():
        try:
            # Simple health check
            is_available = await provider.is_available()
            status[name] = {
                "available": is_available,
                "models": list(provider.get_supported_models()) if hasattr(provider, 'get_supported_models') else [],
            }
        except Exception as e:
            status[name] = {
                "available": False,
                "error": str(e),
            }
    
    return {"providers": status}
