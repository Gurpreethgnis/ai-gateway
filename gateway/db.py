import os
import hashlib
from datetime import datetime
from typing import Optional, List, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import String, Text, Integer, Float, DateTime, ForeignKey, Index, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from gateway.config import DATABASE_URL

class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    rate_limit_rpm: Mapped[int] = mapped_column(Integer, default=60)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(default=True)

    usage_records: Mapped[List["UsageRecord"]] = relationship(back_populates="project")
    file_hashes: Mapped[List["FileHashEntry"]] = relationship(back_populates="project")
    embedding_chunks: Mapped[List["EmbeddingChunk"]] = relationship(back_populates="project")
    plugin_tools: Mapped[List["PluginTool"]] = relationship(back_populates="project")
    repo_nodes: Mapped[List["RepoNode"]] = relationship(back_populates="project")


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("projects.id"), nullable=True)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, index=True)
    cache_read_input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None, index=True)
    cache_creation_input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    gateway_tokens_saved: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None, index=True)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    cached: Mapped[bool] = mapped_column(default=False)
    cf_ray: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    project: Mapped[Optional["Project"]] = relationship(back_populates="usage_records")

    __table_args__ = (
        Index("ix_usage_project_timestamp", "project_id", "timestamp"),
    )


class FileHashEntry(Base):
    __tablename__ = "file_hash_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content_preview: Mapped[str] = mapped_column(Text, default="")
    # full_content stored via raw SQL when column exists (see file_cache.py); run migration to add it
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped["Project"] = relationship(back_populates="file_hashes")

    __table_args__ = (
        Index("ix_filehash_project_hash", "project_id", "content_hash"),
    )


class EmbeddingChunk(Base):
    __tablename__ = "embedding_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_vector: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped["Project"] = relationship(back_populates="embedding_chunks")

    __table_args__ = (
        Index("ix_embedding_project_hash", "project_id", "content_hash"),
    )


class PluginTool(Base):
    __tablename__ = "plugin_tools"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    input_schema_json: Mapped[str] = mapped_column(Text, default="{}")
    endpoint_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped["Project"] = relationship(back_populates="plugin_tools")

    __table_args__ = (
        Index("ix_plugin_project_name", "project_id", "name", unique=True),
    )


class RepoNode(Base):
    __tablename__ = "repo_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    node_type: Mapped[str] = mapped_column(String(16), default="file")
    symbols_json: Mapped[str] = mapped_column(Text, default="[]")
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped["Project"] = relationship(back_populates="repo_nodes")

    __table_args__ = (
        Index("ix_reponode_project_path", "project_id", "file_path", unique=True),
    )


class ModelSuccessRate(Base):
    __tablename__ = "model_success_rates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_successrate_project_model", "project_id", "model", unique=True),
    )


class RoutingOutcome(Base):
    __tablename__ = "routing_outcomes"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("projects.id"), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # Request fingerprint
    query_hash: Mapped[str] = mapped_column(String(64), index=True)
    query_embedding: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    query_length: Mapped[int] = mapped_column(Integer, default=0)
    has_tools: Mapped[bool] = mapped_column(default=False)
    has_tool_results: Mapped[bool] = mapped_column(default=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Routing decision
    initial_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    final_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    escalated: Mapped[bool] = mapped_column(default=False)
    escalation_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    
    # Outcome signals
    response_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(default=True)
    
    __table_args__ = (
        Index("ix_routing_project_timestamp", "project_id", "timestamp"),
        Index("ix_routing_query_hash", "query_hash"),
    )


engine = None
async_session_factory = None


def init_db():
    global engine, async_session_factory
    if not DATABASE_URL:
        return
    
    db_url = DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    engine = create_async_engine(
        db_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=5,
        pool_timeout=10,
        pool_recycle=300,
        connect_args={
            "server_settings": {
                "statement_timeout": "5000"
            }
        }
    )
    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def create_tables():
    if engine is None:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Allow usage records without a project (e.g. single-tenant gateway) so dashboard still shows usage
        try:
            await conn.execute(text("ALTER TABLE usage_records ALTER COLUMN project_id DROP NOT NULL"))
        except Exception:
            pass  # column may already be nullable or table just created with new schema
        
        # Add missing columns if they don't exist
        try:
            await conn.execute(text("ALTER TABLE usage_records ADD COLUMN cache_read_input_tokens INTEGER"))
        except Exception:
            pass  # column may already exist
        try:
            await conn.execute(text("ALTER TABLE usage_records ADD COLUMN cache_creation_input_tokens INTEGER"))
        except Exception:
            pass  # column may already exist
        try:
            await conn.execute(text("ALTER TABLE usage_records ADD COLUMN gateway_tokens_saved INTEGER"))
        except Exception:
            pass  # column may already exist
            
        # Make cache-related columns nullable if they exist
        try:
            await conn.execute(text("ALTER TABLE usage_records ALTER COLUMN cache_read_input_tokens DROP NOT NULL"))
        except Exception:
            pass  # column may not exist or already nullable
        try:
            await conn.execute(text("ALTER TABLE usage_records ALTER COLUMN cache_creation_input_tokens DROP NOT NULL"))
        except Exception:
            pass  # column may not exist or already nullable
        try:
            await conn.execute(text("ALTER TABLE usage_records ALTER COLUMN gateway_tokens_saved DROP NOT NULL"))
        except Exception:
            pass  # column may not exist or already nullable


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if async_session_factory is None:
        raise RuntimeError("Database not initialized. Set DATABASE_URL.")
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def _increment_dashboard_counters(
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    gateway_saved: int,
    cost_usd: float,
):
    """Atomically increment Redis counters for dashboard stats. Never blocks or raises."""
    try:
        from gateway.cache import rds
        if not rds:
            return
        today = datetime.utcnow().strftime("%Y-%m-%d")
        pipe = rds.pipeline()
        for key in [f"dashboard:stats:{today}", "dashboard:stats:all_time"]:
            pipe.hincrby(key, "input_tokens", input_tokens)
            pipe.hincrby(key, "output_tokens", output_tokens)
            pipe.hincrby(key, "cached_tokens", cache_read)
            pipe.hincrby(key, "gateway_saved", gateway_saved)
            pipe.hincrbyfloat(key, "cost_usd", cost_usd)
            pipe.hincrby(key, "request_count", 1)
        pipe.expire(f"dashboard:stats:{today}", 172800)
        pipe.execute()
    except Exception:
        pass


# Timeout for usage record DB write; prevents slow DB from holding connections.
RECORD_USAGE_DB_TIMEOUT = float(os.getenv("RECORD_USAGE_DB_TIMEOUT", "8"))


async def record_usage_to_db(
    project_id,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cf_ray: str,
    cached: bool,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    gateway_tokens_saved: int = 0,
):
    if not DATABASE_URL:
        return
    
    import asyncio
    import logging
    log = logging.getLogger("gateway")

    async def _do_record():
        cost = calculate_cost(model, input_tokens, output_tokens)
        _increment_dashboard_counters(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read_input_tokens,
            gateway_saved=gateway_tokens_saved,
            cost_usd=cost,
        )
        async with get_session() as session:
            record = UsageRecord(
                project_id=project_id,
                model=model,
                input_tokens=input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                gateway_tokens_saved=gateway_tokens_saved,
                output_tokens=output_tokens,
                cost_usd=cost,
                cached=cached,
                cf_ray=cf_ray,
            )
            session.add(record)
            await session.commit()

    try:
        await asyncio.wait_for(_do_record(), timeout=RECORD_USAGE_DB_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning("record_usage_to_db timed out after %.0fs - skipping", RECORD_USAGE_DB_TIMEOUT)
    except Exception as e:
        log.warning("record_usage_to_db failed: %r", e)


COST_PER_1K_TOKENS = {
    "claude-sonnet-4-0": {"input": 0.003, "output": 0.015},
    "claude-opus-4-5": {"input": 0.015, "output": 0.075},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1K_TOKENS.get(model, {"input": 0.003, "output": 0.015})
    return (input_tokens / 1000.0) * rates["input"] + (output_tokens / 1000.0) * rates["output"]
