import os
import hashlib
from datetime import datetime
from typing import Optional, List, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import String, Text, Integer, Float, DateTime, ForeignKey, Index, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from gateway.config import DATABASE_URL
import logging
log = logging.getLogger("gateway")

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
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_creation_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cached: Mapped[bool] = mapped_column(default=False)
    cf_ray: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    project: Mapped["Project"] = relationship(back_populates="usage_records")

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


engine = None
async_session_factory = None


def init_db():
    global engine, async_session_factory
    
    # Try common database URL names for various hosting providers
    db_url = DATABASE_URL or os.getenv("POSTGRES_URL") or os.getenv("DATABASE_PRIVATE_URL") or os.getenv("INTERNAL_DATABASE_URL")
    
    if not db_url:
        log.warning("DATABASE_URL is not set. Database features will be disabled.")
        return
    
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    log.info("Initializing database with URL: %s", db_url.split("@")[-1]) # Log only host for safety
    engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def create_tables():
    if engine is None:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def run_migrations():
    """Manual migration to add new columns to existing tables."""
    if engine is None:
        return
    
    from sqlalchemy import text
    async with engine.begin() as conn:
        # Check dialect for safety
        dialect = conn.dialect.name
        
        if dialect == "postgresql":
            try:
                await conn.execute(text("ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS cache_read_input_tokens INTEGER DEFAULT 0"))
                await conn.execute(text("ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS cache_creation_input_tokens INTEGER DEFAULT 0"))
                log.info("Postgres migrations applied successfully")
            except Exception as e:
                log.debug("Postgres migration error (might already exist): %r", e)
        else:
            # Generic approach for other DBs (SQLite etc)
            try:
                await conn.execute(text("ALTER TABLE usage_records ADD COLUMN cache_read_input_tokens INTEGER DEFAULT 0"))
                await conn.execute(text("ALTER TABLE usage_records ADD COLUMN cache_creation_input_tokens INTEGER DEFAULT 0"))
                log.info("Generic migrations applied successfully")
            except Exception as e:
                log.debug("Generic migration error: %r", e)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if async_session_factory is None:
        raise RuntimeError("Database not initialized. Set DATABASE_URL environment variable.")
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def record_usage_to_db(
    project_id: Optional[int],
    model: str,
    input_tokens: int,
    output_tokens: int,
    cf_ray: str,
    cached: bool,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
):
    cost = calculate_cost(model, input_tokens, output_tokens)
    async with get_session() as session:
        record = UsageRecord(
            project_id=project_id,
            model=model,
            input_tokens=input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            cached=cached,
            cf_ray=cf_ray,
        )
        session.add(record)


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


COST_PER_1K_TOKENS = {
    "claude-sonnet-4-0": {"input": 0.003, "output": 0.015},
    "claude-opus-4-5": {"input": 0.015, "output": 0.075},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1K_TOKENS.get(model, {"input": 0.003, "output": 0.015})
    return (input_tokens / 1000.0) * rates["input"] + (output_tokens / 1000.0) * rates["output"]
