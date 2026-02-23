#!/usr/bin/env python3
"""
Run database migrations.
Usage:
  python run_migration.py
    Runs built-in migration (full_content column).
  python run_migration.py migrations/002_add_auth_and_routing.sql
    Runs the given SQL file (e.g. with Railway: railway run python run_migration.py migrations/002_add_auth_and_routing.sql)
"""
import asyncio
import os
import sys

# Get DATABASE_URL directly without importing config (avoids GATEWAY_API_KEY requirement)
DATABASE_URL = os.getenv("DATABASE_URL")


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _statements_from_file(path: str):
    """Split SQL file into statements (by ;), drop comments-only and empty."""
    raw = open(path, "r", encoding="utf-8").read()
    statements = []
    for s in raw.split(";"):
        s = s.strip()
        if not s:
            continue
        # Skip comment-only blocks
        lines = [l.strip() for l in s.splitlines() if l.strip()]
        if all(l.startswith("--") for l in lines):
            continue
        statements.append(s + ";")
    return statements


async def run_migration_file(path: str) -> bool:
    if not DATABASE_URL:
        print("[SKIP] DATABASE_URL not set. Skipping migration.")
        return True  # Success so deploy can proceed without DB
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        return False
    statements = _statements_from_file(path)
    if not statements:
        print("[WARN] No statements found.")
        return True
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        db_url = _to_asyncpg_url(DATABASE_URL)
        engine = create_async_engine(db_url, echo=True)
        print(f"\n[INFO] Running migration: {path} ({len(statements)} statement(s))")
        async with engine.begin() as conn:
            for i, stmt in enumerate(statements):
                await conn.execute(text(stmt))
                print(f"  [{i+1}/{len(statements)}] OK")
        await engine.dispose()
        print("[SUCCESS] Migration complete!")
        return True
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_builtin_migration():
    if not DATABASE_URL:
        print("[SKIP] DATABASE_URL not set. Skipping migration.")
        return True  # Success so deploy can proceed without DB
    
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        
        db_url = _to_asyncpg_url(DATABASE_URL)
        engine = create_async_engine(db_url, echo=True)
        
        migration_sql = """
        ALTER TABLE file_hash_entries 
        ADD COLUMN IF NOT EXISTS full_content TEXT NULL;
        """
        
        print("\n[INFO] Running migration: Add full_content column to file_hash_entries...")
        
        async with engine.begin() as conn:
            await conn.execute(text(migration_sql))
            print("[SUCCESS] Migration complete!")
        
        await engine.dispose()
        return True
        
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print("=" * 60)
        print(f"Database Migration: {path}")
        print("=" * 60)
        success = asyncio.run(run_migration_file(path))
    else:
        print("=" * 60)
        print("Database Migration: Add full_content to file_hash_entries")
        print("=" * 60)
        success = asyncio.run(run_builtin_migration())
    sys.exit(0 if success else 1)
