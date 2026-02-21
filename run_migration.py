#!/usr/bin/env python3
"""
Run database migration to add full_content column to file_hash_entries.
Usage: python run_migration.py
"""
import asyncio
import os
import sys

# Get DATABASE_URL directly without importing config (avoids GATEWAY_API_KEY requirement)
DATABASE_URL = os.getenv("DATABASE_URL")

async def run_migration():
    if not DATABASE_URL:
        print("[ERROR] DATABASE_URL not set.")
        print("Run: export DATABASE_URL='your_database_url'")
        return False
    
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        
        # Convert postgres:// to postgresql:// for asyncpg
        db_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+asyncpg://")
        
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
    print("=" * 60)
    print("Database Migration: Add full_content to file_hash_entries")
    print("=" * 60)
    success = asyncio.run(run_migration())
    sys.exit(0 if success else 1)
