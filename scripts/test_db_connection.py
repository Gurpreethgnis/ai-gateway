"""Test database connection directly to diagnose Railway Postgres issues."""
import asyncio
import os
import time
import sys

async def test_connection():
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    
    # Mask password for logging
    safe_url = db_url.split("@")[-1] if "@" in db_url else db_url
    print(f"Target: {safe_url}")
    
    # Test 1: Raw DNS resolution
    host = safe_url.split(":")[0] if ":" in safe_url else safe_url.split("/")[0]
    print(f"\n--- Test 1: DNS resolution for '{host}' ---")
    import socket
    t0 = time.time()
    try:
        ips = socket.getaddrinfo(host, 5432)
        print(f"Resolved in {time.time()-t0:.2f}s -> {ips[0][4][0]}")
    except Exception as e:
        print(f"DNS FAILED after {time.time()-t0:.2f}s: {e}")
        print("This means the Postgres service is not reachable from this environment.")
        sys.exit(1)
    
    # Test 2: Raw TCP connection
    print(f"\n--- Test 2: TCP connect to {host}:5432 ---")
    t0 = time.time()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, 5432),
            timeout=10
        )
        print(f"TCP connected in {time.time()-t0:.2f}s")
        writer.close()
        await writer.wait_closed()
    except asyncio.TimeoutError:
        print(f"TCP TIMEOUT after {time.time()-t0:.2f}s")
        print("Port 5432 is not accepting connections.")
        sys.exit(1)
    except Exception as e:
        print(f"TCP FAILED after {time.time()-t0:.2f}s: {type(e).__name__}: {e}")
        sys.exit(1)
    
    # Test 3: asyncpg direct connection
    print(f"\n--- Test 3: asyncpg connect ---")
    try:
        import asyncpg
    except ImportError:
        print("asyncpg not installed, skipping")
        return
    
    t0 = time.time()
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(db_url, timeout=10),
            timeout=15
        )
        elapsed = time.time() - t0
        print(f"Connected in {elapsed:.2f}s")
        
        ver = await conn.fetchval("SELECT version()")
        print(f"Postgres version: {ver[:80]}")
        
        # Check if our tables exist
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        print(f"Tables: {[t['tablename'] for t in tables]}")
        
        await conn.close()
        print("\nDB connection is WORKING. The issue is in SQLAlchemy/engine config.")
        
    except asyncio.TimeoutError:
        print(f"TIMEOUT after {time.time()-t0:.2f}s")
        print("asyncpg can't complete the handshake. Check auth/SSL settings.")
    except Exception as e:
        print(f"FAILED after {time.time()-t0:.2f}s: {type(e).__name__}: {e}")
    
    # Test 4: SQLAlchemy engine (same as our app uses)
    print(f"\n--- Test 4: SQLAlchemy async engine ---")
    from sqlalchemy.ext.asyncio import create_async_engine
    
    sa_url = db_url
    if sa_url.startswith("postgres://"):
        sa_url = sa_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif sa_url.startswith("postgresql://"):
        sa_url = sa_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    engine = create_async_engine(
        sa_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=2,
        pool_timeout=15,
        connect_args={
            "command_timeout": 10,
            "server_settings": {"statement_timeout": "5000"}
        }
    )
    
    t0 = time.time()
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
            print(f"SQLAlchemy connected in {time.time()-t0:.2f}s, SELECT 1 = {result.scalar()}")
    except Exception as e:
        print(f"SQLAlchemy FAILED after {time.time()-t0:.2f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_connection())
