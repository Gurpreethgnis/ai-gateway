import os
import sys
import uvicorn

if __name__ == "__main__":
    # Fail fast with a clear message if required env is missing (before loading app).
    # This makes Railway logs show the cause instead of a generic import error.
    if not os.environ.get("GATEWAY_API_KEY"):
        print("FATAL: GATEWAY_API_KEY is not set. Set it in Railway Variables.", file=sys.stderr)
        sys.exit(1)

    port = int(os.environ.get("PORT", "8080"))
    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    print(f"Gateway starting PORT={port} WORKERS={workers}", file=sys.stderr, flush=True)

    # Pre-import app so any import error is visible in Railway logs (traceback to stderr).
    try:
        from app import app  # noqa: F401 - used to validate import
    except Exception:
        import traceback
        print("FATAL: Failed to load app:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    print(f"Starting uvicorn on 0.0.0.0:{port}", file=sys.stderr, flush=True)
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=port,
            workers=workers,
            timeout_keep_alive=120,
        )
    except Exception as e:
        import traceback
        print("FATAL: uvicorn failed:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
