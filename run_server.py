import os
import sys
import uvicorn

if __name__ == "__main__":
    # Fail fast with a clear message if required env is missing (before loading app).
    # This makes Railway logs show the cause instead of a generic import error.
    if not os.environ.get("GATEWAY_API_KEY"):
        print("FATAL: GATEWAY_API_KEY is not set. Set it in Railway Variables.", file=sys.stderr)
        sys.exit(1)

    port = int(os.environ.get("PORT", 8000))
    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        timeout_keep_alive=120,
    )
