#!/bin/sh
# Railway injects PORT; if not set (e.g. custom target port), default so app and healthcheck use same port.
export PORT="${PORT:-8080}"
echo "Boot PORT=$PORT"
exec python run_server.py
