#!/usr/bin/env bash
# Check Railway project wiring (DB, services, vars). Run from repo root.
# Prereq: npx -y @railway/cli login  &&  npx -y @railway/cli link
set -e
RAILWAY="npx -y @railway/cli"

echo "=== Railway status (project / service link) ==="
$RAILWAY status 2>&1 || true

echo ""
echo "=== Variables for linked service (check DATABASE_URL) ==="
$RAILWAY variable list 2>&1 || true

echo ""
echo "=== Service deployment status ==="
$RAILWAY service status 2>&1 || true

echo ""
echo "=== Done. Look for: DATABASE_URL set (often from Postgres reference), app and Postgres in same project. ==="
