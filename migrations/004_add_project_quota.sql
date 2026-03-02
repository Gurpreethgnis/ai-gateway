-- Migration 004: Add per-project monthly spend quota
-- NULL means unlimited (default). Set a value to cap monthly USD spend.

ALTER TABLE projects
    ADD COLUMN IF NOT EXISTS monthly_spend_limit_usd FLOAT DEFAULT NULL;

-- Composite index speeds up the per-project monthly SUM(cost_usd) query
-- used by the quota check before each request.
CREATE INDEX IF NOT EXISTS ix_usage_project_cost
    ON usage_records(project_id, timestamp, cost_usd);
