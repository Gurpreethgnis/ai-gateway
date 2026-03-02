-- Migration 005: Webhook integrations table
-- Stores per-project webhook endpoints. project_id=NULL means global (all projects).
-- events column is a comma-separated list of event globs, e.g. "request.complete,error.upstream"
-- Use events='*' to receive all events.

CREATE TABLE IF NOT EXISTS webhooks (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    url VARCHAR(2048) NOT NULL,
    secret VARCHAR(255),
    events TEXT NOT NULL DEFAULT '*',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_webhooks_project
    ON webhooks(project_id, is_active);
