-- Migration 003: Reconcile ollama_pull_jobs schema
-- The SQL migration 002 created the table with different column names
-- (progress_percent, error_message) than what the app code expects
-- (progress, error, completed_at). This migration adds the missing columns.
--
-- Safe to run multiple times (uses ADD COLUMN IF NOT EXISTS).

ALTER TABLE ollama_pull_jobs
    ADD COLUMN IF NOT EXISTS progress FLOAT,
    ADD COLUMN IF NOT EXISTS error TEXT,
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP;
