-- Migration: Add users, sessions, model settings, and routing preferences tables
-- Run with: python run_migration.py migrations/002_add_auth_and_routing.sql

-- Users table for dashboard authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',  -- 'admin', 'user'
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Sessions table for session-based auth
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    ip_address VARCHAR(45),
    user_agent TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

-- Model settings (enable/disable, quality overrides)
CREATE TABLE IF NOT EXISTS model_settings (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    model_id VARCHAR(100) NOT NULL,
    is_enabled BOOLEAN DEFAULT true,
    custom_quality_rating FLOAT,
    priority INTEGER DEFAULT 100,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_id, model_id)
);

-- Global model settings (project_id = NULL)
CREATE INDEX IF NOT EXISTS idx_model_settings_global ON model_settings(model_id) WHERE project_id IS NULL;

-- Routing preferences per project
ALTER TABLE projects ADD COLUMN IF NOT EXISTS cost_quality_bias FLOAT DEFAULT 0.5;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS speed_quality_bias FLOAT DEFAULT 0.5;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS cascade_enabled BOOLEAN DEFAULT true;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS max_cascade_attempts INTEGER DEFAULT 2;

-- Ollama model pull jobs (for async downloads)
CREATE TABLE IF NOT EXISTS ollama_pull_jobs (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',  -- 'pending', 'pulling', 'complete', 'error'
    progress_percent INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create default admin user if not exists (password: changeme)
-- Password hash is bcrypt of 'changeme'
INSERT INTO users (email, password_hash, display_name, role)
VALUES (
    'admin@localhost',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.rLIPXTlFJAAhAa',
    'Admin',
    'admin'
) ON CONFLICT (email) DO NOTHING;
