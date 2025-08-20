-- Initial schema for OpenClio
-- Migration: 001_initial_schema
-- Date: 2024-01-01

-- Analysis sessions table
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT CHECK(status IN ('pending', 'running', 'paused', 'completed', 'failed')) DEFAULT 'pending',
    config JSON NOT NULL,
    current_batch INTEGER DEFAULT 0,
    total_batches INTEGER,
    total_conversations INTEGER,
    processed_conversations INTEGER DEFAULT 0,
    error_message TEXT,
    results JSON,
    metadata JSON
);

-- Batch progress tracking
CREATE TABLE IF NOT EXISTS batch_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
    batch_number INTEGER NOT NULL,
    status TEXT CHECK(status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    input_data JSON,
    output_data JSON,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    UNIQUE(session_id, batch_number)
);

-- Analysis templates for reusable configurations
CREATE TABLE IF NOT EXISTS analysis_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    category TEXT CHECK(category IN ('general', 'support', 'sales', 'medical', 'education', 'custom')),
    is_public BOOLEAN DEFAULT false,
    config JSON NOT NULL,
    custom_prompts JSON,
    facet_definitions JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT
);

-- Projects for organizing analyses
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings JSON,
    tags JSON,
    is_archived BOOLEAN DEFAULT false
);

-- Link analyses to projects
CREATE TABLE IF NOT EXISTS project_analyses (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
    name TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags JSON,
    UNIQUE(project_id, session_id)
);

-- Provider configurations for multi-provider support
CREATE TABLE IF NOT EXISTS provider_configs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    provider_type TEXT CHECK(provider_type IN ('openai', 'ollama', 'vllm', 'huggingface', 'alternative')),
    config JSON NOT NULL,
    priority INTEGER DEFAULT 100,
    is_active BOOLEAN DEFAULT true,
    is_fallback BOOLEAN DEFAULT false,
    rate_limit JSON,
    cost_per_token JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Track provider usage for cost optimization
CREATE TABLE IF NOT EXISTS provider_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL REFERENCES provider_configs(id),
    session_id TEXT REFERENCES analysis_sessions(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tokens_used INTEGER,
    cost REAL,
    success BOOLEAN,
    error_message TEXT,
    response_time_ms INTEGER
);

-- Partial results for progressive loading
CREATE TABLE IF NOT EXISTS partial_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
    batch_number INTEGER,
    result_type TEXT CHECK(result_type IN ('facet', 'cluster', 'insight', 'embedding')),
    data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cache for embeddings and LLM responses
CREATE TABLE IF NOT EXISTS response_cache (
    id TEXT PRIMARY KEY,
    cache_key TEXT NOT NULL UNIQUE,
    cache_type TEXT CHECK(cache_type IN ('embedding', 'llm', 'facet')),
    provider TEXT,
    input_hash TEXT NOT NULL,
    output JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    ttl_seconds INTEGER DEFAULT 86400
);

-- User preferences and settings
CREATE TABLE IF NOT EXISTS user_settings (
    id TEXT PRIMARY KEY DEFAULT 'default',
    preferences JSON,
    default_template_id TEXT REFERENCES analysis_templates(id),
    default_providers JSON,
    ui_theme TEXT DEFAULT 'light',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_status ON analysis_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON analysis_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_session ON batch_progress(session_id, batch_number);
CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_progress(status);
CREATE INDEX IF NOT EXISTS idx_projects_archived ON projects(is_archived);
CREATE INDEX IF NOT EXISTS idx_provider_active ON provider_configs(is_active, priority);
CREATE INDEX IF NOT EXISTS idx_cache_key ON response_cache(cache_key, cache_type);
CREATE INDEX IF NOT EXISTS idx_cache_accessed ON response_cache(accessed_at);
CREATE INDEX IF NOT EXISTS idx_partial_results_session ON partial_results(session_id, batch_number);

-- Triggers for updated_at timestamps
CREATE TRIGGER IF NOT EXISTS update_sessions_timestamp 
AFTER UPDATE ON analysis_sessions
BEGIN
    UPDATE analysis_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_templates_timestamp
AFTER UPDATE ON analysis_templates
BEGIN
    UPDATE analysis_templates SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_projects_timestamp
AFTER UPDATE ON projects
BEGIN
    UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_providers_timestamp
AFTER UPDATE ON provider_configs
BEGIN
    UPDATE provider_configs SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;