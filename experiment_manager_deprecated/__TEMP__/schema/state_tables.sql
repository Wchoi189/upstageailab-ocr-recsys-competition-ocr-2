-- State Management Redesign: Database Schema
-- Created: 2025-12-19
-- Purpose: Ultra-concise state tracking with <1ms query performance
-- Ref: /home/vscode/.gemini/antigravity/brain/.../state_management_recommendations.md

-- Idempotent: safe to run multiple times
DROP TABLE IF EXISTS state_transitions;
DROP TABLE IF EXISTS experiment_insights;
DROP TABLE IF EXISTS experiment_decisions;
DROP TABLE IF EXISTS experiment_tasks;
DROP TABLE IF EXISTS experiment_state;

-- Core experiment state (replaces state.yml core fields)
CREATE TABLE experiment_state (
    experiment_id TEXT PRIMARY KEY,
    current_task_id TEXT,
    current_phase TEXT,
    status TEXT CHECK(status IN ('active', 'completed', 'failed', 'paused', 'deprecated')) NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    checkpoint_path TEXT,
    checkpoint_performance TEXT,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY(current_task_id) REFERENCES experiment_tasks(task_id) ON DELETE SET NULL
);

-- Task tracking (replaces state.yml tasks array)
CREATE TABLE experiment_tasks (
    task_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK(status IN ('backlog', 'in_progress', 'completed', 'deferred', 'blocked', 'failed')) NOT NULL DEFAULT 'backlog',
    priority TEXT CHECK(priority IN ('low', 'medium', 'high', 'critical')) NOT NULL DEFAULT 'medium',
    depends_on TEXT,  -- JSON array of task_ids
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    deferred_to_experiment TEXT,
    notes TEXT,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

-- Decision tracking (replaces state.yml decisions array)
CREATE TABLE experiment_decisions (
    decision_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    date TEXT NOT NULL,
    decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    impact TEXT,
    alternatives_considered TEXT,  -- JSON array of rejected options
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

-- Insight tracking (replaces state.yml insights array)
CREATE TABLE experiment_insights (
    insight_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    date TEXT NOT NULL,
    insight TEXT NOT NULL,
    impact TEXT NOT NULL,
    category TEXT CHECK(category IN ('observation', 'hypothesis', 'conclusion', 'optimization', 'issue')),
    related_task_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY(related_task_id) REFERENCES experiment_tasks(task_id) ON DELETE SET NULL
);

-- State transition audit log
CREATE TABLE state_transitions (
    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    transition_type TEXT CHECK(transition_type IN ('status', 'task', 'phase')) NOT NULL,
    triggered_by TEXT,  -- 'ai_agent', 'user', 'script', etc.
    timestamp TEXT NOT NULL,
    metadata TEXT,  -- JSON with additional context
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

-- Indexes for AI query optimization (from database_utilization_guidelines.md)

-- Pattern 1: Get current state (most common - <1ms target)
CREATE INDEX idx_state_status ON experiment_state(status);
CREATE INDEX idx_state_updated ON experiment_state(updated_at DESC);

-- Pattern 2: Get pending tasks (frequent - <5ms target)
CREATE INDEX idx_tasks_experiment ON experiment_tasks(experiment_id, status);
CREATE INDEX idx_tasks_priority ON experiment_tasks(priority DESC, created_at ASC);
CREATE INDEX idx_tasks_status ON experiment_tasks(status);

-- Pattern 3: Decision history retrieval (<10ms target)
CREATE INDEX idx_decisions_experiment ON experiment_decisions(experiment_id, date DESC);

-- Audit trail queries
CREATE INDEX idx_transitions_experiment ON state_transitions(experiment_id, timestamp DESC);

-- Note: idx_experiments_status already exists in database, not recreating

-- Success: 5 tables created, 7 new indexes for O(1) or O(log n) queries
