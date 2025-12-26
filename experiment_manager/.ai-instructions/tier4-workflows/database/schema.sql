-- EDS v1.0 Database Schema
-- SQLite schema for experiment metadata indexing
-- Source of truth: Markdown files
-- Database role: Secondary index for fast queries

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK(status IN ('active', 'complete', 'deprecated')) NOT NULL,
    created_at TEXT NOT NULL,  -- ISO 8601
    updated_at TEXT NOT NULL,  -- ISO 8601
    ads_version TEXT NOT NULL,
    UNIQUE(experiment_id)
);

-- Artifacts table
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    type TEXT CHECK(type IN ('assessment', 'report', 'guide', 'script', 'experiment_manifest')) NOT NULL,
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT CHECK(status IN ('active', 'complete', 'deprecated')) NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    UNIQUE(file_path)
);

-- Type-specific metadata (JSON for flexibility)
CREATE TABLE IF NOT EXISTS artifact_metadata (
    artifact_id TEXT PRIMARY KEY,
    phase TEXT CHECK(phase IN ('planning', 'execution', 'analysis', 'complete')),
    priority TEXT CHECK(priority IN ('low', 'medium', 'high', 'critical')),
    evidence_count INTEGER,
    metrics TEXT,  -- JSON array
    baseline TEXT,
    comparison TEXT CHECK(comparison IN ('baseline', 'previous', 'best')),
    commands TEXT,  -- JSON array
    prerequisites TEXT,  -- JSON array
    dependencies TEXT,  -- JSON array
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE
);

-- Tags (many-to-many)
CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    tag_name TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    UNIQUE(experiment_id, tag_name)
);

-- Artifact tags (many-to-many)
CREATE TABLE IF NOT EXISTS artifact_tags (
    artifact_id TEXT NOT NULL,
    tag_name TEXT NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    PRIMARY KEY (artifact_id, tag_name)
);

-- Metrics table
CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    baseline_value REAL,
    timestamp TEXT NOT NULL,  -- ISO 8601
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

-- Full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
    artifact_id UNINDEXED,
    experiment_id UNINDEXED,
    title,
    content,
    tokenize='porter'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_updated ON experiments(updated_at);

CREATE INDEX IF NOT EXISTS idx_artifacts_experiment ON artifacts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(type);
CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(status);
CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_updated ON artifacts(updated_at);

CREATE INDEX IF NOT EXISTS idx_tags_experiment ON tags(experiment_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(tag_name);

CREATE INDEX IF NOT EXISTS idx_artifact_tags_tag ON artifact_tags(tag_name);

CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_artifact ON metrics(artifact_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

-- Views for common queries

-- Active experiments with artifact counts
CREATE VIEW IF NOT EXISTS v_active_experiments AS
SELECT
    e.experiment_id,
    e.name,
    e.description,
    e.created_at,
    e.updated_at,
    COUNT(DISTINCT a.artifact_id) as artifact_count,
    COUNT(DISTINCT CASE WHEN a.type = 'assessment' THEN a.artifact_id END) as assessment_count,
    COUNT(DISTINCT CASE WHEN a.type = 'report' THEN a.artifact_id END) as report_count,
    COUNT(DISTINCT CASE WHEN a.type = 'guide' THEN a.artifact_id END) as guide_count,
    COUNT(DISTINCT CASE WHEN a.type = 'script' THEN a.artifact_id END) as script_count
FROM experiments e
LEFT JOIN artifacts a ON e.experiment_id = a.experiment_id
WHERE e.status = 'active'
GROUP BY e.experiment_id;

-- Recent activity
CREATE VIEW IF NOT EXISTS v_recent_activity AS
SELECT
    e.experiment_id,
    e.name,
    a.artifact_id,
    a.title,
    a.type,
    a.updated_at
FROM experiments e
JOIN artifacts a ON e.experiment_id = a.experiment_id
ORDER BY a.updated_at DESC
LIMIT 50;

-- Experiment statistics
CREATE VIEW IF NOT EXISTS v_experiment_stats AS
SELECT
    COUNT(*) as total_experiments,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count,
    COUNT(CASE WHEN status = 'complete' THEN 1 END) as complete_count,
    COUNT(CASE WHEN status = 'deprecated' THEN 1 END) as deprecated_count
FROM experiments;

-- Artifact statistics
CREATE VIEW IF NOT EXISTS v_artifact_stats AS
SELECT
    e.experiment_id,
    e.name,
    COUNT(a.artifact_id) as total_artifacts,
    AVG(CASE WHEN a.type = 'assessment' THEN am.evidence_count END) as avg_evidence_count,
    COUNT(DISTINCT at.tag_name) as tag_count
FROM experiments e
LEFT JOIN artifacts a ON e.experiment_id = a.experiment_id
LEFT JOIN artifact_metadata am ON a.artifact_id = am.artifact_id
LEFT JOIN artifact_tags at ON a.artifact_id = at.artifact_id
GROUP BY e.experiment_id;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT
);

INSERT INTO schema_version (version, applied_at, description)
VALUES (1, datetime('now'), 'EDS v1.0 initial schema');
