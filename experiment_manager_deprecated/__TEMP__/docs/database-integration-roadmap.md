---
ads_version: "1.0"
type: guide
artifact_id: "database-integration-roadmap"
status: "active"
created: "2025-12-17T19:00:00Z"
updated: "2025-12-17T19:00:00Z"
tags:
  - "database"
  - "sqlite"
  - "integration"
  - "roadmap"
  - "future-enhancement"
commands: []
prerequisites:
  - "EDS v1.0 implementation complete"
  - "SQLite database at data/ops/tracking.db"
  - "CLI tool (etk) operational"
---

# Database Integration Roadmap

## Overview

Integration plan for existing SQLite database (`data/ops/tracking.db`) with EDS v1.0 experiment tracking framework.

**Status**: Future enhancement (not required for core functionality)

**Database Location**: `data/ops/tracking.db`

**Current State**: Empty tables, no rows (forgotten/unused)

## Database Discovery

### Existing Schema

**Location**: `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/ops/tracking.db`

**Investigation Required**:
```bash
# Inspect existing schema
sqlite3 data/ops/tracking.db ".schema"

# List tables
sqlite3 data/ops/tracking.db ".tables"

# Show table structures
sqlite3 data/ops/tracking.db ".schema table_name"
```

**Expected Tables** (based on typical experiment tracking):
- `experiments` - Experiment metadata
- `runs` - Individual experiment runs
- `metrics` - Performance metrics
- `artifacts` - Artifact references
- `tags` - Tag associations

## Integration Benefits

### 1. Query Performance
- Fast querying across experiments
- Complex filtering and aggregation
- Historical trend analysis
- Multi-experiment comparisons

### 2. Structured Data
- Enforced schema validation
- Referential integrity
- Transaction support
- Concurrent access

### 3. Analytics & Reporting
- Time-series analysis
- Performance trends
- Experiment success rates
- Resource utilization tracking

### 4. Tool Integration
- Python analytics (pandas, plotly)
- SQL queries for ad-hoc analysis
- Dashboard integration (Grafana, Metabase)
- Export to data warehouses

## Proposed Architecture

### Hybrid Model: Files + Database

**Files (Primary)**:
- Artifact content (markdown)
- Detailed documentation
- Code snippets
- Human-readable format
- Git-tracked, reviewable

**Database (Secondary)**:
- Searchable metadata
- Performance metrics
- Quick queries
- Aggregations
- Analytics

### Data Flow

```
ETK CLI Tool
    │
    ├─→ Create/Update Markdown Files (.metadata/)
    │       │
    │       └─→ Git commit (pre-commit hooks validate)
    │
    └─→ Sync Metadata to SQLite (optional)
            │
            ├─→ experiments table (experiment_id, name, status, created, updated)
            ├─→ artifacts table (artifact_id, experiment_id, type, title, path, created)
            ├─→ metrics table (metric_id, experiment_id, name, value, timestamp)
            └─→ tags table (tag_id, experiment_id, tag_name)
```

## Implementation Phases

### Phase 1: Schema Design (2-4 hours)

**Tasks**:
1. Inspect existing `tracking.db` schema
2. Design/update schema to match EDS v1.0
3. Create migration scripts if schema exists
4. Document schema with relationships

**Tables to Create/Update**:

```sql
-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK(status IN ('active', 'complete', 'deprecated')),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    ads_version TEXT NOT NULL,
    UNIQUE(experiment_id)
);

-- Artifacts table
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    type TEXT CHECK(type IN ('assessment', 'report', 'guide', 'script', 'experiment_manifest')),
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT CHECK(status IN ('active', 'complete', 'deprecated')),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    UNIQUE(file_path)
);

-- Type-specific metadata (JSON column for flexibility)
CREATE TABLE IF NOT EXISTS artifact_metadata (
    artifact_id TEXT PRIMARY KEY,
    phase TEXT,
    priority TEXT,
    evidence_count INTEGER,
    metrics TEXT, -- JSON array
    baseline TEXT,
    comparison TEXT,
    commands TEXT, -- JSON array
    prerequisites TEXT, -- JSON array
    dependencies TEXT, -- JSON array
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE
);

-- Tags table (many-to-many)
CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    tag_name TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    UNIQUE(experiment_id, tag_name)
);

-- Metrics table (for report artifacts)
CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    baseline_value REAL,
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

-- Full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
    artifact_id UNINDEXED,
    title,
    content,
    content=artifacts,
    content_rowid=rowid
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_experiment ON artifacts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(type);
CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(status);
CREATE INDEX IF NOT EXISTS idx_tags_experiment ON tags(experiment_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(tag_name);
CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_artifact ON metrics(artifact_id);
```

### Phase 2: Sync Tool (4-6 hours)

**Tasks**:
1. Create `etk sync` command
2. Parse markdown frontmatter → SQLite
3. Bi-directional sync detection
4. Conflict resolution strategy

**Features**:
```bash
# Sync single experiment
etk sync 20251217_024343_image_enhancements

# Sync all experiments
etk sync --all

# Dry-run mode
etk sync --all --dry-run

# Force sync (overwrite DB)
etk sync --all --force

# Bidirectional check
etk sync --check-conflicts
```

**Implementation**:
```python
class DatabaseSync:
    def sync_experiment(self, experiment_id: str):
        """Sync experiment metadata from markdown to DB."""

        # Read README.md (experiment manifest)
        manifest = self.parse_frontmatter(exp_path / "README.md")

        # Insert/update experiments table
        self.db.execute("""
            INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (experiment_id, manifest['name'], ...))

        # Sync all artifacts
        for artifact_path in self.find_artifacts(experiment_id):
            self.sync_artifact(artifact_path)

    def sync_artifact(self, artifact_path: Path):
        """Sync artifact metadata to DB."""
        frontmatter = self.parse_frontmatter(artifact_path)

        # Insert/update artifacts table
        # Insert/update artifact_metadata
        # Insert/update tags
        # Extract metrics if report type
```

### Phase 3: Query Interface (3-5 hours)

**Tasks**:
1. Create `etk query` command
2. Predefined queries (most_active, recent, by_tag)
3. Custom SQL support
4. JSON/CSV export

**Features**:
```bash
# List active experiments
etk query experiments --status active

# Find experiments by tag
etk query experiments --tag "perspective-correction"

# Recent artifacts (last 7 days)
etk query artifacts --recent 7d

# Custom SQL
etk query sql "SELECT * FROM experiments WHERE status='active'"

# Export to JSON
etk query experiments --status active --output experiments.json

# Export to CSV
etk query metrics --experiment 20251217_024343 --output metrics.csv
```

### Phase 4: Analytics Dashboard (6-8 hours)

**Tasks**:
1. Create `etk dashboard` command
2. Experiment timeline visualization
3. Metrics trends
4. Tag cloud
5. Compliance heatmap

**Features**:
```bash
# Launch interactive dashboard
etk dashboard

# Generate static HTML report
etk dashboard --report dashboard.html

# Specific experiment deep-dive
etk dashboard --experiment 20251217_024343
```

**Visualization Libraries**:
- `plotly` - Interactive charts
- `pandas` - Data processing
- `jinja2` - HTML templating

## Integration Considerations

### 1. Consistency Strategy

**Single Source of Truth**: Markdown files

**Database Role**: Secondary index for fast queries

**Sync Direction**: Markdown → Database (one-way)

**Rationale**:
- Git tracks markdown (version control)
- Markdown human-readable and editable
- Database regenerable from markdown
- Pre-commit hooks validate markdown

### 2. Conflict Resolution

**Scenario**: Database out of sync with markdown

**Resolution**:
1. Always trust markdown files
2. `etk sync --all` rebuilds database from markdown
3. Database acts as cache, not authoritative source

### 3. Performance

**Estimated Overhead**:
- Sync time: ~10ms per artifact (negligible)
- Storage: ~5KB per experiment (minimal)
- Query time: <100ms for complex queries (excellent)

**Optimization**:
- Use transactions for batch inserts
- Index frequently queried columns
- FTS5 for full-text search
- Periodic VACUUM for maintenance

### 4. Backwards Compatibility

**No Breaking Changes**:
- Database integration is optional
- Existing workflows unchanged
- ETK CLI works without database
- All tools function independently

## Validation & Testing

### Test Suite

```python
# Test database sync
def test_sync_experiment():
    # Create experiment via ETK
    # Sync to database
    # Query database
    # Assert data matches

# Test bidirectional consistency
def test_consistency_check():
    # Modify markdown
    # Sync to DB
    # Verify DB matches markdown

# Test query performance
def test_query_performance():
    # Create 100 experiments
    # Query with filters
    # Assert response time < 100ms

# Test FTS search
def test_full_text_search():
    # Index artifacts
    # Search for keywords
    # Verify results accuracy
```

## Migration Plan

### Step 1: Inspect Existing Database

```bash
# Check existing schema
sqlite3 data/ops/tracking.db ".schema"

# Backup existing database
cp data/ops/tracking.db data/ops/tracking.db.backup
```

### Step 2: Apply Schema

```bash
# Apply new schema (creates tables if not exist)
sqlite3 data/ops/tracking.db < schema.sql
```

### Step 3: Initial Sync

```bash
# Sync all existing experiments
etk sync --all

# Verify sync
etk query experiments --count
```

### Step 4: Validation

```bash
# Compare markdown count vs DB count
find experiment_manager/experiments -name "*.md" | wc -l
sqlite3 data/ops/tracking.db "SELECT COUNT(*) FROM artifacts"

# Spot-check sample experiment
etk query artifacts --experiment 20251217_024343
```

## Usage Examples

### Query All Active Experiments

```bash
etk query sql "
SELECT
    experiment_id,
    name,
    created_at,
    (SELECT COUNT(*) FROM artifacts WHERE experiment_id = e.experiment_id) as artifact_count
FROM experiments e
WHERE status = 'active'
ORDER BY created_at DESC
"
```

### Find Experiments with Specific Tags

```bash
etk query sql "
SELECT DISTINCT e.*
FROM experiments e
JOIN tags t ON e.experiment_id = t.experiment_id
WHERE t.tag_name IN ('perspective-correction', 'image-processing')
"
```

### Metrics Trend Analysis

```bash
etk query sql "
SELECT
    m.metric_name,
    AVG(m.metric_value) as avg_value,
    MIN(m.metric_value) as min_value,
    MAX(m.metric_value) as max_value,
    COUNT(*) as sample_count
FROM metrics m
WHERE m.timestamp > datetime('now', '-30 days')
GROUP BY m.metric_name
ORDER BY avg_value DESC
"
```

## Roadmap Timeline

**Phase 1**: Schema Design (2-4 hours)
**Phase 2**: Sync Tool (4-6 hours)
**Phase 3**: Query Interface (3-5 hours)
**Phase 4**: Analytics Dashboard (6-8 hours)

**Total Estimated Effort**: 15-23 hours

**Priority**: Low (optional enhancement)

**Dependencies**: EDS v1.0 complete, CLI tool operational

## Decision Matrix

### When to Implement Database Integration

**Implement If**:
- Managing 20+ experiments
- Need complex cross-experiment queries
- Want automated analytics/dashboards
- Have performance concerns with file-based search
- Need integration with external tools (BI, dashboards)

**Defer If**:
- Managing <10 experiments
- File-based search sufficient
- No analytics requirements
- Limited development time
- Prefer simplicity over features

## Notes

- Database integration is **optional** - core functionality complete without it
- Markdown files remain single source of truth
- Database acts as secondary index for performance
- All existing tools continue to work independently
- Can be implemented incrementally (phase-by-phase)
- Existing `tracking.db` can be inspected/reused or recreated

## References

- SQLite Documentation: https://www.sqlite.org/docs.html
- SQLite FTS5: https://www.sqlite.org/fts5.html
- Pandas SQL: https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html
- Plotly Python: https://plotly.com/python/

---

*Database integration roadmap for EDS v1.0 experiment tracker*
