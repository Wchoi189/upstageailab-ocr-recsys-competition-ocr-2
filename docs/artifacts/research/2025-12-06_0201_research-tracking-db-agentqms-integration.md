---
type: "research"
category: "research"
status: "active"
version: "1.0"
tags: ['research', 'investigation', 'analysis']
title: "AgentQMS & Tracking Database Integration Instructions"
date: "2025-12-06 02:01 (KST)"
---

# AgentQMS & Tracking Database Integration Instructions

## Overview for AI Agents

The Tracking Database (`data/ops/tracking.db`) is a critical infrastructure component that AgentQMS tools must integrate with at two levels:

1. **Artifact Lifecycle**: When artifacts are created, validated, or migrated
2. **Tool Operations**: When utility tools (smart_populate, deprecated_registry, plan_progress) modify project state

AI agents should treat the Tracking DB as the **source of truth for development operations** and ensure synchronization at integration points.

## Integration Points

### 1. Artifact Creation & Registration
**When**: `make create-*` targets (plan, assessment, design, etc.)
**Agent Action**:
- Verify artifact is created via Makefile (not manually)
- Check that artifact metadata is auto-populated from `smart_populate.py`
- Confirm artifact key is registered in Tracking DB via `feature_plans` table
- If artifact creation fails, check Tracking DB connectivity with `make track-init`

**Integration Flow**:
```
User → Makefile target → artifact_workflow.py → smart_populate.py → tracking.db
                                                     ↓
                                              metadata suggestions
                                              (title, tags, owner)
```

### 2. Plan Progress Tracking
**When**: Tasks within implementation plans are marked complete
**Agent Action**:
- Use `plan_progress.py` to update plan completion status in Tracking DB
- Verify plan task records exist in `plan_tasks` table before updates
- Sync bidirectionally: artifact status ↔ Tracking DB status
- Before completing a plan, validate all child tasks are marked complete

**Integration Points**:
- Read from: `feature_plans`, `plan_tasks` tables
- Write to: `plan_tasks.status`, `feature_plans.updated_at`
- Validation: Ensure plan_id exists before updating tasks

### 3. Deprecated Symbol Registry
**When**: Deprecated code is detected during artifact validation or manual registration
**Agent Action**:
- Register deprecated symbols via `make deprecated-register SYMBOL=X FILE=Y REPLACEMENT=Z`
- Tracking DB stores deprecation metadata in custom table or JSON blob
- Query deprecated symbols before code generation or refactoring
- Sync deprecated registry with artifact migration operations

**Integration Commands**:
```bash
# Before generating code that uses old APIs
make deprecated-list CATEGORY=api

# When discovering deprecated usage in artifacts
make deprecated-validate --all

# When registering new deprecations
make deprecated-register SYMBOL=OldFunc FILE=src/old.py REPLACEMENT=src/new.py
```

### 4. Experiment & Run Tracking
**When**: Running experiments or validation suites
**Agent Action**:
- Create experiment records via Tracking CLI before running validation
- Log validation/compliance check results as experiment runs
- Summarize findings and store as experiment summaries
- Use experiment key for reproducibility tracking

**Typical Workflow**:
```bash
# Create experiment for validation run
python AgentQMS/agent_tools/utilities/tracking/cli.py exp new \
  --title "AgentQMS Validation Suite" \
  --objective "Ensure all artifacts comply with standards"

# Add run after validation completes
python AgentQMS/agent_tools/utilities/tracking/cli.py exp run-add <exp_key> 1 \
  --params '{"artifacts_checked": 15}' \
  --outcome "passed"

# Summarize results
python AgentQMS/agent_tools/utilities/tracking/cli.py exp summarize <exp_key> \
  --style short --points "15/15 artifacts passing"
```

### 5. Validation & Compliance Feedback Loop
**When**: Running `make validate` or `make compliance`
**Agent Action**:
- Log validation results to Tracking DB as experiment runs
- If compliance fails, create bug report and link to Tracking DB
- Use Tracking DB status to avoid re-running expensive validations
- Query previous validation results to accelerate iterative fixes

**Integration Points**:
- Query: `experiments`, `experiment_runs` for historical validation results
- Write: Create new experiment run records for each validation suite execution

## Required Initialization

Before any Tracking DB operations, AI agents MUST:

```bash
# Initialize Tracking DB schema
make track-init

# Verify DB is accessible
python -c "from AgentQMS.agent_tools.utilities.tracking.query import get_status; print(get_status('all'))"
```

If initialization fails:
1. Check file permissions on `data/ops/tracking.db`
2. Verify `data/ops/` directory exists (create if missing)
3. Confirm Python SQLite support: `python -c "import sqlite3; print(sqlite3.version)"`

## Error Handling & Recovery

### Common Integration Issues

| Issue | Root Cause | Agent Action |
|-------|-----------|--------------|
| "no such table" | Tracking DB not initialized | Run `make track-init` |
| Foreign key violation | Artifact created without registration | Manually add to feature_plans table or recreate |
| Stale status | Tracking DB out of sync with filesystem | Run repair tool (Task 4.3, pending) |
| Validation loop failure | DB transaction deadlock | Retry with backoff (max 3 attempts) |

### Recovery Protocol

1. **Verify DB integrity**: `make track-init` (idempotent, safe to re-run)
2. **Check connectivity**: Query current status to confirm reads work
3. **Inspect recent changes**: Review artifact git history vs. Tracking DB records
4. **Document discrepancy**: Create bug report with DB state snapshot
5. **Escalate**: If mismatch persists, trigger repair workflow (Task 4.3)

## Agent Decision Tree

```
Artifact operation requested?
├─ Create artifact (new)
│  └─ Run Makefile target → verify in Tracking DB within 2 seconds
├─ Update artifact (existing)
│  └─ Check plan_tasks status → sync updates → verify atomicity
└─ Validate/migrate artifact
   └─ Query deprecated registry → apply migrations → register changes

Before any DB write:
├─ Initialize if needed: make track-init
├─ Check key doesn't exist (avoid duplicates)
├─ Verify foreign keys exist (plan_id, owner, etc.)
└─ Prepare rollback (store previous state)
```

## Integration Checklist for Agent Tasks

- [ ] Initialize Tracking DB before operations (`make track-init`)
- [ ] Verify artifact creation via both filesystem AND Tracking DB
- [ ] Sync plan progress bidirectionally (artifact ↔ DB)
- [ ] Query deprecated registry before code generation
- [ ] Log validation results as experiment runs
- [ ] On error: Check connectivity, verify DB schema, escalate if unresolved
- [ ] Document any manual interventions in bug reports

## References

- CLI Commands: `AgentQMS/knowledge/references/tracking/cli_reference.md`
- Database Schema: `AgentQMS/knowledge/references/tracking/db_api.md`
- Smart Populate Tool: `AgentQMS/agent_tools/utilities/smart_populate.py`
- Plan Progress Tool: `AgentQMS/agent_tools/utilities/plan_progress.py`
- Deprecated Registry: `AgentQMS/agent_tools/utilities/deprecated_registry.py`
- Task 4.3: Tracking DB repair tool (pending implementation)
- Reference 2

---

*This research document follows the project's standardized format for research documentation.*
