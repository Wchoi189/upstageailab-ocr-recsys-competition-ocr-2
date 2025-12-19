---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'state-management', 'database', 'optimization']
title: "State Management Redesign: Ultra-Concise Database-First Architecture"
date: "2025-12-19 21:41 (KST)"
branch: "main"
---

# Master Prompt

You are implementing ultra-concise state management redesign based on architecture analysis (ref: brain/architecture_assessment.md). Execute systematically without asking for confirmation. Current experiment (20251218_1900_border_removal_preprocessing) will NOT be affected - all changes apply to new experiments only.

---

# State Management Redesign Implementation

## Objective

Replace 9KB YAML state files with 100-byte .state files + database tables. Achieve 99% core state memory reduction, <1ms state queries, zero-parse AI workflows.

## References

- Architecture analysis: `/home/vscode/.gemini/antigravity/brain/841ca6b4-64e8-441f-9e1f-90b801d24cad/`
- Current ETK tool: `experiment-tracker/etk.py`
- Database: `data/ops/tracking.db`
- Active experiment: `experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/` (UNCHANGED)

## Progress Tracker
- **STATUS:** Not Started
- **CURRENT STEP:** Phase 1, Task 1.1 - Database schema creation
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Create SQL schema file for new tables

---

## Implementation Outline

### Phase 1: Database Foundation (Week 1)

**1.1 Create Database Schema**
- [ ] Create `experiment-tracker/schema/state_tables.sql`
- [ ] Define `experiment_state` table (experiment_id, current_task_id, current_phase, status, checkpoint_path)
- [ ] Define `experiment_tasks` table (task_id, experiment_id, title, status, priority, depends_on, timestamps)
- [ ] Define `experiment_decisions` table (decision_id, experiment_id, decision, rationale, impact, date)
- [ ] Define `experiment_insights` table (insight_id, experiment_id, insight, impact, category, related_task_id)
- [ ] Define `state_transitions` table (transition_id, experiment_id, from_state, to_state, transition_type, timestamp)
- [ ] Add indexes: `idx_state_status`, `idx_tasks_experiment`, `idx_tasks_priority`, `idx_decisions_experiment`, `idx_transitions_experiment`

**1.2 Apply Schema**
- [ ] Run migration: `sqlite3 data/ops/tracking.db < experiment-tracker/schema/state_tables.sql`
- [ ] Verify tables created: `SELECT name FROM sqlite_master WHERE type='table'`
- [ ] Test constraints: Insert/query test records
- [ ] Document schema in `experiment-tracker/docs/database-schema.md`

**1.3 Create .state File Utilities**
- [ ] Create `experiment-tracker/src/experiment_tracker/utils/state_file.py`
- [ ] Implement `read_state(exp_path) -> dict` (line-delimited key=value parser)
- [ ] Implement `update_state(exp_path, **updates)` (atomic write with temp file)
- [ ] Implement `create_state_file(exp_id, checkpoint_path=None)` (generate from experiment ID)
- [ ] Add unit tests in `tests/test_state_file.py`

---

### Phase 2: ETK Integration (Week 2)

**2.1 Extend ETK CLI**
- [ ] Update `experiment-tracker/etk.py`:
  - [ ] Modify `init_experiment()` to create `.state` file (5 fields only)
  - [ ] Modify `init_experiment()` to populate `experiment_state` table
  - [ ] Add `sync_state_to_db(experiment_id)` method
  - [ ] Update `get_current_state()` to read from `.state` file OR database
  - [ ] Add backward compat: read `state.yml` if `.state` doesn't exist

**2.2 Task Management**
- [ ] Add `create_task(experiment_id, title, description, priority)` to ETK
- [ ] Add `set_current_task(experiment_id, task_id)` with state transition logging
- [ ] Add `complete_task(task_id, notes)` to update status and timestamps
- [ ] Add `get_pending_tasks(experiment_id, limit)` query method
- [ ] Update `experiment-tracker/README.md` with new ETK commands

**2.3 Decision/Insight Tracking**
- [ ] Add `record_decision(experiment_id, decision, rationale, impact)` to ETK
- [ ] Add `record_insight(experiment_id, insight, impact, category)` to ETK
- [ ] Add `get_recent_decisions(experiment_id, limit)` query method
- [ ] Add CLI commands: `etk decision add`, `etk insight add`

---

### Phase 3: Migration Tools (Week 3)

**3.1 YAML to Database Migration**
- [ ] Create `experiment-tracker/scripts/migrate_state_yaml_to_db.py`:
  - [ ] Parse `state.yml` tasks array
  - [ ] Insert tasks into `experiment_tasks` table
  - [ ] Parse decisions array, insert into `experiment_decisions`
  - [ ] Parse insights array, insert into `experiment_insights`
  - [ ] Generate `.state` file from YAML metadata
  - [ ] Backup original `state.yml` to `.metadata/archive/`

**3.2 Bulk Migration Script**
- [ ] Create `experiment-tracker/scripts/migrate_all_experiments.py`
- [ ] Iterate all experiments in `experiments/`
- [ ] Skip if `.state` already exists
- [ ] Log skipped experiments (e.g., active 20251218_1900)
- [ ] Generate migration report: `experiments_migrated.txt`

**3.3 Progress-Tracker Integration**
- [ ] Create `experiment-tracker/.metadata/00-status/` template directory
- [ ] Add `current-state.md` auto-generation script
- [ ] Add `next-steps.md` generator from pending tasks query
- [ ] Add `blockers.md` template
- [ ] Update `etk init` to create status hierarchy

---

### Phase 4: Testing & Validation (Week 4)

**4.1 Unit Tests**
- [ ] Test `.state` file read/write atomicity
- [ ] Test database task CRUD operations
- [ ] Test state transition logging
- [ ] Test decision/insight recording
- [ ] Test migration script with sample `state.yml`
- [ ] Coverage target: >90% for new code

**4.2 Integration Tests**
- [ ] Create new test experiment: `etk init test_state_redesign`
- [ ] Verify `.state` file created (100 bytes)
- [ ] Verify `experiment_state` table populated
- [ ] Add task: `etk task add "Test task"`
- [ ] Set current: `etk state --set current_task=test_task`
- [ ] Query: `etk state --json`
- [ ] Validate <1ms query latency

**4.3 Backward Compatibility**
- [ ] Test ETK with old experiment (state.yml only)
- [ ] Test ETK with new experiment (.state only)
- [ ] Test ETK with hybrid (both formats)
- [ ] Verify no disruption to 20251218_1900 experiment

---

## Technical Requirements

### Architecture
- [ ] Database-first design (DB is source of truth)
- [ ] .state file as cached view (100-byte limit)
- [ ] Atomic state transitions with audit trail
- [ ] Line-delimited key=value format (no YAML parser)

### Integration Points
- [ ] SQLite database at `data/ops/tracking.db`
- [ ] ETK CLI tool (`experiment-tracker/etk.py`)
- [ ] AgentQMS tracking integration (existing)
- [ ] Progress-tracker patterns (00-status/ hierarchy)

### Performance
- [ ] State file read: <1ms
- [ ] Database query (current task): <5ms
- [ ] State transition (with audit): <15ms
- [ ] Memory: <500 bytes core state (vs 9KB YAML)

### Quality Assurance
- [ ] Unit test coverage: >90%
- [ ] Integration tests for all ETK commands
- [ ] Migration tested on 3+ experiments
- [ ] Performance benchmarks documented

---

## Success Criteria

### Functional
- [ ] New experiments use .state file (100 bytes)
- [ ] ETK can query current task in <5ms
- [ ] Tasks/decisions/insights tracked in database
- [ ] State transitions logged for audit
- [ ] Old experiments still work (backward compat)

### Technical
- [ ] Core state: 99% memory reduction (9KB → 100 bytes)
- [ ] Overall experiment state: 65% reduction (29KB → 10KB)
- [ ] Zero YAML parsing for state queries
- [ ] AI context consumption: 99% reduction (10KB → 100 bytes)
- [ ] All code type-hinted and documented

---

## Risk Mitigation

### Current Risk Level: MEDIUM

### Active Strategies
1. **Incremental rollout**: New experiments only, existing untouched
2. **Backward compatibility**: Support both .state and state.yml formats
3. **Database backups**: Automated before schema changes
4. **Comprehensive testing**: Unit + integration + migration tests

### Fallback Options
1. **Schema issues**: Rollback SQL, keep using state.yml
2. **Migration failures**: Manual migration for affected experiments
3. **Performance regression**: Optimize indexes, add caching layer
4. **ETK bugs**: Hotfix release, document workarounds

---

## Immediate Next Action

**TASK:** Create database schema SQL file

**OBJECTIVE:** Define 5 new tables with indexes for state management

**APPROACH:**
1. Create `experiment-tracker/schema/state_tables.sql`
2. Define table structures from architecture recommendations
3. Add indexes for AI query patterns
4. Include DROP IF EXISTS for idempotency
5. Test with SQLite CLI

**SUCCESS CRITERIA:**
- [ ] Schema file created and syntactically valid
- [ ] All 5 tables defined with correct columns/constraints
- [ ] 6 indexes created for query optimization
- [ ] File is idempotent (can run multiple times)

**OUTPUT FILES:**
- `experiment-tracker/schema/state_tables.sql`

---

**CONTEXT FOR AI AGENTS:**

**Do NOT modify:**
- `experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/` (active experiment)
- Existing `state.yml` files in other experiments
- Core ETK functionality (only extend, don't break)

**References required:**
- Architecture assessment: `/home/vscode/.gemini/antigravity/brain/841ca6b4-64e8-441f-9e1f-90b801d24cad/architecture_assessment.md`
- State recommendations: `/home/vscode/.gemini/antigravity/brain/841ca6b4-64e8-441f-9e1f-90b801d24cad/state_management_recommendations.md`
- Database guidelines: `/home/vscode/.gemini/antigravity/brain/841ca6b4-64e8-441f-9e1f-90b801d24cad/database_utilization_guidelines.md`

**Key Principles:**
- Machine-parseable formats only (.state file, not YAML)
- Database is source of truth, files are cache
- <1ms state access is requirement
- 99% memory reduction is target
- Zero disruption to running experiments

---

*This implementation plan enables parallel development while active experiment 20251218_1900 continues unaffected. All changes apply forward-only to new experiments.*
