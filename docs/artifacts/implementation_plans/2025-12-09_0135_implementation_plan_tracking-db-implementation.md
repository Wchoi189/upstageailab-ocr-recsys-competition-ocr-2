---
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'development']
title: "Tracking Database Implementation and Sync"
date: "2025-12-09 01:35 (KST)"
branch: "feature/agentqms-dashboard-integration"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Tracking Database Implementation and Sync**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Tracking Database Implementation and Sync

## Progress Tracker
- **STATUS:** Ready for Implementation
- **CURRENT STEP:** Phase 1, Task 1.1 - Database Schema Extension
- **LAST COMPLETED TASK:** None (Initial Database Created)
- **NEXT TASK:** Extend schema to support full artifact metadata and indexing operations

### Implementation Outline (Checklist)

#### **Phase 1: Database Schema & Indexing Engine (1-2 hours)**
1. [ ] **Task 1.1: Extend Database Schema**
   - [ ] Add columns: `category`, `tags_json`, `metadata_json`, `content_hash`
   - [ ] Create indexes on `type`, `status`, `created_at`, `content_hash`
   - [ ] Add `sync_log` table to track indexing operations
   - [ ] Write migration script for schema v2

2. [ ] **Task 1.2: Build Artifact Indexer**
   - [ ] Create `AgentQMS/agent_tools/database/artifact_indexer.py`
   - [ ] Implement `scan_artifacts()` to walk `docs/artifacts/**/*.md`
   - [ ] Parse frontmatter using `python-frontmatter` library
   - [ ] Calculate content hash (SHA256) for duplicate detection
   - [ ] Batch insert/update into SQLite with conflict resolution

3. [ ] **Task 1.3: CLI Interface for Sync Operations**
   - [ ] Add `make db-sync` target to `AgentQMS/interface/Makefile`
   - [ ] Support `--full` (re-index all) and `--incremental` (changed only)
   - [ ] Add `--dry-run` mode to preview changes
   - [ ] Output summary: `X new, Y updated, Z unchanged`

#### **Phase 2: Backend Integration (1-2 hours)**
4. [ ] **Task 2.1: Add Database Query Endpoints**
   - [ ] Create `apps/agentqms-dashboard/backend/routes/database.py`
   - [ ] Endpoint: `GET /api/v1/database/stats` (total artifacts, last sync time)
   - [ ] Endpoint: `POST /api/v1/database/sync` (trigger indexing job)
   - [ ] Endpoint: `GET /api/v1/database/artifacts` (query with filters: type, status, tags)
   - [ ] Use SQLite `LIKE` for tag search, `FTS5` for full-text (optional future enhancement)

5. [ ] **Task 2.2: Replace In-Memory Listing with DB Queries**
   - [ ] Modify `artifacts.py` `list_artifacts()` to query SQLite instead of `glob`
   - [ ] Keep file-based `get_artifact(id)` for content retrieval
   - [ ] Add fallback: if DB is empty, trigger auto-sync on first request

#### **Phase 3: Frontend Dashboard Updates (30 min - 1 hour)**
6. [ ] **Task 3.1: Integration Hub - Database Tab Enhancements**
   - [ ] Update `IntegrationHub.tsx` to call `/api/v1/database/stats`
   - [ ] Display: Last sync time, total artifacts, artifacts by type breakdown
   - [ ] Add "Sync Now" button that calls `/api/v1/database/sync`
   - [ ] Show real-time sync progress (optional: use WebSocket or polling)

7. [ ] **Task 3.2: Librarian - Use DB for Listing**
   - [ ] Update `Librarian.tsx` to call `/api/v1/database/artifacts`
   - [ ] Add filters: Type dropdown, Status dropdown, Tag search
   - [ ] Implement pagination (limit/offset)
   - [ ] Display artifact count badge on each filter

#### **Phase 4: Testing & Validation (30 min)**
8. [ ] **Task 4.1: Unit Tests**
   - [ ] Test indexer with sample artifacts (valid, invalid, missing frontmatter)
   - [ ] Test conflict resolution (update existing artifact)
   - [ ] Test hash-based change detection

9. [ ] **Task 4.2: Integration Tests**
   - [ ] End-to-end: Sync → Query → Verify counts match filesystem
   - [ ] Test API error handling (db locked, permission errors)
   - [ ] Test dashboard UI: Sync button → Poll stats → Verify update

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] SQLite database schema with proper indexes for performance
- [ ] Separation of concerns: Indexer (CLI), Query layer (Backend), UI (Frontend)
- [ ] Idempotent sync operations (safe to run multiple times)
- [ ] Content-hash based change detection (avoid unnecessary updates)

### **Integration Points**
- [ ] Use existing `python-frontmatter` library for YAML parsing
- [ ] Integrate with backend `apps/agentqms-dashboard/backend/` structure
- [ ] Add to `AgentQMS/interface/Makefile` for CLI access
- [ ] Frontend calls new `/api/v1/database/*` endpoints

### **Quality Assurance**
- [ ] Unit tests for indexer: valid/invalid artifacts, conflict resolution
- [ ] Integration tests: E2E sync → query → verify
- [ ] Performance: Index 100+ artifacts in <2 seconds
- [ ] Error handling: DB locked, permission errors, malformed frontmatter

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] Running `make db-sync` indexes all artifacts from `docs/artifacts/` into SQLite
- [ ] Dashboard Integration Hub displays accurate artifact count and last sync time
- [ ] Librarian page lists artifacts from database with type/status/tag filtering
- [ ] Incremental sync only processes changed files (verified via content hash)
- [ ] Manual "Sync Now" button in UI triggers re-indexing

### **Technical Requirements**
- [ ] Database file `.agentqms/tracking.db` contains populated `artifacts` table
- [ ] All Python code is type-hinted and documented with docstrings
- [ ] API endpoints return JSON with proper error codes (404, 500)
- [ ] SQLite database size remains <10MB for 500+ artifacts
- [ ] No N+1 query problems (batch operations used)

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW
### **Active Mitigation Strategies**:
1. **Incremental Rollout**: Backend indexer works standalone before UI integration
2. **Backward Compatibility**: Keep file-based glob listing as fallback if DB fails
3. **Dry-Run Mode**: Test sync operations without modifying database

### **Fallback Options**:
1. **If SQLite locking issues occur**: Implement read replica or switch to readonly mode
2. **If performance degrades**: Add batch size limits, implement background worker
3. **If frontmatter parsing fails**: Log errors, skip invalid files, continue indexing

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
- Task completion (move to next task)
- Blocker encountered (document and propose solution)
- Technical discovery (update approach if needed)
- Quality gate failure (address issues before proceeding)

**Update Format:**
1. Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2. Mark completed items with [x]
3. Add any new discoveries or changes to approach
4. Update risk assessment if needed

---

## 🚀 **Immediate Next Action**

**TASK:** Extend SQLite schema and create migration script

**OBJECTIVE:** Add columns for full artifact metadata and create indexes for efficient querying

**APPROACH:**
1. Create `AgentQMS/agent_tools/database/schema_v2.py` with table definitions
2. Add columns: `category TEXT`, `tags_json TEXT`, `metadata_json TEXT`, `content_hash TEXT`
3. Create indexes on `(type, status)`, `created_at`, `content_hash`
4. Add `sync_log` table: `(id, sync_type, artifact_count, started_at, completed_at, status)`
5. Write migration function that checks schema version and applies changes

**SUCCESS CRITERIA:**
- Schema migration script runs successfully on existing `.agentqms/tracking.db`
- All new columns and indexes are created
- Running `sqlite3 .agentqms/tracking.db ".schema"` shows updated schema
- Migration is idempotent (can run multiple times safely)

---

## 📁 **File Structure**

```
AgentQMS/agent_tools/database/
├── __init__.py
├── schema.py              # Database schema definitions
├── indexer.py             # Artifact scanning and indexing
└── migrations.py          # Schema migration utilities

apps/agentqms-dashboard/backend/routes/
└── database.py            # Database query endpoints

.agentqms/
└── tracking.db            # SQLite database file
```

---

## 🔧 **Key Dependencies**

- **Python**: `sqlite3` (stdlib), `python-frontmatter`, `pathlib`
- **Backend**: FastAPI, Pydantic
- **Frontend**: React, TypeScript (existing bridgeService)

---

## 📝 **Implementation Notes**

### Database Schema Design
```sql
-- Extended artifacts table
CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    title TEXT,
    status TEXT DEFAULT 'draft',
    category TEXT,
    tags_json TEXT,           -- JSON array of tags
    metadata_json TEXT,       -- Full frontmatter as JSON
    content_hash TEXT,        -- SHA256 of content for change detection
    created_at TEXT,
    updated_at TEXT
);

CREATE INDEX idx_type_status ON artifacts(type, status);
CREATE INDEX idx_created ON artifacts(created_at DESC);
CREATE INDEX idx_hash ON artifacts(content_hash);

-- Sync operation log
CREATE TABLE sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sync_type TEXT NOT NULL,  -- 'full' or 'incremental'
    artifacts_scanned INTEGER,
    artifacts_updated INTEGER,
    artifacts_added INTEGER,
    started_at TEXT,
    completed_at TEXT,
    status TEXT               -- 'success', 'failed', 'in_progress'
);
```

### Indexer Workflow
1. Walk `docs/artifacts/` directories
2. For each `.md` file:
   - Parse frontmatter using `frontmatter.load()`
   - Calculate SHA256 hash of content
   - Check if hash differs from DB
   - If changed: parse metadata, update/insert row
3. Batch commits every 50 files for performance
4. Log operation to `sync_log` table

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
