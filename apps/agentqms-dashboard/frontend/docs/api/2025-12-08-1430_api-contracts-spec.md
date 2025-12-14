---
title: "AgentQMS Data Contracts & API Specifications"
type: api
status: active
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 15:00 (KST)
phase: 2
priority: high
tags: [api, contracts, fastapi, endpoints, json-schema]
---

# AgentQMS Data Contracts & API Specifications

**Version:** 1.1.0
**Status:** Active
**Usage:** Defines the interface between the AgentQMS React Dashboard and the Python Bridge (`apps/agentqms-dashboard/backend/server.py`).

---

## 1. System Status (`GET /status`)

**Purpose**: Verifies the connection between UI and Backend.

**Response (200 OK):**
```json
{
  "status": "online",
  "version": "0.1.0",
  "cwd": "/path/to/project/root",
  "agentqms_root": "/path/to/project/root",
  "config": {
    "timezone": "KST",
    "enforce_branch_names": true
  }
}
```

---

## 2. File System Operations (Bridge)

### List Directory (`GET /fs/list`)
**Query Params**: `path` (string, relative to project root)

**Response (200 OK):**
```json
{
  "path": "AgentQMS/modules/ocr-2",
  "items": [
    { "name": "plans", "type": "directory", "size": 4096, "last_modified": 1700000000.0 },
    { "name": "audit_v1.md", "type": "file", "size": 1024, "last_modified": 1700000000.0 }
  ]
}
```

### Read File (`GET /fs/read`)
**Query Params**: `path` (string)

**Response (200 OK):**
```json
{
  "path": "AgentQMS/modules/ocr-2/audit_v1.md",
  "content": "---\ntitle: ...",
  "encoding": "utf-8"
}
```

### Write File (`POST /fs/write`)
**Body:**
```json
{
  "path": "AgentQMS/modules/ocr-2/new_plan.md",
  "content": "..."
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "bytes_written": 2048
}
```

---

## 3. Tool Execution (Bridge)

### Run Tool (`POST /tools/exec`)

**Body:**
```json
{
  "tool_id": "validate_frontmatter",
  "args": {
    "file": "AgentQMS/modules/ocr-2/plan.md"
  }
}
```

**Response (200 OK):**
```json
{
  "tool_id": "validate_frontmatter",
  "exit_code": 0,
  "stdout": "Validation successful.",
  "stderr": ""
}
```

---

## 4. Artifact Management API (v1)

**Base URL**: `/api/v1`

### 4.1 List Artifacts (`GET /api/v1/artifacts`)
**Query Params**:
- `type` (optional): `implementation_plan`, `assessment`, `audit`, `bug_report`
- `status` (optional): `draft`, `active`, `completed`
- `limit` (optional): default 50

**Response (200 OK):**
```json
{
  "items": [
    {
      "id": "2025-12-08_1430_plan_dashboard-integration",
      "type": "implementation_plan",
      "title": "AgentQMS Manager Dashboard Integration Testing",
      "status": "active",
      "path": "docs/artifacts/implementation_plans/2025-12-08_1430_plan_dashboard-integration.md",
      "created_at": "2025-12-08T14:30:00"
    }
  ],
  "total": 1
}
```

### 4.2 Get Artifact (`GET /api/v1/artifacts/{id}`)
**Response (200 OK):**
Returns full artifact content including parsed frontmatter and body.

### 4.3 Create Artifact (`POST /api/v1/artifacts`)
**Body:**
```json
{
  "type": "implementation_plan",
  "title": "New Feature Plan",
  "content": "# Plan Content..."
}
```
**Behavior**: Automatically generates filename with timestamp `YYYY-MM-DD_HHMM_[type]_[slug].md`.

### 4.4 Update Artifact (`PUT /api/v1/artifacts/{id}`)
**Body:**
```json
{
  "content": "Updated content...",
  "frontmatter_updates": { "status": "completed" }
}
```

### 4.5 Delete Artifact (`DELETE /api/v1/artifacts/{id}`)
**Behavior**: Moves file to `docs/artifacts/archive/` or deletes (configurable).

---

## 5. Artifact Schemas

### 5.1 Common Frontmatter
All artifacts must include:
```yaml
type: [implementation_plan | assessment | audit | bug_report]
category: string
status: [draft | active | completed | archived]
version: string
title: string
date: string (YYYY-MM-DD HH:MM (KST))
branch: string (optional)
tags: array<string>
```

### 5.2 Implementation Plan
**File Pattern**: `docs/artifacts/implementation_plans/YYYY-MM-DD_HHMM_implementation_plan_[name].md`
**Required Sections**:
- Master Prompt
- Living Implementation Blueprint
- Progress Tracker
- Implementation Outline

### 5.3 Assessment
**File Pattern**: `docs/artifacts/assessments/YYYY-MM-DD_HHMM_assessment_[name].md`
**Required Sections**:
- Executive Summary
- Detailed Analysis
- Recommendations

### 5.4 Audit
**File Pattern**: `docs/artifacts/audits/YYYY-MM-DD_HHMM_audit_[name].md`
**Required Sections**:
- Scope
- Findings
- Compliance Status

### 5.5 Bug Report
**File Pattern**: `docs/artifacts/bug_reports/YYYY-MM-DD_HHMM_bug_report_[name].md`
**Required Sections**:
- Description
- Reproduction Steps
- Expected vs Actual
- Logs/Evidence

---

## 6. Security & Configuration

### 6.1 Authentication
- **Local Development**: No authentication required. The bridge runs on `localhost` and binds to `0.0.0.0` (be careful in shared environments).
- **Production/Remote**: API Key authentication via `X-API-Key` header (Future Scope).

### 6.2 CORS Configuration
The backend is configured to allow Cross-Origin Resource Sharing (CORS) to support the React frontend running on a different port (e.g., Vite on 3000/5173).

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict to ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 6.3 Versioning
- **Bridge API**: Root level (legacy/simple) or `/api/v0`.
- **Artifact API**: `/api/v1`.
