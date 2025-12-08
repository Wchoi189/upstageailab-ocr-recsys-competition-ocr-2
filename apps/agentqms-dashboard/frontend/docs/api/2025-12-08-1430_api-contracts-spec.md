
---
title: "AgentQMS Data Contracts & API Specifications"
type: api
status: draft
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 14:30 (KST)
phase: 2
priority: high
tags: [api, contracts, fastapi, endpoints, json-schema]
---

# AgentQMS Data Contracts & API Specifications

**Version:** 1.0.0
**Status:** Draft
**Usage:** Defines the interface between the AgentQMS React Dashboard and the Python Bridge (`agent_tools/server.py`).

---

## 1. System Status (`GET /api/status`)

**Purpose**: Verifies the connection between UI and Backend.

**Response (200 OK):**
```json
{
  "status": "online",
  "version": "1.0.0",
  "cwd": "/path/to/project/root",
  "agentqms_root": "AgentQMS/",
  "config": {
    "timezone": "KST",
    "enforce_branch_names": true
  }
}
```

---

## 2. File System Operations

### List Directory (`GET /api/fs/list`)
**Query Params**: `path` (string, relative to project root)

**Response (200 OK):**
```json
{
  "path": "AgentQMS/modules/ocr-2",
  "items": [
    { "name": "plans", "type": "directory" },
    { "name": "audit_v1.md", "type": "file", "size": 1024, "last_modified": "2025-01-01 12:00" }
  ]
}
```

### Read File (`GET /api/fs/read`)
**Query Params**: `path` (string)

**Response (200 OK):**
```json
{
  "path": "AgentQMS/modules/ocr-2/audit_v1.md",
  "content": "---\ntitle: ...",
  "encoding": "utf-8"
}
```

### Write File (`POST /api/fs/write`)
**Body:**
```json
{
  "path": "AgentQMS/modules/ocr-2/new_plan.md",
  "content": "..."
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "bytes_written": 2048
}
```

---

## 3. Tool Execution

### Run Audit Tool (`POST /api/tools/exec`)

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
  "stdout": "Validation Passed: all fields present.",
  "stderr": ""
}
```

---

## 4. Database / Registry

### Get Registry Index (`GET /api/registry/index`)

**Response (200 OK):**
```json
{
  "last_indexed": "2025-05-22 15:00",
  "artifacts": [
     { "id": "plan-001", "path": "...", "tags": ["ocr"] }
  ]
}
```
