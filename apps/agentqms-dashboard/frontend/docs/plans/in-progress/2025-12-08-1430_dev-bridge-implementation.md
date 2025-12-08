
---
title: "Implementation Plan: Python Bridge Backend"
type: development
status: in-progress
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 14:30 (KST)
phase: 2
priority: critical
tags: [development, backend, fastapi, bridge-server, implementation]
---

# Implementation Plan: Python Bridge (Backend)

**Target:** Connect React Dashboard to Local File System.
**Tech Stack:** Python 3.10+, FastAPI, Uvicorn.

## 1. Prerequisites
- [ ] Python Environment installed.
- [ ] Dependencies: `pip install fastapi uvicorn pydantic`

## 2. Directory Structure
Create the following file in `AgentQMS/agent_tools/bridge/`:
```text
AgentQMS/agent_tools/bridge/
├── __init__.py
├── server.py          # Main FastAPI Entry point
├── fs_utils.py        # File System Helpers
└── tool_runner.py     # Subprocess wrapper
```

## 3. Component Specifications

### A. `server.py` (The API)
**Role:** Routes HTTP requests from the UI to local functions.
**CORS:** Must allow `http://localhost:3000` (or wherever React runs).

**Code Skeleton:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="AgentQMS Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
def status():
    return {"status": "online", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### B. `fs_utils.py` (The Hands)
**Role:** Safe reading/writing of files.
**Security:** Ensure paths do not escape `PROJECT_ROOT` (prevent `../../` attacks).

**Key Functions:**
- `list_dir(path: str)`
- `read_file(path: str)`
- `write_file(path: str, content: str)`

### C. `tool_runner.py` (The Muscle)
**Role:** Executes the existing audit scripts.

**Logic:**
1. Receive `tool_id` and `args`.
2. Map `tool_id` to actual script path (e.g., `../audit/validate_frontmatter.py`).
3. Construct command: `['python', script_path, '--arg', val]`.
4. Run via `subprocess.run(capture_output=True)`.
5. Return STDOUT/STDERR.

## 4. Integration Steps
1. **Develop**: Create the python files as specified above.
2. **Run**: Start server `python AgentQMS/agent_tools/bridge/server.py`.
3. **Verify**: Visit `http://localhost:8000/docs` to see Swagger UI.
4. **Connect**: Update React App to fetch from `http://localhost:8000/api/status`.
