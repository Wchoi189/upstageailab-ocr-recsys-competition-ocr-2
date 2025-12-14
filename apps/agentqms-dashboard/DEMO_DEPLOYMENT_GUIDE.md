# AgentQMS Dashboard - Demo Deployment Guide

**Purpose**: Deploy AgentQMS Dashboard to free hosting (Google AI Studio, Vercel, Railway, etc.) for Kaggle Vibe Code with Gemini 3 competition demo.

---

## Prerequisites

### Required Files (Minimum Bundle)
```
agentqms-dashboard/
├── frontend/                    # React app
├── backend/                     # FastAPI server
├── demo_data/                   # NEW: Sample artifacts (see below)
├── demo_scripts/                # NEW: Stub implementations
├── requirements.txt             # Python dependencies
├── package.json                 # Node dependencies
├── Dockerfile                   # Container config
└── README.md
```

---

## Part 1: Create Demo Data

### Create Sample Artifacts

**Location**: `demo_data/artifacts/`

```bash
mkdir -p demo_data/artifacts/{implementation_plans,assessments,audits,bug_reports,design_documents}
```

#### Sample 1: Implementation Plan
**File**: `demo_data/artifacts/implementation_plans/2025-12-01_1000_plan-ocr-feature.md`
```markdown
---
title: "Implementation Plan: OCR Feature Enhancement"
type: implementation_plan
status: active
created: 2025-12-01 10:00 (KST)
updated: 2025-12-01 10:00 (KST)
phase: 2
priority: high
tags: [ocr, feature, enhancement]
---

# Implementation Plan: OCR Feature Enhancement

## Objective
Enhance OCR accuracy for handwritten text detection.

## Tasks
- [x] Research SOTA models
- [ ] Implement text detection pipeline
- [ ] Train on synthetic data
- [ ] Evaluate on validation set

## Timeline
- Week 1: Research
- Week 2-3: Implementation
- Week 4: Testing

## Success Metrics
- Accuracy > 95%
- Inference time < 100ms
```

#### Sample 2: Assessment
**File**: `demo_data/artifacts/assessments/2025-12-01_1100_assessment-model-performance.md`
```markdown
---
title: "Assessment: Model Performance Analysis"
type: assessment
status: complete
created: 2025-12-01 11:00 (KST)
updated: 2025-12-01 11:00 (KST)
category: evaluation
tags: [assessment, performance, analysis]
---

# Assessment: Model Performance Analysis

## Summary
Evaluated OCR model performance on test dataset.

## Findings
- Detection F1: 0.94
- Recognition accuracy: 0.89
- Processing speed: 85ms/image

## Recommendations
1. Fine-tune on domain-specific data
2. Optimize inference pipeline
3. Add post-processing rules

## Next Steps
- [ ] Implement recommendations
- [ ] Retest on new data
```

#### Sample 3: Bug Report
**File**: `demo_data/artifacts/bug_reports/2025-12-01_BUG_001_unicode-error.md`
```markdown
---
title: "BUG-001: Unicode Encoding Error"
type: bug_report
status: resolved
created: 2025-12-01 09:00 (KST)
updated: 2025-12-01 14:00 (KST)
severity: medium
tags: [bug, unicode, encoding]
---

# BUG-001: Unicode Encoding Error

## Description
Application crashes when processing Korean text with special characters.

## Steps to Reproduce
1. Load Korean document
2. Run OCR extraction
3. Error: `UnicodeDecodeError: 'utf-8' codec can't decode`

## Root Cause
File encoding mismatch (CP949 vs UTF-8).

## Fix
Added explicit UTF-8 encoding in file reader.

## Status
✅ Resolved in commit `a1b2c3d`
```

#### Sample 4: Audit
**File**: `demo_data/artifacts/audits/2025-12-01_1200_audit-code-quality.md`
```markdown
---
title: "Audit: Code Quality Review"
type: audit
status: active
created: 2025-12-01 12:00 (KST)
updated: 2025-12-01 12:00 (KST)
tags: [audit, quality, review]
---

# Audit: Code Quality Review

## Scope
Review codebase for compliance with style guide and best practices.

## Findings
- ✅ Type hints: 95% coverage
- ⚠️ Docstrings: 78% coverage (target: 90%)
- ✅ Unit tests: 87% line coverage
- ⚠️ Duplicate code: 3 instances found

## Action Items
- [ ] Add docstrings to utility modules
- [ ] Refactor duplicate validation logic
- [ ] Update CI to enforce 90% docstring coverage
```

#### Sample 5: Design Document
**File**: `demo_data/artifacts/design_documents/2025-12-01_1300_design-api-architecture.md`
```markdown
---
title: "Design: API Architecture"
type: design
status: active
created: 2025-12-01 13:00 (KST)
updated: 2025-12-01 13:00 (KST)
category: architecture
tags: [design, api, architecture]
---

# Design: API Architecture

## Overview
RESTful API for AgentQMS artifact management.

## Endpoints

### Artifacts
- `GET /api/v1/artifacts/list` - List all artifacts
- `POST /api/v1/artifacts/create` - Create new artifact
- `GET /api/v1/artifacts/{id}` - Get artifact by ID

### Tools
- `POST /api/v1/tools/exec` - Execute validation tool

### System
- `GET /api/v1/health` - Health check

## Authentication
Future: OAuth2 with JWT tokens (Phase 5).

## Rate Limiting
- 100 requests/minute per IP
- 1000 requests/hour per API key
```

---

## Part 2: Create Stub Scripts

### Stub 1: Validation Tool
**File**: `demo_scripts/validate_stub.py`
```python
#!/usr/bin/env python3
"""Demo stub: artifact validation."""
import sys

def main():
    print("=" * 60)
    print("ARTIFACT VALIDATION REPORT")
    print("=" * 60)
    print()
    print("✅ Frontmatter validation: PASS (5/5 artifacts)")
    print("✅ Naming convention: PASS (5/5 files)")
    print("⚠️  Boundary check: WARN (1 legacy directory detected)")
    print()
    print("Total artifacts scanned: 5")
    print("Violations found: 1 warning")
    print()
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Stub 2: Compliance Check
**File**: `demo_scripts/compliance_stub.py`
```python
#!/usr/bin/env python3
"""Demo stub: compliance check."""
import sys

def main():
    print("=" * 60)
    print("COMPLIANCE CHECK REPORT")
    print("=" * 60)
    print()
    print("📋 Framework Conventions:")
    print("  ✅ Artifact naming: COMPLIANT")
    print("  ✅ Frontmatter format: COMPLIANT")
    print("  ✅ Directory structure: COMPLIANT")
    print()
    print("📋 Code Standards:")
    print("  ✅ Type hints: 95% coverage")
    print("  ⚠️  Docstrings: 78% coverage (target: 90%)")
    print()
    print("Overall Status: COMPLIANT (1 recommendation)")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Stub 3: Tracking Status
**File**: `demo_scripts/tracking_stub.py`
```python
#!/usr/bin/env python3
"""Demo stub: tracking database status."""
import sys
import json

def main():
    status = {
        "plans": {
            "active": 3,
            "completed": 12,
            "total": 15
        },
        "experiments": {
            "running": 1,
            "completed": 8,
            "failed": 2,
            "total": 11
        },
        "debug_sessions": {
            "active": 0,
            "completed": 5,
            "total": 5
        }
    }

    print("=" * 60)
    print("TRACKING DATABASE STATUS")
    print("=" * 60)
    print()
    print(f"📊 Implementation Plans: {status['plans']['active']} active, {status['plans']['completed']} complete")
    print(f"🧪 Experiments: {status['experiments']['running']} running, {status['experiments']['completed']} complete")
    print(f"🐛 Debug Sessions: {status['debug_sessions']['active']} active, {status['debug_sessions']['completed']} complete")
    print()
    print(json.dumps(status, indent=2))
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

## Part 3: Modify Backend for Demo Mode

### Update Backend Routes to Use Stubs

**File**: `backend/routes/tools.py` (Add demo mode check)
```python
import os

# Check if running in demo mode
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
DEMO_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "../../demo_scripts")

@router.post("/exec")
async def execute_tool(request: ToolExecRequest):
    """Execute an AgentQMS tool via make command or demo stub."""

    if DEMO_MODE:
        # Use demo stubs
        demo_commands = {
            "validate": ["python", os.path.join(DEMO_SCRIPTS_DIR, "validate_stub.py")],
            "compliance": ["python", os.path.join(DEMO_SCRIPTS_DIR, "compliance_stub.py")],
            "boundary": ["python", os.path.join(DEMO_SCRIPTS_DIR, "validate_stub.py")],
            "discover": ["echo", "Demo: Tool discovery not implemented"],
            "status": ["echo", "Demo: Status check OK"],
        }
        cmd = demo_commands.get(request.tool_id)
        if not cmd:
            return {"success": False, "error": f"Unknown tool: {request.tool_id}", "output": ""}
    else:
        # Original production code
        tool_commands = {
            "validate": ["make", "-C", "AgentQMS/interface", "validate"],
            # ... rest of production commands
        }
        cmd = tool_commands.get(request.tool_id)

    # Execute command (rest of function unchanged)
    # ...
```

**File**: `backend/routes/artifacts.py` (Point to demo_data)
```python
import os

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
ARTIFACTS_DIR = "demo_data/artifacts" if DEMO_MODE else "docs/artifacts"

@router.get("/list")
async def list_artifacts():
    """List all artifacts from demo_data or real artifacts."""
    artifacts_path = Path(ARTIFACTS_DIR)
    # ... rest of function
```

**File**: `backend/routes/tracking.py` (Use stub)
```python
import os

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

@router.get("/status")
async def get_tracking_status(kind: str = Query("all")):
    """Get tracking database status (demo or real)."""

    if DEMO_MODE:
        # Call demo stub
        result = subprocess.run(
            ["python", "demo_scripts/tracking_stub.py"],
            capture_output=True,
            text=True
        )
        return {
            "kind": kind,
            "status": result.stdout,
            "success": result.returncode == 0
        }
    else:
        # Original production code
        from AgentQMS.agent_tools.utilities.tracking.query import get_status
        # ...
```

---

## Part 4: Docker Configuration

**File**: `Dockerfile`
```dockerfile
# Multi-stage build for production
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS backend
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY demo_data/ ./demo_data/
COPY demo_scripts/ ./demo_scripts/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Environment
ENV DEMO_MODE=true
ENV PORT=8000

EXPOSE 8000

# Start backend (serves frontend static files too)
CMD ["python", "backend/server.py"]
```

**Add static file serving to `backend/server.py`**:
```python
from fastapi.staticfiles import StaticFiles

# Serve frontend build
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
```

---

## Part 5: Environment Configuration

**File**: `.env.demo`
```bash
# Demo mode configuration
DEMO_MODE=true
GEMINI_API_KEY=your_api_key_here
PORT=8000
FRONTEND_URL=http://localhost:3000
```

---

## Part 6: Deployment Options

### Option A: Google Cloud Run (Free Tier)
```bash
# Build and deploy
gcloud run deploy agentqms-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DEMO_MODE=true
```

### Option B: Vercel (Frontend) + Railway (Backend)
**Vercel**: Deploy frontend only
```bash
cd frontend
vercel --prod
```

**Railway**: Deploy backend
```bash
railway up
```

### Option C: Render (Free Tier)
- Create `render.yaml`:
```yaml
services:
  - type: web
    name: agentqms-dashboard
    env: docker
    plan: free
    envVars:
      - key: DEMO_MODE
        value: true
```

### Option D: Hugging Face Spaces
```bash
# Add requirements to root
cat > requirements.txt << EOF
fastapi==0.115.0
uvicorn==0.38.0
python-multipart
EOF

# Create app.py in root
ln -s backend/server.py app.py

# Push to HF Space
git push hf main
```

---

## Part 7: Minimal Bundle Checklist

### Required Files (12 total)
- [x] `frontend/` (React app - already exists)
- [x] `backend/` (FastAPI - already exists)
- [x] `demo_data/artifacts/` (5 sample artifacts - CREATE)
- [x] `demo_scripts/` (3 stubs - CREATE)
- [x] `Dockerfile` (CREATE)
- [x] `.env.demo` (CREATE)
- [x] `requirements.txt` (already exists in backend/)
- [x] `README.md` (already exists)

### Optional But Recommended
- [ ] `docker-compose.yml` (local testing)
- [ ] `render.yaml` or `railway.json` (deployment config)
- [ ] `DEMO.md` (demo instructions for judges)

---

## Part 8: Testing Demo Locally

```bash
# Set demo mode
export DEMO_MODE=true

# Start backend
cd backend
python server.py

# Start frontend (separate terminal)
cd frontend
npm run dev

# Access demo
open http://localhost:3000
```

**Expected Demo Features**:
1. ✅ Artifact Generator creates sample artifacts
2. ✅ Framework Auditor shows validation report (from stub)
3. ✅ Integration Hub shows tracking status (from stub)
4. ✅ Strategy Dashboard shows metrics
5. ✅ Context Explorer visualizes 5 demo artifacts

---

## Estimated Bundle Size

```
Frontend build:     ~2.5 MB (minified)
Backend:            ~1.5 MB (Python code)
Demo artifacts:     ~15 KB (5 markdown files)
Demo stubs:         ~3 KB (3 Python scripts)
Dependencies:       ~50 MB (Docker image)
──────────────────────────────
Total:              ~54 MB (Docker image)
                    ~4 MB (source code only)
```

---

## Next Steps

1. **Create demo data**: Run script to generate 5-10 sample artifacts
2. **Create stubs**: Implement 3 stub scripts for tool execution
3. **Update backend**: Add demo mode environment variable checks
4. **Test locally**: Verify all features work with demo data
5. **Deploy**: Choose hosting platform and deploy
6. **Document**: Create DEMO.md with walkthrough for judges

---

## Competition Submission Tips

1. **Video Demo**: Record 2-3 minute walkthrough showing:
   - Artifact generation with Gemini API
   - Validation tool execution
   - Dashboard visualization

2. **GitHub README**: Include:
   - Live demo link
   - Screenshots of each feature
   - Architecture diagram
   - Gemini API integration highlights

3. **Highlight Gemini Integration**:
   - AI-powered artifact generation
   - Smart content suggestions
   - Natural language tool descriptions

---

**Quick Start Demo Deployment**:
```bash
# 1. Create demo data
python create_demo_data.py

# 2. Build Docker image
docker build -t agentqms-dashboard .

# 3. Test locally
docker run -p 8000:8000 -e DEMO_MODE=true agentqms-dashboard

# 4. Deploy to Cloud Run
gcloud run deploy --source .
```
