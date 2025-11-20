---
title: "AI Agent System – Single Source of Truth"
date: "2025-11-20"
type: "guide"
category: "ai_agent"
status: "active"
version: "1.1"
tags: ["ai_agent", "rules", "operations"]
---

AI Agent System – Single Source of Truth
=======================================

Status: ACTIVE

🚨 **CRITICAL: Artifact Creation**
----------------------------------
**NEVER use `write` tool for artifacts in `artifacts/` or `docs/bug_reports/`.**

**Preferred:** AgentQMS toolbelt:
```python
from agent_qms.toolbelt import AgentQMSToolbelt

toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="assessment",  # or "implementation_plan", "bug_report"
    title="My Artifact",
    content="## Summary\n...",
    author="ai-agent",
    tags=["tag1", "tag2"]
)
```

**Alternative:** CLI script (two workflows):

**Primary: Create Empty Template → Fill In Manually**
```bash
# Creates template with placeholders, then fill in manually
python scripts/agent_tools/core/artifact_workflow.py create \
  --type bug_report \
  --name "bug-name" \
  --title "Bug Title" \
  --bug-id "BUG-YYYYMMDD-###" \
  --severity "High" \
  --tags "bug,issue"
```

**Alternative: Create with Full Content from File**
```bash
# For programmatic creation with full content
python scripts/agent_tools/core/artifact_workflow.py create \
  --type bug_report \
  --name "bug-name" \
  --title "Bug Title" \
  --content-file path/to/content.md \
  --bug-id "BUG-YYYYMMDD-###" \
  --severity "High" \
  --tags "bug,issue"
```

**Types:** `implementation_plan`, `assessment`, `bug_report`, `data_contract` (see `agent_qms/q-manifest.yaml`)

**Naming Convention:**
- **Assessments, Implementation Plans, Guides**: Must use timestamped filenames (YYYY-MM-DD_HHMM_name.md)
- **All Artifacts**: Frontmatter must include `timestamp` field (YYYY-MM-DD HH:MM KST) and `branch` field (git branch name)
- **Example**: `2025-11-09_1430_architecture-reorganization-plan.md` with frontmatter:
  ```yaml
  ---
  title: "Architecture Reorganization Plan"
  author: "ai-agent"
  timestamp: "2025-11-09 14:30 KST"
  branch: "main"
  status: "draft"
  tags: []
  ---
  ```

**Validation:** Before using `write` tool, check path:
```python
from agent_qms.toolbelt.validation import check_before_write
check_before_write("artifacts/assessments/my-file.md")  # ❌ Raises error
```

**Progress Trackers:**
- Implementation Plans: Always include Progress Tracker (Blueprint Protocol PROTO-GOV-003)
- Assessments: Include Progress Tracker for iterative work
- Update Progress Tracker after each task completion or blocker

**Extended Sessions:**
- Keep intermediate summaries minimal; defer comprehensive summaries until plan/goal completion
- Use progress trackers for extended sessions when applicable
- Artifacts allowed during intermediate steps

**Bug Reports:** Generate ID: `uv run python scripts/bug_tools/next_bug_id.py`, then use artifact workflow. Filename: `BUG-YYYYMMDD-###_description.md`

**⚠️ CRITICAL: Code Indexing with Bug ID**
When making changes to core project files, **ALWAYS embed the bug ID in the code**:
- Add bug ID to function docstrings
- Add bug ID comments at change locations
- Create code changes document: `docs/bug_reports/BUG-YYYYMMDD-###-code-changes.md`
- Index at function level, not just file level (functions survive refactoring)
- See `docs/agents/protocols/development.md` for detailed protocol

**Implementation Plans:** Use Blueprint Protocol Template (PROTO-GOV-003). See `docs/maintainers/protocols/governance/03_blueprint_protocol_template.md`

**Status Updates:** When implementation plans are completed, update frontmatter status:
```bash
# Auto-detect status from Progress Tracker
python scripts/agent_tools/core/artifact_workflow.py update-status --file path/to/plan.md --auto-detect

# Or manually set status
python scripts/agent_tools/core/artifact_workflow.py update-status --file path/to/plan.md --status completed

# Check for status mismatches
python scripts/agent_tools/core/artifact_workflow.py check-status-mismatches
```

Tool Discovery
--------------
**Check existing scripts BEFORE creating new ones:**
```bash
python scripts/agent_tools/core/discover.py
python scripts/agent_tools/core/discover.py --list
python scripts/agent_tools/documentation/validate_manifest.py
```
Pre-commit hook blocks duplicate scripts automatically.

Core Rules
----------
- Always use automation tools; NEVER create artifact files manually
- Check existing scripts first before creating new ones
- No loose docs in project root; no ALL CAPS filenames (except README.md, CHANGELOG.md, bug report IDs like BUG-YYYYMMDD-###)
- Test in browser and check logs; do not rely on unit tests alone
- Use `width="stretch"` or `width="content"` for Streamlit (NOT `use_container_width`) - Legacy apps only
- Frontend development: Use React 19 + TypeScript + Vite for SPA, Next.js 16 for console
- Test scripts belong in `tests/scripts/`, not project root

Path Management
---------------
**CRITICAL**: Always use centralized path utilities to avoid brittle path calculations.

**✅ DO** - Use centralized PROJECT_ROOT:
```python
# Import PROJECT_ROOT from central utility (stable, works from any location)
from ocr.utils.path_utils import PROJECT_ROOT

# Or use path resolver for configurable paths
from ocr.utils.path_utils import get_path_resolver

resolver = get_path_resolver()
output_dir = resolver.config.output_dir
config_dir = resolver.config.config_dir
```

**❌ DON'T** - Brittle patterns that break when files move:
```python
# ❌ Brittle - breaks if file moved
project_root = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
```

**Environment Variables**: Paths support `OCR_*` environment variable overrides.
See `docs/maintainers/environment-variables.md` for details.

For Hydra config paths, paths are automatically resolved from project root.

Operational Commands
--------------------

**Training & Inference (Python):**
```bash
# Training
uv run python runners/train.py preset=example

# Testing
uv run python runners/test.py preset=example checkpoint_path="..."

# Prediction
uv run python runners/predict.py preset=example checkpoint_path="..."

# Process monitoring
python scripts/process_monitor.py
```

**Backend API (FastAPI):**
```bash
# Start FastAPI backend (development)
uv run uvicorn services.playground_api.app:app --reload --host 127.0.0.1 --port 8000

# API documentation available at http://127.0.0.1:8000/docs
```

**Frontend Development (Vite SPA):**
```bash
# Start Vite dev server
cd frontend
npm run dev

# Or use Makefile
make fe  # or make frontend-dev

# Access at http://localhost:5173
```

**Next.js Console:**
```bash
# Start Next.js console (development)
cd apps/playground-console
npm run dev

# Or use workspace script
npm run dev:console

# Access at http://localhost:3000
```

**Combined Stack (Backend + Frontend):**
```bash
# Start both FastAPI backend and Vite frontend together
make fs  # or make stack-dev

# Stops backend when frontend exits
# Frontend: http://localhost:5173
# API Docs: http://127.0.0.1:8000/docs
```

**Legacy Streamlit UI (Deprecated):**
```bash
# Legacy Streamlit apps (being phased out)
python run_ui.py command_builder
python run_ui.py inference
python run_ui.py evaluation_viewer
```

**Testing:**
```bash
# Python tests
uv run pytest tests/

# Frontend E2E tests
cd frontend
npm run test:e2e

# Frontend type checking
cd frontend
npm run type-check
```

File Impact (Test Scope)
------------------------
- **Model components**: test training/inference
- **UI components**:
  - Frontend (React): test in browser, check browser console
  - Streamlit (legacy): test in browser, check Streamlit logs
- **API endpoints**: test via FastAPI docs or curl/requests
- **Web workers**: test in browser DevTools, check worker console
- **Data processing**: test with sample data
- **Config changes**: test with validation

When Stuck
----------
- **Frontend issues**: Check browser console, Network tab, and worker errors
- **Backend issues**: Check FastAPI logs and `/docs` endpoint
- **Build issues**: Re-run discovery/validate, check dependencies
- **Python imports**: Ensure using `uv run` prefix, check PYTHONPATH
- **TypeScript errors**: Run `npm run type-check` in frontend directory
- **CORS errors**: Verify FastAPI CORS middleware is configured
- **Worker errors**: Check browser console for detailed worker logs
- **General debugging**:
  - Re-run discovery/validate
  - Check logs and training outputs
  - Read `docs/maintainers/architecture/` for detailed context
  - Review `docs/agents/protocols/` for workflows
  - Check implementation plans in `artifacts/implementation_plans/`

Frontend Development Rules
--------------------------
**React/Vite SPA:**
- Use TypeScript for all new components
- Test components in browser (not just unit tests)
- Check browser console for errors
- Web workers in `frontend/workers/` use ES modules
- API client in `frontend/src/api/` handles retries and errors
- Use `make fs` to start both backend and frontend

**Next.js Console:**
- Use Chakra UI components for styling
- API calls go through Next.js proxy routes (`/app/api/*`)
- Follow Next.js 16 App Router conventions
- Use React Query for data fetching

**Web Workers:**
- Workers in `frontend/workers/` process images
- ONNX.js for client-side ML (rembg)
- Check worker errors in browser DevTools → Sources → Worker
- Worker errors are logged with `[Worker]` or `[rembg]` prefixes

**Legacy Streamlit (Being Phased Out):**
- NEVER use `use_container_width` (deprecated after 2025-12-31)
- ALWAYS use `width="stretch"` instead of `use_container_width=True`
- ALWAYS use `width="content"` instead of `use_container_width=False`
- Applies to: `st.plotly_chart()`, `st.dataframe()`, `st.button()`, all chart/table components
- Look for inline comments "CRITICAL: NEVER pass use_container_width" at call sites
- Assertions will fail if deprecated parameter is used - this is intentional
