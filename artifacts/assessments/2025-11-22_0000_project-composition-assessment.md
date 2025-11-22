---

title: "Project Composition Assessment"

author: "ai-agent"

timestamp: "2025-11-22 00:00 KST"

branch: "main"

status: "draft"

tags: ["created-by-script"]

type: "assessment"

category: "evaluation"

---

# Project Composition Assessment

## Progress Tracker
*(Required for iterative assessments, debugging sessions, or incremental work)*

- **STATUS:** Not Started / In Progress / Completed
- **CURRENT STEP:** [Current phase or task being worked on]
- **LAST COMPLETED TASK:** [Description of last completed task]
- **NEXT TASK:** [Description of the immediate next task]

### Assessment Checklist
- [ ] Initial assessment complete
- [ ] Analysis phase complete
- [ ] Recommendations documented
- [ ] Review and validation complete

---

## 1. Summary

The project is a hybrid monorepo containing both Python (OCR, RecSys, Analysis) and Node.js (Frontend, Console) components. While the project follows some conventions (e.g., `packages/`, `apps/`), there is significant fragmentation in application placement and tooling organization. Specifically, the `agent_qms` system is underutilized due to structural issues preventing proper discovery.

## 2. Assessment

### Project Organization
- **Structure:** The project root is cluttered with a mix of configuration files and source directories.
- **Applications:** Distributed inconsistently across `apps/`, `frontend/`, `ui/`, and `services/`.
- **Tooling:** Split between `scripts/` and `agent_qms/`, leading to confusion and duplication.

### Points of Confusion
- **Inconsistent App Placement:**
  - `frontend/` exists at the root level, while `apps/playground-console/` is in `apps/`.
  - `ui/` (Python-based) and `services/` (Backend) are also at the root.
- **Script/Tooling Fragmentation:**
  - `scripts/` contains a mix of project scripts and `agent_tools/`.
  - `agent_qms/` contains `toolbelt` but is not configured as a package.

### Package Directories & Convention
- **`packages/`**: Contains `console-shared`. Follows convention.
- **`apps/`**: Contains `playground-console`. Follows convention.
- **`frontend/`**: Root-level package. **Breaks convention** (Should be in `apps/`).
- **`node_modules/`**: Root level. Standard for monorepos.

### Node Modules Organization
- The project uses NPM Workspaces.
- **Efficiency:** Moving `frontend` to `apps/frontend` would align with the workspace pattern and clean up the root.

### Frontend & UI Applications
1.  **`frontend/`**: Vite + React/Vue SPA.
2.  **`apps/playground-console/`**: Frontend application.
3.  **`ui/`**: Python-based UII  (`preprocessing_viewer_app.py`).

### Agent QMS Organization & Discovery
- **Problem:** `agent_qms/` is not a valid Python package (missing `__init__.py`). This prevents tools like Copilot from discovering the modules and workflows defined therein.
- **Duplication:** `scripts/agent_tools/` overlaps with `agent_qms/`.

## 3. Recommendations

### Re-organization
1.  **Move Apps:**
    - `frontend/` -> `apps/frontend/`
    - `ui/` -> `apps/ui-python/`
    - `services/` -> `apps/backend-service/`
2.  **Consolidate Tooling:**
    - Move `scripts/agent_tools/*` to `agent_qms/`.
    - Delete `scripts/agent_tools/`.

### Fix Agent QMS Discovery
1.  **Create `agent_qms/__init__.py`** to make it a discoverable Python package.
2.  **Install** `agent_qms` in editable mode or ensure it is in `PYTHONPATH`.
3.  **Update Instructions** to reference the `agent_qms` package paths.
