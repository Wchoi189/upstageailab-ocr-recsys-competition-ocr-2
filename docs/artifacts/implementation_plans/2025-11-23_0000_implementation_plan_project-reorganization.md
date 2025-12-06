---
title: "Project Reorganization Plan"
date: "2025-12-06 18:09 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---







# Project Reorganization Implementation Plan

## 1. Overview
This plan outlines the steps to reorganize the project structure to improve modularity, follow conventions, and fix tooling discovery issues. The primary changes involve moving frontend applications to `apps/`, backend services to `apps/backend/`, and consolidating agent tools into a proper `agent_qms` package.

## 2. Objectives
- **Standardize App Structure:** Move all applications into `apps/` (except `ui/` which is deprecated).
- **Consolidate Backend:** Group backend services under `apps/backend/`.
- **Fix Agent Tooling:** Convert `agent_qms` into a proper Python package and consolidate all agent scripts there.
- **Clean Root:** Reduce clutter in the root directory.

## 3. Implementation Steps

### Phase 1: Structural Moves (Completed)
- [x] **Frontend Relocation:**
    - Move `frontend/` to `apps/frontend/`.
    - Update `package.json` workspaces.
    - Update `package.json` scripts.
    - Fix build errors in `apps/frontend`.
- [x] **Backend Relocation:**
    - Create `apps/backend/`.
    - Move `services/` to `apps/backend/services/`.
    - Update imports in `run_spa.py` and tests.
- [x] **Agent QMS Consolidation:**
    - Move `scripts/agent_tools/*` to `agent_qms/`.
    - Delete `scripts/agent_tools/`.
    - Create `agent_qms/__init__.py`.
    - Update imports in Python files.

### Phase 2: Configuration & Documentation (Completed)
- [x] **Update `pyproject.toml`:**
    - Add `agent_qms` to the project configuration (if needed) or ensure it's discoverable.
    - Verify `apps/backend` is discoverable.
- [x] **Update Documentation:**
    - Update `README.md` to reflect new paths.
    - Update `CONTRIBUTING.md` with new setup instructions.
- [x] **Deprecation:**
    - Add deprecation notice to `ui/README.md`.

### Phase 3: Verification (Completed)
- [x] **Verify Agent QMS:** Ensure `python -m agent_qms` works.
- [x] **Verify Full Build:** Ensure `npm run build:spa` and Python tests pass.

## 4. Rollback Plan
If critical issues arise, revert file moves using `git` and restore `package.json` to its previous state.
