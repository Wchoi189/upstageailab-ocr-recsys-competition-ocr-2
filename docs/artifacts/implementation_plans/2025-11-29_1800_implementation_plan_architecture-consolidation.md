---
category: planning
status: active
type: implementation_plan
title: Implementation Plan: AgentQMS Architecture Consolidation
version: 1.0.0
ads_version: 1.0.0
date: 2025-11-29 18:00 (KST)
---

# Implementation Plan: AgentQMS Architecture Consolidation

> **Context**: Converging `.agentqms/`, `.ai-instructions/`, and `AgentQMS/` into a single `AgentQMS/` root directory to creates a single source of truth for the project.

## 🎯 Goal
Consolidate architecture to improve discoverability, reduce maintenance burden, and align with "Zero-Archaeology" principles for AI agents.

## 📦 Phase 1: Migration (High Impact)
1.  [ ] **Task 1.1: Standards Migration**
    -   Move `.ai-instructions/tier1-sst` → `AgentQMS/standards/tier1-sst`
    -   Move `.ai-instructions/tier2-framework` → `AgentQMS/standards/tier2-framework`
    -   Move `.ai-instructions/tier3-agents` → `AgentQMS/standards/tier3-agents`
    -   Move `.ai-instructions/tier4-workflows` → `AgentQMS/standards/tier4-workflows`
    -   Move `.ai-instructions/INDEX.yaml` → `AgentQMS/standards/INDEX.yaml`

2.  [ ] **Task 1.2: Conventions Migration**
    -   Move `AgentQMS/conventions/schemas` → `AgentQMS/standards/schemas`
    -   Move `AgentQMS/conventions/templates` → `AgentQMS/standards/templates`
    -   Consolidate `conventions` README/docs into `AgentQMS/docs/standards_guide.md`

3.  [ ] **Task 1.3: Configuration Migration**
    -   Move `.agentqms/settings.yaml` → `AgentQMS/config/settings.yaml`
    -   Move `.agentqms/plugins/` → `AgentQMS/config/plugins/`
    -   Move `.agentqms/state/` → `AgentQMS/state/`

## 🔨 Phase 2: Refactoring (Internal)
4.  [ ] **Task 2.1: Package Restructuring**
    -   Rename `AgentQMS/agent_tools` → `AgentQMS/tools`
    -   Rename `AgentQMS/knowledge` → `AgentQMS/docs`
    -   Rename `AgentQMS/interface` → `AgentQMS/bin`

5.  [ ] **Task 2.2: Path Updates**
    -   Update `AGENTS.md` to point to `AgentQMS/standards/INDEX.yaml`
    -   Update Python imports (bulk replace `AgentQMS.agent_tools` → `AgentQMS/tools`)
    -   Update `AgentQMS/bin/Makefile` paths
    -   Update `INDEX.yaml` relative paths (if any)

## 🧹 Phase 3: Cleanup & Verification
6.  [ ] **Task 3.1: Cleanup**
    -   Remove empty `.ai-instructions/` directory
    -   Remove empty `.agentqms/` directory
    -   Remove empty `AgentQMS/conventions/` directory

7.  [ ] **Task 3.2: Verification**
    -   Run `validate_environment.py`
    -   Run `AgentQMS/bin/make test` (or equivalent)
    -   Verify AI context map in `AGENTS.md` works

## ⚠️ Implementation Notes
-   **Atomic Moves**: Suggest performing moves using `git mv` to preserve history.
-   **Breaking Changes**: This is a major structural change. Ensure no active experiments rely on hardcoded paths to `.agentqms` or `.ai-instructions`.
