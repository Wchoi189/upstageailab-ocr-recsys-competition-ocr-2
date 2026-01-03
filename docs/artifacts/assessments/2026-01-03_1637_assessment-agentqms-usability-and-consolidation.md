---
ads_version: "1.0"
type: "assessment"
category: "evaluation"
status: "draft"
version: "1.0"
tags: ['assessment', 'usability', 'agentqms', 'consolidation']
title: "AgentQMS Usability and Consolidation Assessment"
date: "2026-01-03 16:37 (KST)"
branch: "main"
---

# AgentQMS Usability and Consolidation Assessment

## Purpose
To assess the usability of AgentQMS for AI Agents, resolve architecture overlaps, and fix environment reliability issues.

## Findings

### 1. Architecture Overlaps & Path Rot
- **Current State**:
    - `AgentQMS/standards` defines Policies and Schemas.
    - `AgentQMS/.agentqms/plugins` defines Implementations (Templates, Context Bundles).
    - `AgentQMS/archive_agentqms` contains deprecated docs.
- **Issue**:
    - Legacy path references (`agent_tools`, `conventions`, `knowledge`) persist in code, causing failures (e.g., `reindex_artifacts.py` not found).
    - `artifact_workflow.py` references `AgentQMS/agent_tools`, but the directory is `AgentQMS/tools`.
- **Resolution Status**: major path references fixed in `discovery.py`, `context_bundle.py`, `validation.py`, `discover.py`. `artifact_workflow.py` still needs `agent_tools` -> `tools` fix.

### 2. Environment Reliability
- **Issue**: Agents (and users) invoke `python` directly, missing dependencies. `uv` is required but easily forgotten.
- **Failed Attempt**: modifying `bin/python` caused recursion/reference errors.
- **Root Cause**: Scripts assume `python` is the system python or a specific venv python, but `uv` manages ephemeral environments or specific venv locations.

### 3. Tool Exposure
- **Issue**: Manual prompting of rules is not scalable. Entrypoints are ignored.
- **Current Mechanism**: `discover.py` and `AGENTS.yaml` exist but aren't strictly enforced or "sticky".

## Recommendations

### 1. Consolidate & Fix Paths (Result: Low Latency/High Stability)
- **Action**: Fix `artifact_workflow.py` to point to correct `AgentQMS/tools/documentation/reindex_artifacts.py`.
- **Action**: Update `.agentqms/plugins/validators.yaml` to point to `AgentQMS/standards/schemas/plugin_validators.json`.
- **Action**: Formalize the `Standards` (Policy) vs `.agentqms` (Impl) split in documentation to reduce "bloat" perception.

### 2. Enforce Environment via Shim/Configuration (Result: Reliability)
- **Recommendation**: Do NOT replace `bin/python` if it breaks scripts.
- **Alternative**: Create `bin/task_runner` or `bin/aqms` that wraps `uv run`.
- **Agent Instruction**: Update `AGENTS.yaml` (or system prompt source) to explicitly require `uv run` for all python execution.
- **Validation**: Add a "Check Environment" tool/step that verifies `uv` is active.

### 3. Improve Tool Exposure (Result: Usability)
- **Action**: Ensure `mcp_server.py` exposes the "Discovery" tool (`discover.py` functionality) so agents can "ask" what tools are available.
- **Action**: Create a "Manifest" or "Cheatsheet" artifact that acts as a quick-reference context bundle for agents.

## Implementation Plan

### Phase 1: Fix Broken References (Completed)
- [x] Fix `artifact_workflow.py` (`agent_tools` typo).
- [x] Fix `validators.yaml` schema path.

### Phase 2: Environment Hardening (Completed)
- [x] Verify `AGENTS.yaml` contains `uv` enforcement instructions.
- [x] Create `bin/aqms` wrapper for common tasks.
- [x] Remove `bin/python` shim.
- [x] Update `AgentQMS/bin/Makefile` to force `uv run` usage.

### Phase 3: Documentation
- [ ] Update `README.md` to explain the Standards vs Plugins architecture clearly.

## Conclusion
The "bloat" is largely due to legacy paths and lack of clear separation documentation. By fixing the remaining path errors and documenting the architecture, the system becomes robust. Environment issues should be solved by policy (Agent Instructions) + convenience wrappers (`aqms`) rather than hijacking `python` binary which is fragile.
