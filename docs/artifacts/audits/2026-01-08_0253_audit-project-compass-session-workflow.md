---
type: "audit"
category: "compliance"
status: "draft"
version: "1.0"
ads_version: "1.0"
related_artifacts: []
generated_artifacts: []
tags: "project_compass,audit,session_workflow,malfunction"
title: "Project Compass Session Workflow Malfunction"
date: "2026-01-08 02:53 (KST)"
branch: "main"
description: "Audit of the Project Compass session export workflow identifying context staleness and folder naming issues."
---

# Audit Report: Project Compass Session Workflow

**Date**: 2026-01-08
**Auditor**: Antigravity (Agent)

## Executive Summary
A structural malfunction was identified in the Project Compass session export workflow. While the actual work (Verification, Handover Notes) is being recorded correctly in `session_handover.md` and `manifest.json` notes, the **Session Context** (`active_context/current_session.yml`) has become stale. This stale context is being used to generate session IDs and folder names, resulting in duplicate metadata across distinct sessions.

## Key Findings

### 1. Stale Session Context
The file `active_context/current_session.yml` is identical across the last 4 exported sessions, despite significant progress in the project.
- **Found in**:
    - `.../20260108_005808_.../active_context/current_session.yml` (Latest)
    - `.../20260107_151929_.../active_context/current_session.yml` (Oldest of 4)
- **Stale Content**:
    - `session_id`: "2026-01-07_phase1_2_complete"
    - `completed_date`: "2026-01-07 02:20 (KST)"
- **Impact**: The system believes every new session is still "Phase 1 & 2 Complete", regardless of actual work (Phase 3, 4, 3.2, etc.).

### 2. Repetitive Folder Naming
Session export folders use the stale `session_id` as a suffix.
- **Observed Folders**:
    - `20260108_005808_2026-01-07_phase1_2_complete`
    - `20260107_232850_2026-01-07_phase1_2_complete`
    - `20260107_165741_2026-01-07_phase1_2_complete`
    - `20260107_151929_2026-01-07_phase1_2_complete`
- **Correction**: Folders should likely be named based on the *current* objective or a unique slug derived from the new manifest note.

### 3. Manifest Integrity
- **Partial Failure**: `manifest.json` correctly captures unique `note` fields for each session (e.g., "Phase 3.2 complete..."), proving that *some* input is fresh.
- **Root Cause**: However, `original_session_id` is pulled from the stale `active_context`, perpetuating the lineage error.

### 4. Missing Instructions
- **Gap**: `project_compass/AGENTS.yaml` defines entry points for `session_init` but lacks specific protocols for **updating the session context** before export.
- **Observation**: There is no `README.md` or `PROTOCOL.md` in `project_compass` to guide the agent or user on how to "roll over" the session ID to the next phase clearly.

## Recommendations

1.  **Force Context Refresh**: The `session_export` or `update_session` workflow must verify that `active_context/current_session.yml` has been updated since the last export, or prompt the user/agent to update it.
2.  **Uncouple Folder Naming**: Do not rely solely on `original_session_id` for folder naming if that ID is expected to be static for a long duration. Use the current date + a slug from the `note`.
3.  **Documentation**: Add a "Session Lifecycle" section to `AGENTS.yaml` or a new `README.md` in `project_compass` defining the `init -> work -> update_context -> export` cycle.

## Next Steps for Remediation
1.  Manually update `active_context/current_session.yml` to reflect the current state (Phase 3.2 Complete).
2.  Investigate the `etk.factory` or export scripts to see why they don't auto-update the context.
