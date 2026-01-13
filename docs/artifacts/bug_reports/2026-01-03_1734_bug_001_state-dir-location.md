---
ads_version: "1.0"
type: "bug_report"
category: "troubleshooting"
status: "active"
severity: "medium"
version: "1.0"
tags: ['bug', 'issue', 'troubleshooting']
title: "Incorrect State Directory Location"
date: "2026-01-03 17:34 (KST)"
branch: "main"
description: "Bug report regarding the incorrect generation of state directories in the repository root."
---

# Bug Report - Incorrect State Directory Location
Bug ID: BUG-001


## Summary
When AgentQMS runs in a "self-contained" mode (where the `.agentqms` configuration directory is located inside `AgentQMS/.agentqms`), the project root detection logic incorrectly identifies the repository root (parent of `AgentQMS/`) as the project root. This causes state files and the `.agentqms` folder to be regenerated in the repository root instead of staying contained within `AgentQMS/`.

## Environment
- **OS/Env**: Linux / Dev Container
- **Dependencies**: AgentQMS (internal toolchain)

## Reproduction
1. Refactor AgentQMS to move `.agentqms/` inside `AgentQMS/` (e.g., `AgentQMS/.agentqms/`).
2. Run a command that triggers config loading or state generation (e.g., `make plugins-snapshot` or any script using `get_project_root()`).
3. Observe that a new `.agentqms/` directory is created in the repo root (`/workspaces/.../`), duplicating the one inside `AgentQMS/`.

## Comparison
**Expected**: `get_project_root()` should return `/workspaces/.../AgentQMS` when a local `.agentqms` exists, keeping all state contained.
**Actual**: `get_project_root()` returned `/workspaces/.../` (the parent), causing state leakage.

## Logs
```
# Python check before fix:
>>> get_project_root()
Path('/workspaces/upstageailab-ocr-recsys-competition-ocr-2')

# After fix:
>>> get_project_root()
Path('/workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS')
```

## Impact
- **Severity**: Low (annoyance/clutter)
- **Affected Users**: Developers refactoring the AgentQMS core.
- **Workaround**: Manually delete the root `.agentqms` folder.

