---
type: walkthrough
category: feature
status: active
version: "1.0"
tags: github, sync, entrypoints, walkthrough
title: GitHub Sync Extension Walkthrough
date: "2026-01-05 18:54"
branch: main
---

# Walkthrough - GitHub Sync Extension

I have successfully implemented the GitHub Sync Extension planning, refactoring the `sync_github_projects.py` script to be more robust and feature-rich.

## Changes

### `AgentQMS/tools/utils/sync_github_projects.py`

- **Refactoring**: Converted the procedural script into a `GitHubManager` class.
- **New Features**:
    - `--init`: Checks for an existing GitHub Project V2 by title (Session ID - Objective) and creates one if it doesn't exist.
    - `--roadmap <file>`: Reads a markdown artifact and publishes it as an Issue in the project, with labels "roadmap" and "compass-plan".
- **Enhancements**:
    - **Existence Checks**: Added `find_project_by_title` and `find_issue_by_title` to prevent duplicates.
    - **Error Handling**: Wrapped subprocess calls in try/except blocks.

## Verification Results

### Dry Run

I performed a dry run using the following command:

```bash
python AgentQMS/tools/utils/sync_github_projects.py --init --roadmap docs/artifacts/implementation_plans/2026-01-05_github_sync_extension_plan.md --dry-run
```

**Output:**

```
Searching for project: 2026-01-05_strategic_refactor - Master Architecture Refactoring (Phase 1 & 2 Planning)...
[DRY RUN] Would execute: gh project list --format json --limit 50
Creating Project: 2026-01-05_strategic_refactor - Master Architecture Refactoring (Phase 1 & 2 Planning)
[DRY RUN] Would execute: gh project create --title 2026-01-05_strategic_refactor - Master Architecture Refactoring (Phase 1 & 2 Planning) --format json
[DRY RUN] Would execute: gh issue list --search Implementation Roadmap: 2026-01-05_strategic_refactor in:title --json title,url --limit 1
Publishing Roadmap from docs/artifacts/implementation_plans/2026-01-05_github_sync_extension_plan.md
Creating Issue: Implementation Roadmap: 2026-01-05_strategic_refactor
[DRY RUN] Would execute: gh issue create --title Implementation Roadmap: 2026-01-05_strategic_refactor --body ... --label roadmap --label compass-plan
Syncing 3 tasks...
...
```

The dry run confirmed that the script:
1.  Correctly constructs the project title.
2.  Checks for project existence.
3.  Proposes creation of the project.
4.  Checks for roadmap issue existence.
5.  Proposes creation of the roadmap issue.
6.  Syncs individual tasks.

### Demo & Iteration

I verified the implementation by running a live sync using the **Master Architecture Refactoring Plan**.

**Command:**
```bash
python AgentQMS/tools/utils/sync_github_projects.py --init --roadmap docs/artifacts/implementation_plans/2026-01-05_1750_implementation_plan_master-refactoring.md
```

**Issues Encountered & Resolved:**

1.  **"Owner is required" Error**: `gh project create` failed because it required an `--owner` flag in non-interactive mode.
    *   *Fix*: Implemented `get_current_user()` to fetch the authenticated user via `gh api user` and pass it to project creation.
2.  **"Label not found" Error**: `gh issue create` failed because labels like `roadmap` did not exist in the repository.
    *   *Fix*: Implemented `ensure_label()` to check for label existence and create them on demand.

**Final Result:**
- **Project Created**: `2026-01-05_strategic_refactor - Master Architecture Refactoring (Phase 1 & 2 Planning)` covering the session.
- **Roadmap Published**: The Master Plan was successfully posted as an issue with the `roadmap` label.
- **Tasks Synced**: 3 active compass tasks were synced as issues.

Success confirmed with exit code 0.

## Entrypoints Configured

I have configured the following entrypoints for easy access:

### VS Code Tasks
- **`🎯 AgentQMS: Init GitHub Project`**: initializes a new project and optionally publishes a roadmap (prompts for file).
- **`🎯 AgentQMS: Publish Roadmap`**: Publishes a specified artifact as a roadmap issue (prompts for file).

### `AGENTS.yaml` (Environment Tools)
- `sync_github`: `uv run python AgentQMS/tools/utils/sync_github_projects.py`
- `init_github`: `uv run python AgentQMS/tools/utils/sync_github_projects.py --init`
