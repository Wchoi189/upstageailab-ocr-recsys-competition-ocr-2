---
type: implementation_plan
category: feature
status: active
version: "1.0"
tags: github, sync, extension
title: Implementation Plan - Github Sync Extension
date: "2026-01-05 18:54"
branch: main
---

# Implementation Plan - Github Sync Extension

This plan outlines the steps to extend the `sync_github_projects.py` script to support comprehensive GitHub Project management, including initialization, roadmap publishing, and robust synchronization.

## User Review Required

> [!IMPORTANT]
> **Configuration Strategy**: This plan assumes `project_compass/state/current_session.yml` (or similar persistent session state) will be used to store the `github_project_id` and `roadmap_issue_url` after they are created/identified.

> [!WARNING]
> **GitHub CLI Dependency**: The implementation strictly relies on `gh` CLI being installed and authenticated with `project` scope.

## Proposed Changes

### AgentQMS Tools

#### [MODIFY] [sync_github_projects.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/sync_github_projects.py)

- **Refactor**: Convert the script to use a `GitHubManager` class for better encapsulation.
- **New Feature**: Add `--init --roadmap <file>` arguments.
    - If `--init` is passed, check if a project exists for the current session.
    - If not, create a new GitHub Project (V2).
    - If `--roadmap` is passed, read the artifact and post it as an Issue.
- **Enhancement**: Implement robust existence checks.
    - `find_project_by_title(title)`: JSON-based query using `gh project list`.
    - `find_issue_by_title(title)`: Search issues to avoid duplicates.
- **Enhancement**: Improve error handling (try/except blocks around `subprocess` calls).

## Verification Plan

### Automated Tests
- **Dry Run verification**:
    ```bash
    python AgentQMS/tools/utils/sync_github_projects.py --init --roadmap docs/artifacts/implementation_plans/2026-01-05_github_sync_extension_plan.md --dry-run
    ```
    *Expectation*: Output should show commands that WOULD be executed, without errors.

### Manual Verification
- **Live Sync Test**:
    1. Run the script with a real session.
    2. Verify a new Project is created in the user's GitHub.
    3. Verify the Roadmap artifact is posted as an issue.
