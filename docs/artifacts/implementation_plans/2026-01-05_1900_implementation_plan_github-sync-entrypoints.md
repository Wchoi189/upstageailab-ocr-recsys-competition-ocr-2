---
ads_version: "1.0"
type: implementation_plan
category: planning
status: approved
version: "1.0"
tags: github, sync, entrypoints, vscode, agents
title: Implementation Plan - GitHub Sync Entrypoints
date: "2026-01-05 18:54 (KST)"
branch: main
---

# Implementation Plan - GitHub Sync Entrypoints

This plan outlines the steps to expose the new GitHub Sync features (`--init`, `--roadmap`) via VS Code Tasks and `AGENTS.yaml`, making them easily accessible to the user and agents.

## User Review Required

> [!NOTE]
> **VS Code Inputs**: I will add new inputs to `.vscode/tasks.json` to allow the user to interactively select the roadmap file when running the task.

## Proposed Changes

### Configuration

#### [MODIFY] [.vscode/tasks.json](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/.vscode/tasks.json)

- **New Task**: `🎯 AgentQMS: Init GitHub Project`
    - Command: `uv run python AgentQMS/tools/utils/sync_github_projects.py --init`
    - Optional: Prompt for roadmap to publish during init? Yes, let's keep it simple and just do init, or maybe init + roadmap if selected.
    - Let's make it flexible: `uv run python AgentQMS/tools/utils/sync_github_projects.py --init --roadmap ${input:roadmapFile}`
- **New Task**: `🎯 AgentQMS: Publish Roadmap`
    - Command: `uv run python AgentQMS/tools/utils/sync_github_projects.py --roadmap ${input:roadmapFile}`
- **New Input**: `roadmapFile`
    - Type: `promptString`
    - Description: "Path to roadmap artifact (e.g. docs/artifacts/plans/...)"
    - Default: "docs/artifacts/implementation_plans/"

#### [MODIFY] [AGENTS.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AGENTS.yaml)

- Add commands to `environment_tools.agent_qms.commands`:
    - `sync_github`: `uv run python AgentQMS/tools/utils/sync_github_projects.py`
    - `init_github`: `uv run python AgentQMS/tools/utils/sync_github_projects.py --init`

## Verification Plan

### Automated Verification
- None (Config changes only).

### Manual Verification
- **Code Review**: Verify the JSON/YAML structure is valid.
- **User Test**: User will test the new tasks via VS Code UI.
