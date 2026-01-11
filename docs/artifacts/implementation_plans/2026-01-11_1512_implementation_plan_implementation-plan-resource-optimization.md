---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: "None"
title: "Unified Server Resource Optimization"
date: "2026-01-11 15:12 (KST)"
branch: "main"
description: "Plan to optimize unified_server.py resource reading."
---

# Implementation Plan: Unified Server Resource Optimization

## Goal
Optimize the `read_resource` function in `scripts/mcp/unified_server.py` to reduce file system I/O overhead by using pre-computed alias maps, as recommended in the Code Review Assessment.

## User Review Required
> [!IMPORTANT]
> This change involves modifying the core resource resolution logic. While tested in theory, it should be verified that server startup time is not negatively impacted by the pre-computation of paths.

## Proposed Changes

### 1. Configuration Externalization (New)
To reduce cognitive load and file size, we will externalize definitions to YAML files in `scripts/mcp/config/`.

#### [NEW] [resources.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/config/resources.yaml)
- Move the entire `RESOURCES_CONFIG` list here.
- Format: List of resource objects.

#### [NEW] [tools.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/config/tools.yaml)
- Move all `Tool(...)` definitions here.
- Format: Dictionary or List of tool definitions including `name`, `description`, and `inputSchema`.

### 2. Unified Server Logic (`unified_server.py`)

#### [MODIFY] [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py)
- **Imports**: Add `yaml` (already present).
- **Initialization**:
    - Load `resources.yaml` -> `RESOURCES_CONFIG`.
    - Load `tools.yaml` -> `TOOLS_DEFINITIONS`.
    - **Optimization**: Build `URI_MAP` and `PATH_MAP` from the loaded resources (with `Path(None)` fix).
- **`list_tools()`**:
    - Return tools from `TOOLS_DEFINITIONS`, filtering by `is_tool_enabled`.
- **`read_resource()`**:
    - Use the optimized `URI_MAP` and `PATH_MAP` logic.
- **`call_tool()`**:
    - Implementation remains in `unified_server.py` (for now) to avoid major behavioral regressions, but with reduced clutter around it.

## Verification Plan

### Automated Tests
- Restart the MCP server to ensure no crash on startup (verifies `Path(None)` fix).
- Test exact URI resolution (e.g., `compass://compass.json`).
- Test alias resolution (e.g., `file:///absolute/path/to/compass.json`).
- Test virtual URIs (e.g., `agentqms://templates/list`).

### Manual Verification
- Monitor server logs for errors during the first few requests.
