---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: "None"
title: "Automated MCP Tool Synchronization"
date: "2026-01-11 20:14 (KST)"
branch: "main"
description: "Plan to automate MCP tool synchronization via a generator script and dynamic dispatcher."
---

# Automated MCP Tool Synchronization

## Goal Description
Automate the synchronization of MCP tools between component servers (AgentQMS, Agent Debug Toolkit, Experiment Manager) and the central `unified_server.py`. This resolves the "Synchronization Failure" issue where developers must manually update `unified_server.py` logic when adding tools to components.

The solution uses a "Build-time Aggregation, Runtime Dispatch" model:
1.  **Aggregator Script (`refresh_tools.py`)**: Programmatically imports component servers, extracts tool definitions, and generates a consolidated `tools.yaml` with execution metadata.
2.  **Dynamic Dispatcher**: Refactors `unified_server.py` to genericize `call_tool`, using the `tools.yaml` metadata to route calls to the correct component module dynamically.

## User Review Required
> [!IMPORTANT]
> This refactor changes how `unified_server.py` executes tools. Instead of hardcoded logic, it will rely on the `config/tools.yaml` file. If that file is out of sync, tools will fail. The `refresh_tools.py` script becomes a required step in the development workflow (e.g., `make refresh-mcp`).

## Proposed Changes

### 1. Tool Aggregator
#### [NEW] [scripts/mcp/refresh_tools.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/refresh_tools.py)
-   Imports `list_tools` from:
    -   `AgentQMS.mcp_server`
    -   `experiment_manager.mcp_server`
    -   `agent_debug_toolkit.mcp_server`
    -   `project_compass.mcp_server`
-   Executes them to get the current schema.
-   Adds `implementation` metadata:
    ```yaml
    implementation:
      module: "AgentQMS.mcp_server"
      function: "call_tool"
    ```
-   Writes `scripts/mcp/config/tools.yaml`.
-   Updates `scripts/mcp/mcp_tools_config.yaml` (groups) automatically.

### 2. Unified Server Refactor
#### [MODIFY] [scripts/mcp/unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py)
-   **Remove** the massive `if/else` block in `call_tool`.
-   **Implement** dynamic dispatch:
    -   Load `tools.yaml`.
    -   Lookup tool by name.
    -   Import `implementation.module`.
    -   Call `call_tool(name, args)` on that module.

### 3. Workflow Integration
#### [MODIFY] [Makefile](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/Makefile)
-   Add `refresh-mcp` target running `uv run python scripts/mcp/refresh_tools.py`.

## Verification Plan

### Automated Tests
-   **Run the Refresh Script**: Verify `tools.yaml` is generated correctly.
-   **Verify Dispatch**: Use `verify_server.py` (existing) to ensure generic tools still work.
-   **Integration Test**: Call a tool from each component (e.g., `create_artifact`, `trace_merge_order`, `init_experiment`) via `unified_server.py` and verify functionality.

### Manual Verification
-   Inspect `tools.yaml` to check for the new `implementation` field.
