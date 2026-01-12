---
type: implementation_plan
category: development
status: active
version: 1.0
tags: mcp,adt,bundles,bugfix
ads_version: 1.0
artifact_type: implementation_plan
title: Repair MCP Tools and Context Bundles
date: 2026-01-13 03:57 (KST)
branch: main
description: Repair ADT MCP tools and Context Bundle resources in Unified Server
---

# Repair MCP Tools and Context Bundles

The goal is to fix the integration of Agent Debug Toolkit (ADT) tools and Context Bundles within the `unified_project` MCP server. Currently, ADT tools fail due to missing implementation metadata, and context bundle resources fail due to a routing bug in the resource handler.

## User Review Required

> [!IMPORTANT]
> This change impacts how ADT tools are dispatched and how context bundles are accessed. Specifically, `adt_meta_query` and `adt_meta_edit` will now correctly route to their underlying analyzers/editors.

## Proposed Changes

### Agent Debug Toolkit

---

#### [MODIFY] [tools.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/config/tools.yaml)

Add `implementation` blocks to `adt_meta_query` and `adt_meta_edit` to ensure they are dispatched to the ADT MCP server implementation.

```yaml
implementation:
  module: agent_debug_toolkit.mcp_server
  function: call_tool
```

### Unified MCP Server

---

#### [MODIFY] [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py)

1.  **Fix Resource Routing**: Reorder the logic in `read_resource` to check dynamic handlers (like `bundle://`) BEFORE raising a `ValueError` for unknown URIs.
2.  **Enhance Context Bundle Resources**:
    -   `bundle://list`: Return a list of available context bundles.
    -   `bundle://<name>`: Return a summary of files in the specified bundle.

## Verification Plan

### Automated Tests
- Run `mcp_unified_project_get_server_info` to verify tool availability.
- Call `mcp_unified_project_adt_meta_query` with `kind: "symbol_search", target: "ToolRegistry"` to verify ADT dispatch.
- Read `bundle://list` resource to verify bundle discovery.
- Read `bundle://development` to verify bundle content retrieval.

### Manual Verification
- Verify that the agent receives context suggestions (💡 **Context Suggestion**) when using other tools like `get_server_info`.
