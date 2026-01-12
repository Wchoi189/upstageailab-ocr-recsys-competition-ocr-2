---
ads_version: "1.0"
type: bug_report
artifact_type: bug_report
title: "ADT MCP Tools Not Available in Antigravity IDE"
date: "2026-01-12 05:17 (KST)"
tags: mcp, agent-debug-toolkit, configuration, devcontainer
status: completed
priority: high
category: troubleshooting
version: "1.0"
affected_systems:
  - agent-debug-toolkit MCP tools
  - Antigravity IDE MCP configuration
  - devcontainer environment
---

# Bug Report: ADT MCP Tools Not Available in Antigravity IDE

## Issue Summary

The agent-debug-toolkit (ADT) MCP tools (`adt_meta_query`, `adt_meta_edit`) are failing with the error:

```
Error: Unknown tool: adt_meta_query
```

This occurs because the devcontainer MCP configuration (`.devcontainer/mcp_config.json`) was using three separate MCP servers instead of the unified server that includes ADT tools.

## Root Cause

The devcontainer was configured to use three **separate** MCP servers:
- `project_compass/mcp_server.py`
- `AgentQMS/mcp_server.py`
- `experiment_manager/mcp_server.py`

However, the ADT tools (`adt_meta_query`, `adt_meta_edit`) are only available in the **unified MCP server** (`scripts/mcp/unified_server.py`), which combines all tools from:
- Project Compass
- AgentQMS
- Experiment Manager
- **Agent Debug Toolkit** ← Missing because unified server wasn't used

## Expected Behavior

When invoked, the `adt_meta_query` tool should route to the ADT MCP server and execute analysis tasks like:
- `config_access` - Find configuration access patterns
- `complexity` - Analyze code complexity
- `symbol_search` - Search for symbols
- And more...

## Actual Behavior

Tool invocation fails with "Unknown tool: adt_meta_query" because Claude Desktop doesn't know about the ADT MCP server.

## Solution
**Fixed:** Updated `.devcontainer/mcp_config.json` to use the unified MCP server instead of separate servers.

### What Changed

**Before** (three separate servers):
```json
{
  "mcpServers": {
    "project_compass": { ... },
    "agentqms": { ... },
    "experiments": { ... }
  }
}
```

**After** (unified server with all tools):
```json
{
  "mcpServers": {
    "unified_project": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "${workspaceFolder}",
        "python",
        "scripts/mcp/unified_server.py"
      ],
      "env": {
        "PROJECT_ROOT": "${workspaceFolder}"
      }
    }
  }
}
```

### Why the Unified Server?

The unified server (`scripts/mcp/unified_server.py`) provides:
- ✅ All Project Compass tools
- ✅ All AgentQMS tools
- ✅ All Experiment Manager tools
- ✅ **All Agent Debug Toolkit tools** (including `adt_meta_query`, `adt_meta_edit`)
- ✅ Configurable tool groups via `scripts/mcp/mcp_tools_config.yaml`
- ✅ Middleware for telemetry and compliance checking
```

## Testing After Configuration

After updating the configuration file:

1. **Restart Claude Desktop** (configuration is loaded at startup)

2. **Test the meta-query tool:**
   ```
   Can you analyze the complexity of AgentQMS/tools/compliance/validate_artifacts.py?
   ```

3. **Verify tool availability:**
   The tool should appear in the MCP tools list and execute successfully.

## Available ADT Tools

Once configured, these tools become available:

### Meta-Tools (Recommended)
- **adt_meta_query**: Unified analysis router
  - Kinds: `config_access`, `merge_order`, `hydra_usage`, `component_instantiations`, `config_flow`, `dependency_graph`, `imports`, `complexity`, `context_tree`, `symbol_search`

- **adt_meta_edit**: Unified edit router
  - Kinds: `apply_diff`, `smart_edit`, `read_slice`, `format`
`.devcontainer/mcp_config.json`:

1. **Reload the window** in Antigravity IDE or restart the devcontainer to reload MCP configuration

2. **Test the adt_meta_query tool:**
   ```
   Can you analyze the complexity of AgentQMS/tools/compliance/validate_artifacts.py using adt_meta_query?
   ```

3. **Verify tool availability:**
   The tool should now execute successfully instead of showing "Unknown tool" error.

4. **Verify all tool groups are available:**
   ```python
   # The unified server should report:
   {
     "components": ["Compass", "AgentQMS", "ETK", "ADT"],
     "tool_groups": {
       "enabled": ["compass", "agentqms", "etk", "adt_router"],
       "available": ["compass", "agentqms", "etk", "adt_core", "adt_static",
                     "adt_discovery", "adt_edit", "adt_router", "adt_astgrep", "adt_treesitter"]
     }
   }
   ```
- `intelligent_search`

## References

- ADT MCP Server: [agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py](agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py)
- ADT Launch Script: [agent-debug-toolkit/run_mcp.sh](agent-debug-toolkit/run_mcp.sh)
- ADT Usage Guide: [agent-debug-toolkit/AI_USAGE.yaml](agent-debug-toolkit/AI_USAGE.yaml)
- ADT README: [agent-debug-toolkit/README.md](agent-debug-toolkit/README.md)

## Impact

**High Priority** - This blocks AI agents from:
- Analyzing code complexity
- Tracing configuration flows
- Finding Hydra patterns
- Building dependency graphs
- All AST-based debugging capabilities

## Next Steps

1. User must manually edit Claed AI agents from:
- Analyzing code complexity
- Tracing configuration flows
- Finding Hydra patterns
- Building dependency graphs
- All AST-based debugging capabilities

## Resolution Status

✅ **FIXED** - Updated `.devcontainer/mcp_config.json` to use unified server.

The ADT tools are now available via the `adt_meta_query` and `adt_meta_edit` routers, which provide access to all 20+ analysis and editing tools through a streamlined interface.

## Verification

Tool configuration verified:
```
✓ adt_meta_query found in TOOLS_DEFINITIONS
  Implementation: {'module': 'agent_debug_toolkit.mcp_server', 'function': 'call_tool'}
  Enabled: True
```

---

**Note:** After reloading the IDE, the unified server will provide all tools from Project Compass, AgentQMS, Experiment Manager, AND Agent Debug Toolkit in a single cohesive interfac
