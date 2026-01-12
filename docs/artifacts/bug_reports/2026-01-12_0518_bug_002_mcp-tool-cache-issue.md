---
ads_version: "1.0"
type: bug_report
artifact_type: bug_report
title: "MCP Client Not Seeing Updated Tool List"
date: "2026-01-12 05:18 (KST)"
tags: mcp, antigravity, cursor, tool-cache
status: active
priority: high
category: troubleshooting
version: "1.0"
affected_systems:
  - Antigravity IDE MCP client
  - unified_project MCP server
---

# Bug Report: MCP Client Not Seeing Updated Tool List

## Issue Summary

Antigravity IDE reports "Unknown tool: adt_meta_query" even though the unified MCP server is running and correctly exposing the tool.

## Investigation Results

### ✅ Server is Working Correctly

```bash
# Server processes are running
$ ps aux | grep unified_server
vscode 13022 uv run --directory /workspaces/... python scripts/mcp/unified_server.py
vscode 13037 /workspaces/.../python3 scripts/mcp/unified_server.py
```

### ✅ Tools Are Defined and Enabled

```python
# Verified via direct test
✓ adt_meta_query found in TOOLS_DEFINITIONS
  Implementation module: agent_debug_toolkit.mcp_server
✓ Tool execution successful
```

### ✅ Server Exposes the Tools

```
Total tools exposed: 11

Tool names:
  - adt_meta_edit
  - adt_meta_query  ← Present!
  - check_compliance
  - create_artifact
  - get_server_info
  - get_standard
  - init_experiment
  - list_artifact_templates
  - log_insight
  - manage_session
  - validate_artifact
```

## Root Cause

**MCP client-side caching issue**. The Antigravity IDE (Cursor-based) has an older tool list cached from when it first connected to the MCP servers. The server was updated/restarted but the client hasn't refreshed its tool list.

## Solutions (in order of preference)

### Solution 1: Reload MCP Servers in IDE

**For Antigravity/Cursor IDE:**
1. Open Command Palette (Cmd/Ctrl+Shift+P)
2. Search for "MCP: Restart Servers" or "MCP: Reload"
3. OR: "Developer: Reload Window"

### Solution 2: Kill and Restart Server Processes

```bash
# Kill existing unified_server processes
pkill -f "unified_server.py"

# The IDE should auto-restart them, or manually:
# (Usually not needed - IDE restarts servers automatically)
```

### Solution 3: Completely Restart the IDE

Close and reopen Antigravity IDE / Cursor to force a full reconnection to MCP servers.

### Solution 4: Check MCP Configuration (if above doesn't work)

The configuration should point to the unified server. For Cursor/Antigravity, this is typically in:

**User-level config:**
- Linux: `~/.cursor/mcp_config.json` or `~/.config/cursor/mcp_config.json`
- macOS: `~/Library/Application Support/Cursor/mcp_config.json`
- Windows: `%APPDATA%\Cursor\mcp_config.json`

**Should contain:**
```json
{
  "mcpServers": {
    "unified_project": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/workspaces/upstageailab-ocr-recsys-competition-ocr-2",
        "python",
        "scripts/mcp/unified_server.py"
      ],
      "env": {
        "PROJECT_ROOT": "/workspaces/upstageailab-ocr-recsys-competition-ocr-2"
      }
    }
  }
}
```

## Verification Steps

After applying a solution:

1. **Check tool availability:** Ask the AI to use `adt_meta_query`
2. **Expected behavior:** Tool executes successfully
3. **Test command:**
   ```
   Use adt_meta_query to analyze the complexity of AgentQMS/tools/compliance/validate_artifacts.py
   ```

## Technical Details

### Server Configuration Files

- **Tool definitions:** `scripts/mcp/config/tools.yaml`
- **Tool groups:** `scripts/mcp/mcp_tools_config.yaml`
- **Server entrypoint:** `scripts/mcp/unified_server.py`

### Enabled Tool Groups

Current configuration enables:
- `compass` - 5 tools
- `agentqms` - 4 tools
- `etk` - 2 tools
- **`adt_router`** - 2 meta-tools (adt_meta_query, adt_meta_edit)

### Tool Routing

The `adt_meta_query` tool routes to these analysis kinds:
- `config_access` - Find cfg.X patterns
- `merge_order` - Trace OmegaConf.merge()
- `hydra_usage` - Find Hydra patterns
- `component_instantiations` - Track factories
- `config_flow` - Config flow summary
- `dependency_graph` - Module dependencies
- `imports` - Import analysis
- `complexity` - Code complexity metrics
- `context_tree` - Directory tree with context
- `symbol_search` - Symbol/path search

## Impact

**Medium-High Priority** - Tools are available but not accessible due to client caching. This blocks ADT functionality until the client refreshes its tool list.

## Status

**Diagnosed** - Server is working correctly. Issue is client-side cache. Awaiting user to reload IDE/servers.

---

**Next Action:** User should reload MCP servers in Antigravity IDE or restart the IDE completely.
