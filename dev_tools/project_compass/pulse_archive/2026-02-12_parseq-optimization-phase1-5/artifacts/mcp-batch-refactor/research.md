# MCP Batch Refactor Research Notes

Date: 2026-02-12

## Observations
- Unified MCP dispatcher (scripts/mcp/unified_server.py) routes all tools and currently has no concurrency policy or timeouts.
- Project Compass and ADT MCP servers execute tool handlers directly; Compass mutates shared state files.
- The reported async sibling tool call error is consistent with batched tool execution without isolation.

## Tool Call Failure: suggest_context.py
Observed error:
  uv run python AgentQMS/tools/utilities/suggest_context.py "recognition optimization"
  /workspaces/.venv/bin/python3: can't open file '/workspaces/AgentQMS/tools/utilities/suggest_context.py': [Errno 2] No such file or directory

Resolution:
- The script now lives under AgentQMS/tools/core/context/suggest_context.py.
- Use:
  uv run python AgentQMS/tools/core/context/suggest_context.py "recognition optimization"

## Recommended Approach
- Centralize concurrency policy in the unified dispatcher with per-module limits.
- Serialize Compass state writes and staging exports.
- Offload ADT analyzers to background threads to avoid event-loop blocking.
- Emit deterministic ordering metadata in telemetry for troubleshooting.
