# Shared MCP Architecture

This project uses a unified Model Context Protocol (MCP) server and a centralized configuration system to share tools across multiple AI agents (Gemini, Claude, etc.).

## Shared Configuration Workflow

The "Source of Truth" for your MCP servers is located in the project itself, making it easy to version and share:

1.  **Shared Config**: [scripts/mcp/shared_config.json](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/shared_config.json)
2.  **Sync Script**: [scripts/mcp/sync_configs.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/sync_configs.py)

### How to use it:
To update all your AI agents with the latest server definitions:
```bash
python3 scripts/mcp/sync_configs.py
```
This script automatically propagates the shared configuration to:
- `~/.gemini/antigravity/mcp_config.json`
- `~/.claude/config.json`

## Unified Project Server

Instead of running multiple separate MCP servers, we use a single **Unified Server** to minimize resource overhead.

- **Path**: [scripts/mcp/unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py)
- **Servers Combined**: Project Compass, AgentQMS, Experiment Manager.

### Benefits:
- **Efficiency**: Reduces memory and CPU usage by running one process instead of three+.
- **Standardization**: All agents have access to the exact same tools and resources.
- **Portability**: Config is stored in the repo, not hidden in agent-specific home directories.

## Adding a New Server
1. Add the server definition to `scripts/mcp/shared_config.json`.
2. Run `python3 scripts/mcp/sync_configs.py`.
3. Clear your agent's cache or restart the agent to see the new server.
