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

### AgentQMS: Plugin Artifact Types Resource

The unified server exposes dynamic artifact type discovery from AgentQMS plugins:

- **URI**: agentqms://plugins/artifact_types
- **MIME**: application/json
- **Purpose**: Lists all discoverable artifact types (hardcoded + plugins) with metadata, summary counts, and validation info.

Example usage via MCP client:

1. List resources and confirm the URI exists.
2. Read the resource and parse JSON fields: `artifact_types`, `summary`, `metadata`.

This resource mirrors the AgentQMS MCP handler and ensures consistent discovery through the unified server.

## Adding a New Server
1. Add the server definition to `scripts/mcp/shared_config.json`.
2. Run `uv run python scripts/mcp/sync_configs.py`.
3. Clear your agent's cache or restart the agent to see the new server.

---

## Known Issues and Dependencies

### AgentQMS Import Dependencies

**Status:** Expected limitation (separate package)

The `unified_server.py` imports from `AgentQMS.tools.core.context_bundle`:
```python
from AgentQMS.tools.core.context_bundle import (
    auto_suggest_context,
    list_available_bundles,
    get_context_bundle
)
```

**Impact:**
- These imports will fail if AgentQMS is not installed as a separate package
- The unified server gracefully handles missing AgentQMS features
- Core MCP functionality works without AgentQMS

**Resolution:**
- ✅ Expected behavior - AgentQMS is a separate package/module
- ✅ Not a blocker for core OCR functionality
- ✅ Import errors are documented and expected

**If AgentQMS features are needed:**
1. Ensure AgentQMS is installed and importable
2. Or modify unified_server.py to make AgentQMS features optional with try/except

**Last Updated:** 2026-01-29 (Audit: import-script-audit-2026-01-29)
