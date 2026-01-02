# Project Compass MCP Server

## Overview

The Project Compass MCP (Model Context Protocol) server exposes key project state and configuration files as standardized resources that can be easily discovered and accessed by AI assistants.

## Purpose

This MCP server improves discoverability of project metadata by:
- Providing a standardized interface to access project state
- Eliminating the need to know exact file paths
- Offering descriptive metadata for each resource
- Enabling consistent access patterns across different AI tools

## Available Resources

The server exposes the following resources using the `compass://` URI scheme:

### 1. `compass://compass.json`
**Main Project State**
- Current project phase (e.g., "kie", "ocr")
- Overall health status ("healthy", "degraded", etc.)
- Reference to active handoff document
- UV version constraints

### 2. `compass://session_handover.md`
**Current Session Handover**
- Recent accomplishments
- Current state of ongoing work
- Next steps and blockers
- Session-specific notes and context

### 3. `compass://current_session.yml`
**Active Session Context**
- Session ID and timestamps
- Current objective and goals
- Active pipeline being worked on
- Environment lock snapshot

### 4. `compass://uv_lock_state.yml`
**Environment Lock State**
- UV binary path and version
- Python version in use
- CUDA/GPU configuration
- Torch version and constraints

### 5. `compass://agents.yaml`
**Agent Configuration**
- AI agent rules and protocols
- Entry point commands (env check, session init, etc.)
- Schema root path
- Consumption metrics

## Usage

### Listing Available Resources

Use the `list_resources` MCP tool to see all available resources:

```python
list_resources(ServerName="project_compass")
```

This returns metadata about each resource including:
- URI
- Human-readable name
- Description of contents
- MIME type

### Reading Resource Content

Use the `read_resource` MCP tool to access resource content:

```python
read_resource(
    ServerName="project_compass",
    Uri="compass://compass.json"
)
```

## Technical Details

### Server Implementation

- **Location**: `project_compass/mcp_server.py`
- **Protocol**: Model Context Protocol (MCP)
- **Transport**: stdio
- **Language**: Python 3.11+
- **Dependencies**: `mcp` Python SDK

### Configuration

The server is registered in `/home/vscode/.gemini/antigravity/mcp_config.json`:

```json
{
  "mcpServers": {
    "project_compass": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/workspaces/upstageailab-ocr-recsys-competition-ocr-2",
        "python",
        "project_compass/mcp_server.py"
      ]
    }
  }
}
```

### Auto-Discovery

The server automatically discovers the project root by:
1. Starting from its own location (`project_compass/mcp_server.py`)
2. Searching upward for a directory containing `project_compass/`
3. Using that as the project root for all resource paths

This makes the server portable and resilient to being run from different working directories.

## Error Handling

The server handles common errors gracefully:

- **Unknown URI**: Returns an error if the requested URI is not in the resource list
- **Missing File**: Returns a clear error if the resource file doesn't exist on disk
- **Invalid Path**: Validates all paths are within the project compass directory

## Development

### Testing the Server

Test that the server starts correctly:

```bash
uv run python project_compass/mcp_server.py
```

The server will wait for stdio input (MCP protocol messages). Use Ctrl+C to exit.

### Adding New Resources

To add a new resource:

1. Add an entry to the `RESOURCES` list in `mcp_server.py`:
   ```python
   {
       "uri": "compass://new_resource.yml",
       "name": "Human Readable Name",
       "description": "What this resource contains",
       "path": COMPASS_DIR / "path" / "to" / "file.yml",
       "mimeType": "application/x-yaml",
   }
   ```

2. Ensure the file exists in the `project_compass/` directory structure

3. Update this documentation with the new resource details

### MIME Types

Common MIME types used:
- `application/json` - JSON files
- `application/x-yaml` - YAML files
- `text/markdown` - Markdown documents
- `text/plain` - Plain text files

## Troubleshooting

### Server Not Found Error

If you see `MCP server not found: project_compass`:
1. Check that the server is registered in `mcp_config.json`
2. Verify the path to `mcp_server.py` is correct
3. Restart the AI assistant to reload MCP configuration

### Resource Not Found

If a resource can't be read:
1. Verify the file exists at the expected path
2. Check file permissions
3. Ensure the project root is correctly detected

### Server Won't Start

If the server fails to start:
1. Ensure `mcp` Python package is installed: `uv add mcp`
2. Check for Python syntax errors in `mcp_server.py`
3. Verify Python version compatibility (3.11+)
