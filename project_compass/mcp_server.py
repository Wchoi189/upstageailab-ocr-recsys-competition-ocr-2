#!/usr/bin/env python3
"""
Project Compass MCP Server

Exposes project state and configuration files as MCP resources for improved
discoverability and standardized access by AI assistants.

Resources exposed:
- compass://compass.json - Main project state
- compass://session_handover.md - Current session handover
- compass://current_session.yml - Active session context
- compass://uv_lock_state.yml - Environment lock state
- compass://agents.yaml - Agent configuration
"""

import asyncio
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource


# Auto-discover project root
def find_project_root() -> Path:
    """Find project root by locating project_compass/ directory."""
    current = Path(__file__).resolve().parent

    # We're already in project_compass/
    if current.name == "project_compass":
        return current.parent

    # Search upward
    for parent in current.parents:
        if (parent / "project_compass").exists():
            return parent

    raise RuntimeError("Cannot find project root with project_compass/")


PROJECT_ROOT = find_project_root()
COMPASS_DIR = PROJECT_ROOT / "project_compass"


# Define available resources
RESOURCES = [
    {
        "uri": "compass://compass.json",
        "name": "Project Compass State",
        "description": "Main compass state: current phase, health, handoff reference",
        "path": COMPASS_DIR / "compass.json",
        "mimeType": "application/json",
    },
    {
        "uri": "compass://session_handover.md",
        "name": "Session Handover",
        "description": "Current session handover document with accomplishments and next steps",
        "path": COMPASS_DIR / "session_handover.md",
        "mimeType": "text/markdown",
    },
    {
        "uri": "compass://current_session.yml",
        "name": "Active Session Context",
        "description": "Active session metadata: objective, pipeline, environment lock",
        "path": COMPASS_DIR / "active_context" / "current_session.yml",
        "mimeType": "application/x-yaml",
    },
    {
        "uri": "compass://uv_lock_state.yml",
        "name": "Environment Lock State",
        "description": "UV environment lock state: Python version, CUDA config, dependencies",
        "path": COMPASS_DIR / "environments" / "uv_lock_state.yml",
        "mimeType": "application/x-yaml",
    },
    {
        "uri": "compass://agents.yaml",
        "name": "Agent Configuration",
        "description": "AI agent configuration: rules, entry points, schema root",
        "path": COMPASS_DIR / "AGENTS.yaml",
        "mimeType": "application/x-yaml",
    },
]


# Create MCP server
app = Server("project_compass")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List all available compass resources."""
    return [
        Resource(
            uri=res["uri"],
            name=res["name"],
            description=res["description"],
            mimeType=res["mimeType"],
        )
        for res in RESOURCES
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read content of a compass resource."""
    # Find matching resource
    resource = next((r for r in RESOURCES if r["uri"] == uri), None)

    if not resource:
        raise ValueError(f"Unknown resource URI: {uri}")

    path: Path = resource["path"]

    if not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")

    # Read and return content
    content = path.read_text(encoding="utf-8")
    return content


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
