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
import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent
from mcp.server.lowlevel.helper_types import ReadResourceContents


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


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="get_server_info",
            description="Get information about the Project Compass MCP server",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="manage_session",
            description="Manage project sessions (export, import, list, new) to save/restore context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["export", "import", "list", "new"],
                        "description": "Action to perform on the session."
                    },
                    "session_name": {
                        "type": "string",
                        "description": "Name of the session to import (required for import action)."
                    },
                    "note": {
                        "type": "string",
                        "description": "Note to attach to the session (optional for export)."
                    }
                },
                "required": ["action"]
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute a tool."""
    if name == "get_server_info":
        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "name": "project_compass",
                    "version": "0.1.0",
                    "status": "running"
                }, indent=2)
            )
        ]

    if name == "manage_session":
        import subprocess
        action = arguments.get("action")
        session_name = arguments.get("session_name")
        note = arguments.get("note")

        cmd = ["uv", "run", "python", str(COMPASS_DIR / "scripts/session_manager.py"), action]

        if action == "export" and note:
            cmd.extend(["--note", note])
        elif action == "import":
            if not session_name:
                raise ValueError("session_name is required for import action")
            cmd.append(session_name)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            if result.returncode != 0:
                output = f"Error executing session manager:\n{result.stderr}\n{result.stdout}"
            else:
                output = result.stdout

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Failed to execute session manager: {str(e)}")]

    raise ValueError(f"Unknown tool: {name}")


@app.read_resource()
async def read_resource(uri: str) -> list[ReadResourceContents]:
    """Read content of a compass resource."""
    # Find matching resource
    uri = str(uri).strip()
    resource = next((r for r in RESOURCES if r["uri"] == uri), None)

    if not resource:
        raise ValueError(f"Unknown resource URI: {repr(uri)}. Available: {[r['uri'] for r in RESOURCES]}")

    path: Path = resource["path"]

    if not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")

    # Read and return content wrapped in proper MCP types
    content = path.read_text(encoding="utf-8")

    return [
        ReadResourceContents(
            content=content,
            mime_type=resource["mimeType"]
        )
    ]


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
