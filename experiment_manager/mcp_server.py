#!/usr/bin/env python3
"""
Experiment Manager MCP Server

Exposes experiment lifecycle management tools and resources.

Resources:
- experiments://agent_interface - Command reference
- experiments://active_list - List of active experiments
- experiments://schemas/manifest - Manifest JSON schema
- experiments://schemas/artifact - Artifact JSON schema

Tools:
- init_experiment - Initialize new experiment
- get_experiment_status - Get experiment status
- add_task - Add task to experiment
- log_insight - Log insight/decision/failure
- sync_experiment - Sync to database
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool


# Auto-discover project root
def find_project_root() -> Path:
    """Find project root by locating experiment_manager/ directory."""
    current = Path(__file__).resolve().parent

    # We're already in experiment_manager/
    if current.name == "experiment_manager":
        return current.parent

    # Search upward
    for parent in current.parents:
        if (parent / "experiment_manager").exists():
            return parent

    raise RuntimeError("Cannot find project root with experiment_manager/")


PROJECT_ROOT = find_project_root()
EXPERIMENT_MANAGER_DIR = PROJECT_ROOT / "experiment_manager"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Add to Python path
sys.path.insert(0, str(EXPERIMENT_MANAGER_DIR / "src"))


# Define available resources
RESOURCES = [
    {
        "uri": "experiments://agent_interface",
        "name": "Agent Interface Commands",
        "description": "Complete ETK command reference with usage examples",
        "path": EXPERIMENT_MANAGER_DIR / "agent_interface.yaml",
        "mimeType": "application/x-yaml",
    },
    {
        "uri": "experiments://active_list",
        "name": "Active Experiments",
        "description": "List of current experiments (dynamically generated)",
        "path": None,  # Dynamic
        "mimeType": "application/json",
    },
    {
        "uri": "experiments://schemas/manifest",
        "name": "Manifest Schema",
        "description": "Experiment manifest JSON schema",
        "path": EXPERIMENT_MANAGER_DIR / ".schemas" / "manifest.schema.json",
        "mimeType": "application/json",
    },
    {
        "uri": "experiments://schemas/artifact",
        "name": "Artifact Schema",
        "description": "Experiment artifact JSON schema",
        "path": EXPERIMENT_MANAGER_DIR / ".schemas" / "artifact.schema.json",
        "mimeType": "application/json",
    },
]


# Create MCP server
app = Server("experiments")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List all available experiment resources."""
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
    """Read content of an experiment resource."""
    # Find matching resource
    resource = next((r for r in RESOURCES if r["uri"] == uri), None)

    if not resource:
        raise ValueError(f"Unknown resource URI: {uri}")

    # Handle dynamic active experiments list
    if uri == "experiments://active_list":
        return await _get_active_experiments()

    # Handle file-based resources
    path: Path = resource["path"]

    if not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")

    content = path.read_text(encoding="utf-8")
    return content


async def _get_active_experiments() -> str:
    """Get list of active experiments."""
    try:
        if not EXPERIMENTS_DIR.exists():
            return json.dumps({"experiments": []}, indent=2)

        experiments = []
        for exp_dir in EXPERIMENTS_DIR.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                manifest_path = exp_dir / "manifest.json"
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text())
                        experiments.append({
                            "id": exp_dir.name,
                            "name": manifest.get("name", exp_dir.name),
                            "status": manifest.get("status", "unknown"),
                            "created": manifest.get("created_at", "unknown"),
                        })
                    except:
                        # Skip malformed manifests
                        pass

        return json.dumps({"experiments": experiments}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available experiment management tools."""
    return [
        Tool(
            name="init_experiment",
            description="Initialize a new experiment with standardized directory structure",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique experiment identifier (slug format, e.g., 'baseline-kie-v2')",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional experiment description",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags (e.g., 'optimization,vlm')",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="get_experiment_status",
            description="Get detailed status of current or specific experiment",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment ID (optional, uses current if not specified)",
                    },
                },
            },
        ),
        Tool(
            name="add_task",
            description="Add a task to the active experiment plan",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Task description",
                    },
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment ID (optional, uses current if not specified)",
                    },
                },
                "required": ["description"],
            },
        ),
        Tool(
            name="log_insight",
            description="Log a key finding, decision, or failure to the experiment",
            inputSchema={
                "type": "object",
                "properties": {
                    "insight": {
                        "type": "string",
                        "description": "The insight, decision, or failure to log",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["insight", "decision", "failure"],
                        "description": "Type of log entry",
                    },
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment ID (optional, uses current if not specified)",
                    },
                },
                "required": ["insight"],
            },
        ),
        Tool(
            name="sync_experiment",
            description="Sync experiment artifacts and metadata to database",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment ID (optional, uses current if not specified)",
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute an experiment management tool."""
    try:
        if name == "init_experiment":
            exp_name = arguments["name"]
            cmd = ["uv", "run", "python3", "-m", "etk.factory", "init", "--name", exp_name]

            if "description" in arguments:
                cmd.extend(["-d", arguments["description"]])
            if "tags" in arguments:
                cmd.extend(["-t", arguments["tags"]])

            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "success": True,
                            "experiment_id": exp_name,
                            "output": result.stdout,
                        }, indent=2)
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": result.stderr,
                        }, indent=2)
                    )
                ]

        elif name == "get_experiment_status":
            exp_id = arguments.get("experiment_id", "")
            cmd = ["uv", "run", "python3", "-m", "etk.factory", "status"]
            if exp_id:
                cmd.append(exp_id)

            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "success": result.returncode == 0,
                        "output": result.stdout if result.returncode == 0 else result.stderr,
                    }, indent=2)
                )
            ]

        elif name == "add_task":
            description = arguments["description"]
            exp_id = arguments.get("experiment_id")

            # Use script directly for task management
            script_path = EXPERIMENT_MANAGER_DIR / "scripts" / "add-task.py"
            cmd = ["uv", "run", "python3", str(script_path), description]

            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "success": result.returncode == 0,
                        "output": result.stdout if result.returncode == 0 else result.stderr,
                    }, indent=2)
                )
            ]

        elif name == "log_insight":
            insight = arguments["insight"]
            log_type = arguments.get("type", "insight")

            cmd = ["uv", "run", "python3", "-m", "etk.factory", "log",
                   "--msg", insight, "--type", log_type]

            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "success": result.returncode == 0,
                        "output": result.stdout if result.returncode == 0 else result.stderr,
                    }, indent=2)
                )
            ]

        elif name == "sync_experiment":
            exp_id = arguments.get("experiment_id", "")
            cmd = ["uv", "run", "python3", "-m", "etk.factory", "sync"]
            if exp_id:
                cmd.append(exp_id)

            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "success": result.returncode == 0,
                        "output": result.stdout if result.returncode == 0 else result.stderr,
                    }, indent=2)
                )
            ]

        else:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Unknown tool: {name}",
                    }, indent=2)
                )
            ]

    except Exception as e:
        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "error": str(e),
                    "tool": name,
                }, indent=2)
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
