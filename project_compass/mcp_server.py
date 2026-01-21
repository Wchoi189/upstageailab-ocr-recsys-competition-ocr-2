#!/usr/bin/env python3
"""
Project Compass V2 - Vessel MCP Server

Exposes vessel state and pulse tools via Model Context Protocol.

Resources:
- vessel://state - Current vessel state (THE Source of Truth)
- vessel://rules - Injected rules for active pulse
- vessel://staging - List of staging artifacts

Tools:
- pulse_init - Initialize a new pulse
- pulse_sync - Register staging artifact
- pulse_export - Archive pulse to history
- pulse_status - Get current pulse status
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
    if current.name == "project_compass":
        return current.parent
    for parent in current.parents:
        if (parent / "project_compass").exists():
            return parent
    raise RuntimeError("Cannot find project root with project_compass/")


PROJECT_ROOT = find_project_root()
COMPASS_DIR = PROJECT_ROOT / "project_compass"

# V2 Paths
VESSEL_DIR = COMPASS_DIR / ".vessel"
VESSEL_STATE = VESSEL_DIR / "vessel_state.json"
VAULT_DIR = COMPASS_DIR / "vault"
STAGING_DIR = COMPASS_DIR / "pulse_staging"


# Create MCP server
app = Server("vessel")


# ============ Resources ============

RESOURCES = [
    {
        "uri": "vessel://state",
        "name": "Vessel State",
        "description": "THE Single Source of Truth - current project and pulse state",
        "mimeType": "application/json",
    },
    {
        "uri": "vessel://rules",
        "name": "Active Rules",
        "description": "Injected rules for the current pulse (from vault)",
        "mimeType": "text/plain",
    },
    {
        "uri": "vessel://staging",
        "name": "Staging Artifacts",
        "description": "List of files in pulse_staging/artifacts/",
        "mimeType": "text/plain",
    },
]


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List all available vessel resources."""
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
async def read_resource(uri: str) -> list[ReadResourceContents]:
    """Read content of a vessel resource."""
    uri = str(uri).strip()

    if uri == "vessel://state":
        if not VESSEL_STATE.exists():
            content = json.dumps({"error": "No vessel_state.json found. Run pulse-init first."})
        else:
            content = VESSEL_STATE.read_text(encoding="utf-8")
        return [ReadResourceContents(content=content, mime_type="application/json")]

    elif uri == "vessel://rules":
        try:
            from project_compass.src.core import PulseManager
            from project_compass.src.rule_injector import get_injected_context

            manager = PulseManager()
            state = manager.load_state()
            content = get_injected_context(state)
        except Exception as e:
            content = f"Error loading rules: {e}"
        return [ReadResourceContents(content=content, mime_type="text/plain")]

    elif uri == "vessel://staging":
        artifacts_dir = STAGING_DIR / "artifacts"
        if not artifacts_dir.exists():
            content = "No staging directory. Create files in: project_compass/pulse_staging/artifacts/"
        else:
            files = [str(p.relative_to(artifacts_dir)) for p in artifacts_dir.rglob("*") if p.is_file()]
            if files:
                content = "\n".join(files)
            else:
                content = "(empty - no artifacts in staging)"
        return [ReadResourceContents(content=content, mime_type="text/plain")]

    raise ValueError(f"Unknown resource URI: {uri}")


# ============ Tools ============

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all pulse management tools."""
    return [
        Tool(
            name="pulse_init",
            description="Initialize a new pulse (work cycle). Requires pulse_id (domain-action-target format), objective (20-500 chars), and milestone_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pulse_id": {
                        "type": "string",
                        "description": "Pulse ID in domain-action-target format (e.g., 'recognition-optimize-vocab')",
                    },
                    "objective": {
                        "type": "string",
                        "description": "Work objective (20-500 characters)",
                    },
                    "milestone_id": {
                        "type": "string",
                        "description": "Milestone ID from star-chart (e.g., 'rec-opt')",
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["detection", "recognition", "kie", "integration"],
                        "description": "Active pipeline phase (default: kie)",
                    },
                },
                "required": ["pulse_id", "objective", "milestone_id"],
            },
        ),
        Tool(
            name="pulse_sync",
            description="Register a staging artifact in the manifest. File must exist in pulse_staging/artifacts/.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Artifact path relative to pulse_staging/artifacts/",
                    },
                    "artifact_type": {
                        "type": "string",
                        "enum": ["design", "research", "walkthrough", "implementation_plan", "bug_report", "audit"],
                        "description": "Type of artifact",
                    },
                    "milestone_id": {
                        "type": "string",
                        "description": "Override milestone ID (defaults to pulse milestone)",
                    },
                },
                "required": ["path", "artifact_type"],
            },
        ),
        Tool(
            name="pulse_export",
            description="Archive current pulse to history. Performs staging audit - blocks if unregistered files exist.",
            inputSchema={
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "Skip staging audit (NOT RECOMMENDED)",
                    },
                },
            },
        ),
        Tool(
            name="pulse_status",
            description="Get current pulse status including artifact count and token burden.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="pulse_checkpoint",
            description="Update token burden and get pulse maturity assessment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "token_burden": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Update token burden level",
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute a pulse tool."""
    try:
        from project_compass.src.core import PulseManager, VesselPaths
        from project_compass.src.pulse_exporter import export_pulse, register_artifact
    except ImportError:
        from src.core import PulseManager, VesselPaths
        from src.pulse_exporter import export_pulse, register_artifact

    paths = VesselPaths()
    manager = PulseManager(paths)

    if name == "pulse_init":
        success, message = manager.init_pulse(
            pulse_id=arguments["pulse_id"],
            objective=arguments["objective"],
            milestone_id=arguments["milestone_id"],
            phase=arguments.get("phase", "kie"),
        )
        result = {"success": success, "message": message}
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "pulse_sync":
        success, message = register_artifact(
            state_path=paths.vessel_state,
            artifact_path=arguments["path"],
            artifact_type=arguments["artifact_type"],
            milestone_id=arguments.get("milestone_id"),
        )
        result = {"success": success, "message": message}
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "pulse_export":
        result = export_pulse(
            state_path=paths.vessel_state,
            staging_path=paths.staging_dir,
            history_path=paths.history_dir,
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "pulse_status":
        status = manager.get_pulse_status()
        return [TextContent(type="text", text=json.dumps(status, indent=2))]

    elif name == "pulse_checkpoint":
        state = manager.load_state()
        if not state.active_pulse:
            return [TextContent(type="text", text=json.dumps({"error": "No active pulse"}))]

        if arguments.get("token_burden"):
            state.active_pulse.token_burden = arguments["token_burden"]
            manager.save_state(state)

        assessment = {
            "pulse_id": state.active_pulse.pulse_id,
            "artifact_count": len(state.active_pulse.artifacts),
            "token_burden": state.active_pulse.token_burden,
            "recommendation": "export" if state.active_pulse.token_burden == "high" else "continue",
        }
        return [TextContent(type="text", text=json.dumps(assessment, indent=2))]

    raise ValueError(f"Unknown tool: {name}")


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
