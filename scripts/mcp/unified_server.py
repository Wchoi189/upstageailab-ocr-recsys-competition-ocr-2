#!/usr/bin/env python3
"""
Unified Project MCP Server

A single server that combines all project MCP functionality:
- Project Compass resources and tools
- AgentQMS artifact workflows and standards
- Experiment Manager lifecycle tools
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent
from mcp.server.lowlevel.helper_types import ReadResourceContents

# Auto-discover project root
def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in current.parents:
        if (parent / "project_compass").exists():
            return parent
    return Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2")

PROJECT_ROOT = find_project_root()
COMPASS_DIR = PROJECT_ROOT / "project_compass"
AGENTQMS_DIR = PROJECT_ROOT / "AgentQMS"
EXPERIMENT_MANAGER_DIR = PROJECT_ROOT / "experiment_manager"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Add AgentQMS to path for dynamic templates
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = Server("unified_project")

# --- Resources Definitions ---

RESOURCES_CONFIG = [
    # Project Compass
    {"uri": "compass://compass.json", "name": "Project Compass State", "path": COMPASS_DIR / "compass.json", "mimeType": "application/json"},
    {"uri": "compass://session_handover.md", "name": "Session Handover", "path": COMPASS_DIR / "session_handover.md", "mimeType": "text/markdown"},
    {"uri": "compass://current_session.yml", "name": "Active Session Context", "path": COMPASS_DIR / "active_context" / "current_session.yml", "mimeType": "application/x-yaml"},
    {"uri": "compass://uv_lock_state.yml", "name": "Environment Lock State", "path": COMPASS_DIR / "environments" / "uv_lock_state.yml", "mimeType": "application/x-yaml"},
    {"uri": "compass://agents.yaml", "name": "Agent Configuration", "path": COMPASS_DIR / "AGENTS.yaml", "mimeType": "application/x-yaml"},
    # AgentQMS
    {"uri": "agentqms://standards/index", "name": "Standards Index", "path": AGENTQMS_DIR / "standards" / "INDEX.yaml", "mimeType": "application/x-yaml"},
    {"uri": "agentqms://standards/artifact_types", "name": "Artifact Types", "path": AGENTQMS_DIR / "standards" / "tier1-sst" / "artifact-types.yaml", "mimeType": "application/x-yaml"},
    {"uri": "agentqms://standards/workflows", "name": "Workflow Requirements", "path": AGENTQMS_DIR / "standards" / "tier1-sst" / "workflow-requirements.yaml", "mimeType": "application/x-yaml"},
    {"uri": "agentqms://templates/list", "name": "Template Catalog", "path": None, "mimeType": "application/json"},
    {"uri": "agentqms://config/settings", "name": "QMS Settings", "path": AGENTQMS_DIR / "config" / "settings.yaml", "mimeType": "application/x-yaml"},
    # Experiments
    {"uri": "experiments://agent_interface", "name": "Experiment Commands", "path": EXPERIMENT_MANAGER_DIR / "agent_interface.yaml", "mimeType": "application/x-yaml"},
    {"uri": "experiments://active_list", "name": "Active Experiments", "path": None, "mimeType": "application/json"},
    {"uri": "experiments://schemas/manifest", "name": "Manifest Schema", "path": EXPERIMENT_MANAGER_DIR / ".schemas" / "manifest.schema.json", "mimeType": "application/json"},
]

@app.list_resources()
async def list_resources() -> list[Resource]:
    return [Resource(uri=res["uri"], name=res["name"], mimeType=res["mimeType"]) for res in RESOURCES_CONFIG]

@app.read_resource()
async def read_resource(uri: str) -> list[ReadResourceContents]:
    uri = str(uri).strip()
    resource = next((r for r in RESOURCES_CONFIG if r["uri"] == uri), None)
    if not resource: raise ValueError(f"Unknown resource URI: {uri}")

    if uri == "agentqms://templates/list":
        try:
            from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow
            workflow = ArtifactWorkflow(quiet=True)
            return [ReadResourceContents(content=json.dumps({"templates": workflow.get_available_templates()}, indent=2), mime_type="application/json")]
        except: return [ReadResourceContents(content=json.dumps({"error": "Failed to load templates"}), mime_type="application/json")]

    if uri == "experiments://active_list":
        try:
            exps = []
            if EXPERIMENTS_DIR.exists():
                for d in EXPERIMENTS_DIR.iterdir():
                    if d.is_dir() and (d / "manifest.json").exists():
                        m = json.loads((d / "manifest.json").read_text())
                        exps.append({"id": d.name, "name": m.get("name", d.name), "status": m.get("status")})
            return [ReadResourceContents(content=json.dumps({"experiments": exps}, indent=2), mime_type="application/json")]
        except: return [ReadResourceContents(content=json.dumps({"error": "Failed to load experiments"}), mime_type="application/json")]

    path: Path = resource["path"]
    if not path or not path.exists(): raise FileNotFoundError(f"Resource file not found: {path}")
    return [ReadResourceContents(content=path.read_text(encoding="utf-8"), mime_type=resource["mimeType"])]

# --- Tools Definitions ---

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="manage_session", description="Manage project sessions", inputSchema={"type": "object", "properties": {"action": {"type": "string", "enum": ["export", "import", "list", "new"]}, "session_name": {"type": "string"}, "note": {"type": "string"}}, "required": ["action"]}),
        Tool(name="env_check", description="Validate project environment", inputSchema={"type": "object", "properties": {}}),
        Tool(name="create_artifact", description="Create standard artifact", inputSchema={"type": "object", "properties": {"artifact_type": {"type": "string"}, "name": {"type": "string"}, "title": {"type": "string"}, "description": {"type": "string"}}, "required": ["artifact_type", "name", "title"]}),
        Tool(name="validate_artifact", description="Validate artifact(s)", inputSchema={"type": "object", "properties": {"file_path": {"type": "string"}, "validate_all": {"type": "boolean"}}}),
        Tool(name="init_experiment", description="Initialize experiment", inputSchema={"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}}, "required": ["name"]}),
        Tool(name="log_insight", description="Log experiment insight", inputSchema={"type": "object", "properties": {"insight": {"type": "string"}, "type": {"type": "string", "enum": ["insight", "decision", "failure"]}}, "required": ["insight"]}),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        # Generic runner for simple script calls
        def run_py(args):
            res = subprocess.run(["uv", "run", "python3"] + args, capture_output=True, text=True, cwd=PROJECT_ROOT)
            return [TextContent(type="text", text=res.stdout + res.stderr)]

        if name == "manage_session":
            action = arguments["action"]
            cmd = [str(COMPASS_DIR / "scripts/session_manager.py"), action]
            if action == "import" and arguments.get("session_name"): cmd.append(arguments["session_name"])
            if action == "export" and arguments.get("note"): cmd.extend(["--note", arguments["note"]])
            return run_py(cmd)

        if name == "env_check": return run_py([str(COMPASS_DIR / "env_check.py")])

        if name == "create_artifact":
            cmd = [str(AGENTQMS_DIR / "bin/create-artifact.py"), "--type", arguments["artifact_type"], "--name", arguments["name"], "--title", arguments["title"]]
            if arguments.get("description"): cmd.extend(["--description", arguments["description"]])
            return run_py(cmd)

        if name == "validate_artifact":
            cmd = [str(AGENTQMS_DIR / "bin/validate-artifact.py")]
            if arguments.get("validate_all"): cmd.append("--all")
            else: cmd.append(arguments.get("file_path", ""))
            return run_py(cmd)

        if name == "init_experiment":
            cmd = ["-m", "etk.factory", "init", "--name", arguments["name"]]
            if arguments.get("description"): cmd.extend(["-d", arguments["description"]])
            return run_py(cmd)

        if name == "log_insight":
            return run_py(["-m", "etk.factory", "log", "--msg", arguments["insight"], "--type", arguments.get("type", "insight")])

        raise ValueError(f"Unknown tool: {name}")
    except Exception as e: return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
