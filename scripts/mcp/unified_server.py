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
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent
from mcp.server.lowlevel.helper_types import ReadResourceContents

# Add AgentQMS to path before importing from it
from pathlib import Path as PathlibPath
_scripts_mcp_dir = PathlibPath(__file__).resolve().parent
_project_root_candidate = _scripts_mcp_dir.parent.parent
if str(_project_root_candidate) not in sys.path:
    sys.path.insert(0, str(_project_root_candidate))

from AgentQMS.tools.utils.config_loader import ConfigLoader

# --- Middleware Imports ---
from AgentQMS.middleware.telemetry import TelemetryPipeline, PolicyViolation
from AgentQMS.middleware.policies import RedundancyInterceptor, ComplianceInterceptor, FileOperationInterceptor, StandardsInterceptor

# Initialize Middleware
TELEMETRY_PIPELINE = TelemetryPipeline([
    RedundancyInterceptor(),
    ComplianceInterceptor(),
    FileOperationInterceptor(),
    StandardsInterceptor()
])


# Import path utility
from AgentQMS.tools.utils.paths import get_project_root

PROJECT_ROOT = get_project_root()
COMPASS_DIR = PROJECT_ROOT / "project_compass"
AGENTQMS_DIR = PROJECT_ROOT / "AgentQMS"
EXPERIMENT_MANAGER_DIR = PROJECT_ROOT / "experiment_manager"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Add AgentQMS and Agent Debug Toolkit to path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEBUG_TOOLKIT_SRC = PROJECT_ROOT / "agent-debug-toolkit/src"
if DEBUG_TOOLKIT_SRC.exists() and str(DEBUG_TOOLKIT_SRC) not in sys.path:
    sys.path.insert(0, str(DEBUG_TOOLKIT_SRC))

app = Server("unified_project")

# --- Configuration Loader Setup ---
config_loader = ConfigLoader(cache_size=5)


# --- Resources Aggregation from Server Modules ---

async def load_resources_from_servers() -> list[dict]:
    """Aggregate resources from all MCP servers + unified config."""
    import importlib

    resources = []

    # 1. Load from AgentQMS server (agentqms:// URIs)
    mod = importlib.import_module("AgentQMS.mcp_server")
    agentqms_resources = await mod.list_resources()
    for res in agentqms_resources:
        resources.append({
            "uri": res.uri,
            "name": res.name,
            "description": res.description,
            "mimeType": res.mimeType,
            "path": None  # Handled by AgentQMS server's read_resource
        })

    # 2. Load from project_compass server (compass:// URIs)
    mod = importlib.import_module("project_compass.mcp_server")
    compass_resources = await mod.list_resources()
    for res in compass_resources:
        resources.append({
            "uri": res.uri,
            "name": res.name,
            "description": res.description,
            "mimeType": res.mimeType,
            "path": None  # Handled by compass server's read_resource
        })

    # 3. Load from experiment_manager server (experiments:// URIs)
    mod = importlib.import_module("experiment_manager.mcp_server")
    exp_resources = await mod.list_resources()
    for res in exp_resources:
        resources.append({
            "uri": res.uri,
            "name": res.name,
            "description": res.description,
            "mimeType": res.mimeType,
            "path": None  # Handled by experiments server's read_resource
        })

    # 4. Load remaining resources from unified config (utilities://, standards://)
    config_path = Path(__file__).parent / "config/resources.yaml"
    config_resources = config_loader.get_config(config_path, defaults=[])
    for res in config_resources:
        if res.get("path"):
            res["path"] = PROJECT_ROOT / res["path"]
        resources.append(res)

    # 5. Add bundle:// resources (context bundles)
    resources.append({
        "uri": "bundle://list",
        "name": "Context Bundles",
        "description": "List available context bundles for AI agents",
        "mimeType": "application/json",
        "path": None  # Dynamic handler
    })

    return resources

RESOURCES_CONFIG = []

@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(uri=res["uri"], name=res["name"], description=res.get("description", ""), mimeType=res["mimeType"])
        for res in RESOURCES_CONFIG
    ]


# --- Helper Functions for Dynamic Resources ---

async def _handle_experiments_list() -> list[ReadResourceContents]:
    try:
        exps = []
        if EXPERIMENTS_DIR.exists():
            for d in EXPERIMENTS_DIR.iterdir():
                if d.is_dir() and (d / "manifest.json").exists():
                    m = json.loads((d / "manifest.json").read_text())
                    exps.append({"id": d.name, "name": m.get("name", d.name), "status": m.get("status")})
        return [ReadResourceContents(content=json.dumps({"experiments": exps}, indent=2), mime_type="application/json")]
    except:
        return [ReadResourceContents(content=json.dumps({"error": "Failed to load experiments"}), mime_type="application/json")]

async def _handle_plugin_artifacts() -> list[ReadResourceContents]:
    try:
        from AgentQMS.mcp_server import _get_plugin_artifact_types
        content = await _get_plugin_artifact_types()
        return [ReadResourceContents(content=content, mime_type="application/json")]
    except Exception as e:
        return [ReadResourceContents(content=json.dumps({"error": f"Failed to load plugin artifact types: {e}"}), mime_type="application/json")]

def _handle_templates_list() -> list[ReadResourceContents]:
    try:
        from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow
        workflow = ArtifactWorkflow(quiet=True)
        return [ReadResourceContents(content=json.dumps({"templates": workflow.get_available_templates()}, indent=2), mime_type="application/json")]
    except:
        return [ReadResourceContents(content=json.dumps({"error": "Failed to load templates"}), mime_type="application/json")]

@app.read_resource()
async def read_resource(uri: str) -> list[ReadResourceContents]:
    uri = str(uri).strip()

    # Route to individual servers based on URI scheme
    import importlib

    if uri.startswith("agentqms://"):
        mod = importlib.import_module("AgentQMS.mcp_server")
        return await mod.read_resource(uri)

    elif uri.startswith("compass://"):
        mod = importlib.import_module("project_compass.mcp_server")
        return await mod.read_resource(uri)

    elif uri.startswith("experiments://"):
        mod = importlib.import_module("experiment_manager.mcp_server")
        return await mod.read_resource(uri)

    elif uri.startswith("bundle://"):
        try:
            from AgentQMS.tools.core.context_bundle import list_available_bundles, get_context_bundle

            if uri == "bundle://list":
                bundles = list_available_bundles()
                return [ReadResourceContents(content=json.dumps(bundles, indent=2), mime_type="application/json")]
            else:
                bundle_name = uri.replace("bundle://", "")
                # Get bundle files by passing bundle name as task description
                files = get_context_bundle(task_description=bundle_name, task_type=bundle_name)
                return [ReadResourceContents(content=json.dumps({"bundle": bundle_name, "files": files}, indent=2), mime_type="application/json")]
        except Exception as e:
            return [ReadResourceContents(content=json.dumps({"error": str(e)}), mime_type="application/json")]

    # Handle file-based resources (utilities://, standards://)
    resource = next((r for r in RESOURCES_CONFIG if r["uri"] == uri), None)

    if not resource:
        raise ValueError(f"Unknown resource URI: {uri}")

    file_path: Path = resource.get("path")
    if not file_path or not file_path.exists():
        raise FileNotFoundError(f"Resource file not found: {file_path}")

    return [ReadResourceContents(content=file_path.read_text(encoding="utf-8"), mime_type=resource["mimeType"])]





# --- Tools Aggregation from Server Modules ---

async def load_tools_from_servers() -> list[dict]:
    """Runtime aggregation: import list_tools() from each MCP server."""
    import importlib

    tools = []
    servers = [
        ("AgentQMS.mcp_server", "agentqms"),
        ("project_compass.mcp_server", "compass"),
        ("experiment_manager.mcp_server", "experiments"),
        ("agent_debug_toolkit.mcp_server", "adt"),
    ]

    for module_name, _prefix in servers:
        mod = importlib.import_module(module_name)
        server_tools = await mod.list_tools()

        for tool in server_tools:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
                "implementation": {
                    "module": module_name,
                    "function": "call_tool"
                }
            })

    return tools

TOOLS_DEFINITIONS = []

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools aggregated from individual MCP servers."""
    return [
        Tool(
            name=tool_def["name"],
            description=tool_def["description"],
            inputSchema=tool_def["inputSchema"]
        )
        for tool_def in TOOLS_DEFINITIONS
    ]



@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        # --- Middleware Validation ---
        TELEMETRY_PIPELINE.validate(name, arguments)

        # --- Find tool implementation ---
        tool_def = next((t for t in TOOLS_DEFINITIONS if t["name"] == name), None)

        if not tool_def or "implementation" not in tool_def:
            raise ValueError(f"Unknown tool: {name}")

        # --- Route to appropriate server module ---
        impl = tool_def["implementation"]
        module_name = impl.get("module")

        if not module_name:
            raise ValueError(f"No module specified for tool: {name}")

        import importlib
        module = importlib.import_module(module_name)

        if not hasattr(module, "call_tool"):
            raise ValueError(f"Module {module_name} does not have a 'call_tool' function")

        result_content = await module.call_tool(name, arguments)

        # --- Context Suggestion Injection ---
        try:
            # Suggest relevant context bundles for non-context tools
            if "context" not in name and "bundle" not in name:
                from AgentQMS.tools.core.context_bundle import auto_suggest_context
                task_desc = f"{name}: {str(arguments)}"
                suggestions = auto_suggest_context(task_desc)

                # Only suggest if we have a specific bundle
                if suggestions.get("bundle_files"):
                    bundle_name = suggestions.get("context_bundle")
                    if bundle_name and bundle_name not in ("general", "development", "documentation"):
                        suggestion_text = (
                            f"\n💡 **Context Suggestion**: The '{bundle_name}' bundle seems relevant.\n"
                            f"   Access it: read_resource('bundle://{bundle_name}')"
                        )
                        result_content.append(TextContent(type="text", text=suggestion_text))
        except Exception:
            # Suppress suggestion errors to not break main tool execution
            pass

        return result_content

    except PolicyViolation as e:
        return [TextContent(type="text", text=f"⚠️ FEEDBACK TRIGGERED: {e.feedback_to_ai}")]
    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}\n{traceback.format_exc()}")]




async def main():
    global TOOLS_DEFINITIONS, RESOURCES_CONFIG
    TOOLS_DEFINITIONS = await load_tools_from_servers()
    RESOURCES_CONFIG = await load_resources_from_servers()

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())