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
import os
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
from AgentQMS.middleware.policies import RedundancyInterceptor, ComplianceInterceptor, FileOperationInterceptor

# Initialize Middleware
TELEMETRY_PIPELINE = TelemetryPipeline([
    RedundancyInterceptor(),
    ComplianceInterceptor(),
    FileOperationInterceptor()
])


# Auto-discover project root
def find_project_root() -> Path:
    # First, check environment variable
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root)

    # Second, traverse up from this file to find project_compass marker
    current = Path(__file__).resolve().parent
    for parent in current.parents:
        if (parent / "project_compass").exists():
            return parent

    # Fallback to current working directory (portable)
    return Path.cwd()


PROJECT_ROOT = find_project_root()
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

# --- Load Tool Groups Configuration ---

def load_tool_groups_config() -> dict[str, Any]:
    """Load tool groups configuration from YAML file."""
    config_path = Path(__file__).parent / "mcp_tools_config.yaml"
    defaults = {
        "enabled_groups": ["compass", "agentqms", "etk", "adt_core", "adt_phase1", "adt_phase3"],
        "tool_groups": {}
    }
    return config_loader.get_config(config_path, defaults=defaults)


TOOLS_CONFIG = load_tool_groups_config()
ENABLED_GROUPS = set(TOOLS_CONFIG.get("enabled_groups", []))


def is_tool_enabled(tool_name: str) -> bool:
    """Check if a tool is enabled based on group configuration."""
    tool_groups = TOOLS_CONFIG.get("tool_groups", {})

    for group_name, group_config in tool_groups.items():
        if group_name in ENABLED_GROUPS:
            if tool_name in group_config.get("tools", []):
                return True

    # If no groups defined or tool not in any group, enable by default
    if not tool_groups:
        return True

    return False


# --- Resources Definitions ---

def load_resources_config() -> list[dict]:
    """Load resources configuration from YAML file."""
    config_path = Path(__file__).parent / "config/resources.yaml"
    raw_resources = config_loader.get_config(config_path, defaults=[])

    if not isinstance(raw_resources, list):
        return []

    # Process paths: resolve relative paths against project root
    for res in raw_resources:
        if res.get("path"):
            res["path"] = PROJECT_ROOT / res["path"]
        else:
            res["path"] = None
    return raw_resources

RESOURCES_CONFIG = load_resources_config()

# --- Optimized Resource Lookups ---
# Pre-computing these maps turns O(N) searches into O(1) lookups
URI_MAP = {r["uri"]: r for r in RESOURCES_CONFIG}
PATH_MAP = {}

for r in RESOURCES_CONFIG:
    if r.get("path"):
        try:
            resolved = r["path"].resolve()
            PATH_MAP[resolved] = r
        except Exception:
            continue


@app.list_resources()
async def list_resources() -> list[Resource]:
    return [Resource(uri=res["uri"], name=res["name"], description=res.get("description", ""), mimeType=res["mimeType"]) for res in RESOURCES_CONFIG]


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

    # 1. Fast Path: Exact URI Match O(1)
    resource = URI_MAP.get(uri)

    # 2. Alias Handling: Path Match O(1)
    if not resource:
        input_path = None
        if uri.startswith("file://"):
            from urllib.parse import urlparse, unquote
            input_path = Path(unquote(urlparse(uri).path)).resolve()
        elif uri.startswith("/") or (len(uri) > 1 and uri[1] == ":"):
            input_path = Path(uri).resolve()

        if input_path:
            resource = PATH_MAP.get(input_path)

    # 3. Error Handling & Suggestions
    if not resource:
        msg = f"Unknown resource URI: {uri}"
        if any(keyword in uri for keyword in ["compass", "agentqms"]):
            # Filtered list comprehension for suggestions
            matches = [u for u in URI_MAP.keys() if uri.split("://")[-1] in u]
            if matches:
                msg += f". Did you mean: {', '.join(matches[:3])}?"
        raise ValueError(msg)

    # 4. Dynamic Resource Handlers
    if uri == "agentqms://templates/list":
        return _handle_templates_list()

    if uri == "agentqms://plugins/artifact_types":
        return await _handle_plugin_artifacts()

    if uri == "experiments://active_list":
        return await _handle_experiments_list()

    # 5. Static File Reading
    file_path: Path = resource["path"]
    if not file_path or not file_path.exists():
        raise FileNotFoundError(f"Resource file not found: {file_path}")

    return [ReadResourceContents(content=file_path.read_text(encoding="utf-8"), mime_type=resource["mimeType"])]


# --- Tools Definitions ---


def load_tools_definitions() -> list[dict]:
    """Load tools configuration from YAML file."""
    config_path = Path(__file__).parent / "config/tools.yaml"
    tools = config_loader.get_config(config_path, defaults=[])
    return tools if isinstance(tools, list) else []

TOOLS_DEFINITIONS = load_tools_definitions()

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all enabled tools based on tool groups configuration and external definitions."""
    enabled_tools = []

    # 1. Filter enabled tools
    enabled_names = set()
    # If using groups:
    # Iterate all definitions, check if enabled

    for tool_def in TOOLS_DEFINITIONS:
        name = tool_def["name"]
        if is_tool_enabled(name):
            enabled_tools.append(Tool(
                name=name,
                description=tool_def["description"],
                inputSchema=tool_def["inputSchema"]
            ))

    return enabled_tools


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        # --- Middleware Validation ---
        TELEMETRY_PIPELINE.validate(name, arguments)

        if name == "get_server_info":
             # Keep local implementation for server info
            info = {
                "name": "unified_project",
                "version": "1.0.0",
                "status": "running",
                "components": ["Compass", "AgentQMS", "ETK", "ADT"],
                "tool_groups": {
                    "enabled": list(ENABLED_GROUPS),
                    "available": list(TOOLS_CONFIG.get("tool_groups", {}).keys()),
                },
            }
            return [TextContent(type="text", text=json.dumps(info, indent=2))]

        # --- Dynamic Dispatch ---
        # Find tool definition
        tool_def = next((t for t in TOOLS_DEFINITIONS if t["name"] == name), None)

        if tool_def and "implementation" in tool_def:
            impl = tool_def["implementation"]
            module_name = impl.get("module")
            # function_name = impl.get("function", "call_tool") # Default to call_tool

            if module_name:
                import importlib
                module = importlib.import_module(module_name)
                # We assume the module exposes the same call_tool interface or we access the app?
                # Most standard MCP servers using @app.call_tool decorate a function wrapped by FastMCP or Server
                # BUT here we are importing the module directly.
                # If the module has a top-level 'call_tool' function (which we checked they do), use it.
                if hasattr(module, "call_tool"):
                    return await module.call_tool(name, arguments)

                # Fallback: if it's decorating 'app', we might need to find the handler.
                # But in our analyzed files, 'call_tool' is a dedicated async function.
                raise ValueError(f"Module {module_name} does not have a 'call_tool' function.")

        raise ValueError(f"Unknown tool or missing implementation: {name}")

    except PolicyViolation as e:
        # Return the feedback message to the agent instead of executing the tool
        return [TextContent(type="text", text=f"⚠️ FEEDBACK TRIGGERED: {e.feedback_to_ai}")]
    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}\n{traceback.format_exc()}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
