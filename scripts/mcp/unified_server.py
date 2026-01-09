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
import os
import yaml
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent
from mcp.server.lowlevel.helper_types import ReadResourceContents


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

# --- Load Tool Groups Configuration ---

def load_tool_groups_config() -> dict[str, Any]:
    """Load tool groups configuration from YAML file."""
    config_path = Path(__file__).parent / "mcp_tools_config.yaml"
    if not config_path.exists():
        # Default: enable all groups
        return {
            "enabled_groups": ["compass", "agentqms", "etk", "adt_core", "adt_phase1", "adt_phase3"],
            "tool_groups": {}
        }

    with open(config_path) as f:
        return yaml.safe_load(f)


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

RESOURCES_CONFIG = [
    # Project Compass
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
    # AgentQMS
    {
        "uri": "agentqms://standards/index",
        "name": "Standards Index",
        "path": AGENTQMS_DIR / "standards" / "INDEX.yaml",
        "mimeType": "application/x-yaml",
    },
    {
        "uri": "agentqms://standards/artifact_types",
        "name": "Artifact Types",
        "path": AGENTQMS_DIR / "standards" / "tier1-sst" / "artifact-types.yaml",
        "mimeType": "application/x-yaml",
    },
    {
        "uri": "agentqms://plugins/artifact_types",
        "name": "Plugin Artifact Types",
        "description": "Discoverable artifact types with complete metadata",
        "mimeType": "application/json",
        "path": None,
    },
    {
        "uri": "agentqms://standards/workflows",
        "name": "Workflow Requirements",
        "path": AGENTQMS_DIR / "standards" / "tier1-sst" / "workflow-requirements.yaml",
        "mimeType": "application/x-yaml",
    },
    {"uri": "agentqms://templates/list", "name": "Template Catalog", "path": None, "mimeType": "application/json"},
    {
        "uri": "agentqms://config/settings",
        "name": "QMS Settings",
        "path": AGENTQMS_DIR / "config" / "settings.yaml",
        "mimeType": "application/x-yaml",
    },
    # Experiments
    {
        "uri": "experiments://agent_interface",
        "name": "Experiment Commands",
        "path": EXPERIMENT_MANAGER_DIR / "agent_interface.yaml",
        "mimeType": "application/x-yaml",
    },
    {"uri": "experiments://active_list", "name": "Active Experiments", "path": None, "mimeType": "application/json"},
    {
        "uri": "experiments://schemas/manifest",
        "name": "Manifest Schema",
        "path": EXPERIMENT_MANAGER_DIR / ".schemas" / "manifest.schema.json",
        "mimeType": "application/json",
    },
]


@app.list_resources()
async def list_resources() -> list[Resource]:
    return [Resource(uri=res["uri"], name=res["name"], description=res.get("description", ""), mimeType=res["mimeType"]) for res in RESOURCES_CONFIG]


@app.read_resource()
async def read_resource(uri: str) -> list[ReadResourceContents]:
    uri = str(uri).strip()
    resource = next((r for r in RESOURCES_CONFIG if r["uri"] == uri), None)
    if not resource:
        raise ValueError(f"Unknown resource URI: {uri}")

    if uri == "agentqms://templates/list":
        try:
            from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow

            workflow = ArtifactWorkflow(quiet=True)
            return [
                ReadResourceContents(
                    content=json.dumps({"templates": workflow.get_available_templates()}, indent=2), mime_type="application/json"
                )
            ]
        except:
            return [ReadResourceContents(content=json.dumps({"error": "Failed to load templates"}), mime_type="application/json")]

    if uri == "agentqms://plugins/artifact_types":
        try:
            from AgentQMS.mcp_server import _get_plugin_artifact_types

            content = await _get_plugin_artifact_types()
            return [ReadResourceContents(content=content, mime_type="application/json")]
        except Exception as e:
            return [
                ReadResourceContents(
                    content=json.dumps({"error": f"Failed to load plugin artifact types: {e}"}), mime_type="application/json"
                )
            ]

    if uri == "experiments://active_list":
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

    path: Path = resource["path"]
    if not path or not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")
    return [ReadResourceContents(content=path.read_text(encoding="utf-8"), mime_type=resource["mimeType"])]


# --- Tools Definitions ---


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all enabled tools based on tool groups configuration."""
    all_tools = [
        Tool(
            name="get_server_info",
            description="Get information about the Unified Project MCP server",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="project_compass",
            description="Project Compass guide and entrypoint information",
            inputSchema={"type": "object", "properties": {}},
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
                        "description": "Action to perform on the session.",
                    },
                    "session_name": {"type": "string", "description": "Name of the session to import (required for import action)."},
                    "note": {"type": "string", "description": "Note to attach to the session (optional for export)."},
                },
                "required": ["action"],
            },
        ),
        Tool(name="env_check", description="Validate project environment", inputSchema={"type": "object", "properties": {}}),
        Tool(
            name="create_artifact",
            description="Create standard artifact following project standards",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_type": {
                        "type": "string",
                        "enum": [
                            "assessment",
                            "audit",
                            "bug_report",
                            "design_document",
                            "implementation_plan",
                            "walkthrough",
                            "completed_plan",
                            "vlm_report",
                        ],
                    },
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "tags": {"type": "string"},
                    "content": {"type": "string", "description": "Markdown content to write to the file immediately"},
                },
                "required": ["artifact_type", "name", "title"],
            },
        ),
        Tool(
            name="validate_artifact",
            description="Validate artifact(s) against naming and structure standards",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "validate_all": {"type": "boolean"},
                },
            },
        ),
        Tool(name="list_artifact_templates", description="List all available artifact templates", inputSchema={"type": "object", "properties": {}}),
        Tool(name="check_compliance", description="Check overall artifact compliance status", inputSchema={"type": "object", "properties": {}}),
        Tool(
            name="get_standard",
            description="Retrieve project standard or rule content by name",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        # Experiment Manager (ETK)
        Tool(
            name="init_experiment",
            description="Initialize experiment",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                "required": ["name"],
            },
        ),
        Tool(
            name="log_insight",
            description="Log experiment insight",
            inputSchema={
                "type": "object",
                "properties": {"insight": {"type": "string"}, "type": {"type": "string", "enum": ["insight", "decision", "failure"]}},
                "required": ["insight"],
            },
        ),
        # Agent Debug Toolkit
        Tool(
            name="analyze_config_access",
            description="Analyze Python code for configuration access patterns (cfg.X, self.cfg.X, config['key']).",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "component": {"type": "string"},
                    "output": {"type": "string", "enum": ["json", "markdown"]},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="trace_merge_order",
            description="Trace OmegaConf.merge() operations and their precedence order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "explain": {"type": "boolean"},
                    "output": {"type": "string", "enum": ["json", "markdown"]},
                },
                "required": ["file"],
            },
        ),
        Tool(
            name="find_hydra_usage",
            description="Find Hydra framework usage patterns including @hydra.main decorators.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "output": {"type": "string", "enum": ["json", "markdown"]},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="find_component_instantiations",
            description="Track component instantiation patterns: get_*_by_cfg() factory calls.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "component": {"type": "string"},
                    "output": {"type": "string", "enum": ["json", "markdown"]},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="explain_config_flow",
            description="Generate a high-level summary of configuration flow through a file.",
            inputSchema={
                "type": "object",
                "properties": {"file": {"type": "string"}},
                "required": ["file"],
            },
        ),
        Tool(
            name="context_tree",
            description="Generate annotated directory tree with semantic context. Extracts docstrings, exports, and key definitions for AI navigation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to analyze"},
                    "depth": {"type": "integer", "description": "Maximum directory depth (default: 3)"},
                    "output": {"type": "string", "enum": ["json", "markdown"], "description": "Output format"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="intelligent_search",
            description="Search for symbols by name or qualified path with fuzzy matching. Resolves Hydra _target_ paths and finds class definitions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Symbol name or qualified path"},
                    "root": {"type": "string", "description": "Root directory to search (default: project root)"},
                    "fuzzy": {"type": "boolean", "description": "Enable fuzzy matching (default: true)"},
                    "threshold": {"type": "number", "description": "Min similarity 0.0-1.0 (default: 0.6)"},
                },
                "required": ["query"],
            },
        ),
        # --- Meta-Tools (Router Pattern) ---
        Tool(
            name="adt_meta_query",
            description="Unified analysis tool. Routes based on 'kind': config_access, merge_order, hydra_usage, component_instantiations, config_flow, dependency_graph, imports, complexity, context_tree, symbol_search",
            inputSchema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["config_access", "merge_order", "hydra_usage", "component_instantiations", "config_flow", "dependency_graph", "imports", "complexity", "context_tree", "symbol_search"]},
                    "target": {"type": "string", "description": "Target path or query"},
                    "options": {"type": "object", "additionalProperties": True},
                },
                "required": ["kind", "target"],
            },
        ),
        Tool(
            name="adt_meta_edit",
            description="Unified edit tool. Routes based on 'kind': apply_diff, smart_edit, read_slice, format",
            inputSchema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["apply_diff", "smart_edit", "read_slice", "format"]},
                    "target": {"type": "string", "description": "Target file or diff content"},
                    "options": {"type": "object", "additionalProperties": True},
                },
                "required": ["kind", "target"],
            },
        ),
        # --- Edit Tools ---
        Tool(
            name="apply_unified_diff",
            description="Apply a unified diff with fuzzy matching. Handles whitespace drift. Returns detailed hunk report.",
            inputSchema={
                "type": "object",
                "properties": {
                    "diff": {"type": "string", "description": "Unified diff text (git diff format)"},
                    "strategy": {"type": "string", "enum": ["exact", "whitespace_insensitive", "fuzzy"], "default": "fuzzy"},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["diff"],
            },
        ),
        Tool(
            name="smart_edit",
            description="Intelligent search/replace with exact, regex, or fuzzy matching.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Path to file"},
                    "search": {"type": "string", "description": "Text or pattern to find"},
                    "replace": {"type": "string", "description": "Replacement text"},
                    "mode": {"type": "string", "enum": ["exact", "regex", "fuzzy"], "default": "exact"},
                    "all_occurrences": {"type": "boolean", "default": False},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["file", "search", "replace"],
            },
        ),
        Tool(
            name="read_file_slice",
            description="Read specific line range from a file for targeted editing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Path to file"},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed)"},
                    "end_line": {"type": "integer", "description": "End line (inclusive)"},
                    "context_lines": {"type": "integer", "default": 0},
                },
                "required": ["file", "start_line", "end_line"],
            },
        ),
        Tool(
            name="format_code",
            description="Format code using black, ruff, or isort.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file or directory"},
                    "style": {"type": "string", "enum": ["black", "ruff", "isort"], "default": "black"},
                    "check_only": {"type": "boolean", "default": False},
                },
                "required": ["path"],
            },
        ),
    ]

    # Filter tools based on enabled groups
    enabled_tools = [tool for tool in all_tools if is_tool_enabled(tool.name)]

    return enabled_tools


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        # Generic runner for simple script calls
        def run_py(args):
            res = subprocess.run(["uv", "run", "python3"] + args, capture_output=True, text=True, cwd=PROJECT_ROOT)
            return [TextContent(type="text", text=res.stdout + res.stderr)]

        if name == "get_server_info":
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

        if name == "project_compass":
            entrypoint_path = COMPASS_DIR / "AI_ENTRYPOINT.md"
            if entrypoint_path.exists():
                return [TextContent(type="text", text=entrypoint_path.read_text())]
            return [TextContent(type="text", text="Project Compass Entrypoint not found.")]

        if name == "manage_session":
            action = arguments["action"]
            cmd = [str(COMPASS_DIR / "scripts/session_manager.py"), action]
            if action == "import" and arguments.get("session_name"):
                cmd.append(arguments["session_name"])
            if action == "export" and arguments.get("note"):
                cmd.extend(["--note", arguments["note"]])
            return run_py(cmd)

        if name == "env_check":
            try:
                # Import the EnvironmentChecker from ETK
                from etk.compass import EnvironmentChecker

                checker = EnvironmentChecker()
                passed, errors, warnings = checker.check_all()

                result_lines = []
                if warnings:
                    for warning in warnings:
                        result_lines.append(f"⚠️  {warning}")

                if errors:
                    result_lines.append("❌ ENVIRONMENT BREACH DETECTED")
                    for error in errors:
                        result_lines.append(f"  ✗ {error}")
                    result_lines.append("\n🔧 Path Restoration Instructions:")
                    result_lines.append("   1. Ensure you are using the correct UV binary")
                    result_lines.append("   2. Run: uv sync")
                    result_lines.append('   3. Verify with: uv run python -c "import torch; print(torch.__version__)"')
                else:
                    result_lines.append("✅ Environment validated against Compass lock state")

                return [TextContent(type="text", text="\n".join(result_lines))]
            except Exception as e:
                return [TextContent(type="text", text=f"Error checking environment: {e}")]

        if name == "create_artifact":
            try:
                from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow
                workflow = ArtifactWorkflow(quiet=True)
                result = workflow.create_artifact(
                    artifact_type=arguments["artifact_type"],
                    name=arguments["name"],
                    title=arguments["title"],
                    description=arguments.get("description"),
                    tags=arguments.get("tags"),
                    content=arguments.get("content")
                )
                return [TextContent(type="text", text=result)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error creating artifact: {e}")]

        if name == "validate_artifact":
            try:
                from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow
                workflow = ArtifactWorkflow(quiet=True)

                if arguments.get("validate_all"):
                    success = workflow.validate_all()
                    return [TextContent(type="text", text=f"Validation {'passed' if success else 'failed'}")]
                elif arguments.get("file_path"):
                    success = workflow.validate_artifact(arguments["file_path"])
                    return [TextContent(type="text", text=f"Validation {'passed' if success else 'failed'}")]
                else:
                    return [TextContent(type="text", text="Error: Must specify either validate_all or file_path")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error validating artifact: {e}")]

        if name == "list_artifact_templates":
            try:
                from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow
                workflow = ArtifactWorkflow(quiet=True)
                return [TextContent(type="text", text=json.dumps({"templates": workflow.get_available_templates()}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {e}")]

        if name == "check_compliance":
            try:
                from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow
                workflow = ArtifactWorkflow(quiet=True)
                return [TextContent(type="text", text=json.dumps(workflow.check_compliance(), indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {e}")]

        if name == "get_standard":
            # Direct python implementation for get_standard as it's simple file searching
            query = arguments["name"].lower()
            standards_dir = AGENTQMS_DIR / "standards"
            matches = []
            if standards_dir.exists():
                for path in standards_dir.rglob("*"):
                    if path.is_file() and path.suffix in [".yaml", ".md", ".json"]:
                        if query in path.stem.lower():
                            matches.append(path)
            if not matches:
                return [TextContent(type="text", text=f"No standards found matching '{query}'")]
            if len(matches) == 1:
                return [TextContent(type="text", text=f"Standard: {matches[0].name}\n\n{matches[0].read_text(encoding='utf-8')}")]
            return [TextContent(type="text", text=f"Multiple matches: {[str(p.relative_to(standards_dir)) for p in matches]}")]

        if name == "init_experiment":
            cmd = ["-m", "etk.factory", "init", "--name", arguments["name"]]
            if arguments.get("description"):
                cmd.extend(["-d", arguments["description"]])
            return run_py(cmd)

        if name == "log_insight":
            return run_py(["-m", "etk.factory", "log", "--msg", arguments["insight"], "--type", arguments.get("type", "insight")])

        # Agent Debug Toolkit Tools
        def resolve_adt_path(p: str) -> Path:
            path = Path(p)
            return path if path.is_absolute() else PROJECT_ROOT / path

        if name == "analyze_config_access":
            from agent_debug_toolkit.analyzers.config_access import ConfigAccessAnalyzer
            path = resolve_adt_path(arguments["path"])
            report = ConfigAccessAnalyzer().analyze_file(path) if path.is_file() else ConfigAccessAnalyzer().analyze_directory(path)
            if arguments.get("component"):
                report.results = report.filter_by_component(arguments["component"])
            return [TextContent(type="text", text=report.to_json() if arguments.get("output") == "json" else report.to_markdown())]

        if name == "trace_merge_order":
            from agent_debug_toolkit.analyzers.merge_order import MergeOrderTracker
            path = resolve_adt_path(arguments["file"])
            analyzer = MergeOrderTracker()
            report = analyzer.analyze_file(path)
            content = report.to_json() if arguments.get("output") == "json" else f"{analyzer.explain_precedence()}\n\n{report.to_markdown()}"
            return [TextContent(type="text", text=content)]

        if name == "find_hydra_usage":
            from agent_debug_toolkit.analyzers.hydra_usage import HydraUsageAnalyzer
            path = resolve_adt_path(arguments["path"])
            report = HydraUsageAnalyzer().analyze_file(path) if path.is_file() else HydraUsageAnalyzer().analyze_directory(path)
            return [TextContent(type="text", text=report.to_json() if arguments.get("output") == "json" else report.to_markdown())]

        if name == "find_component_instantiations":
            from agent_debug_toolkit.analyzers.instantiation import ComponentInstantiationTracker
            path = resolve_adt_path(arguments["path"])
            report = ComponentInstantiationTracker().analyze_file(path) if path.is_file() else ComponentInstantiationTracker().analyze_directory(path)
            content = report.to_json() if arguments.get("output") == "json" else report.to_markdown()
            return [TextContent(type="text", text=content)]

        if name == "explain_config_flow":
            # Nested imports to avoid circular/missing dependencies if not used
            from agent_debug_toolkit.analyzers.config_access import ConfigAccessAnalyzer
            from agent_debug_toolkit.analyzers.merge_order import MergeOrderTracker
            from agent_debug_toolkit.analyzers.hydra_usage import HydraUsageAnalyzer
            from agent_debug_toolkit.analyzers.instantiation import ComponentInstantiationTracker

            path = resolve_adt_path(arguments["file"])
            if not path.is_file(): return [TextContent(type="text", text="Error: File not found")]

            creport = ConfigAccessAnalyzer().analyze_file(path)
            mtracker = MergeOrderTracker()
            mreport = mtracker.analyze_file(path)
            hreport = HydraUsageAnalyzer().analyze_file(path)
            ireport = ComponentInstantiationTracker().analyze_file(path)

            summary = [f"# Config Flow: {path.name}", "", f"Accesses: {len(creport.results)}", f"Merges: {len(mreport.results)}", f"Hydra: {len(hreport.results)}", f"Components: {len(ireport.results)}"]
            if mreport.results: summary.extend(["", "## Precedence", mtracker.explain_precedence()])

            return [TextContent(type="text", text="\n".join(summary))]

        if name == "context_tree":
            from agent_debug_toolkit.analyzers.context_tree import ContextTreeAnalyzer, format_tree_markdown
            path = resolve_adt_path(arguments["path"])

            if not path.exists():
                return [TextContent(type="text", text=f"Error: Path not found: {path}")]
            if not path.is_dir():
                return [TextContent(type="text", text=f"Error: Not a directory: {path}")]

            depth = arguments.get("depth", 3)
            analyzer = ContextTreeAnalyzer(max_depth=depth)
            report = analyzer.analyze_directory(str(path))

            if "error" in report.summary:
                return [TextContent(type="text", text=f"Error: {report.summary['error']}")]

            if arguments.get("output") == "json":
                return [TextContent(type="text", text=report.to_json())]
            else:
                return [TextContent(type="text", text=format_tree_markdown(report))]

        if name == "intelligent_search":
            from agent_debug_toolkit.analyzers.intelligent_search import IntelligentSearcher, format_search_results_markdown

            query = arguments["query"]
            root = resolve_adt_path(arguments.get("root", PROJECT_ROOT))
            fuzzy = arguments.get("fuzzy", True)
            threshold = arguments.get("threshold", 0.6)

            searcher = IntelligentSearcher(str(root), str(PROJECT_ROOT))
            results = searcher.search(query, fuzzy=fuzzy, threshold=threshold)

            if arguments.get("output") == "json":
                return [TextContent(type="text", text=json.dumps([r.to_dict() for r in results], indent=2))]
            else:
                return [TextContent(type="text", text=format_search_results_markdown(results, query))]

        # --- Edit Tools ---
        if name == "apply_unified_diff":
            from agent_debug_toolkit.edits import apply_unified_diff
            diff = arguments.get("diff", "")
            strategy = arguments.get("strategy", "fuzzy")
            dry_run = arguments.get("dry_run", False)
            report = apply_unified_diff(diff=diff, strategy=strategy, project_root=PROJECT_ROOT, dry_run=dry_run)
            return [TextContent(type="text", text=report.to_json())]

        if name == "smart_edit":
            from agent_debug_toolkit.edits import smart_edit
            path = resolve_adt_path(arguments.get("file", ""))
            report = smart_edit(
                file=path,
                search=arguments.get("search", ""),
                replace=arguments.get("replace", ""),
                mode=arguments.get("mode", "exact"),
                all_occurrences=arguments.get("all_occurrences", False),
                dry_run=arguments.get("dry_run", False),
            )
            return [TextContent(type="text", text=report.to_json())]

        if name == "read_file_slice":
            from agent_debug_toolkit.edits import read_file_slice
            path = resolve_adt_path(arguments.get("file", ""))
            content = read_file_slice(
                path,
                int(arguments.get("start_line", 1)),
                int(arguments.get("end_line", 50)),
                int(arguments.get("context_lines", 0)),
            )
            return [TextContent(type="text", text=content)]

        if name == "format_code":
            from agent_debug_toolkit.edits import format_code
            path = resolve_adt_path(arguments.get("path", ""))
            report = format_code(path, style=arguments.get("style", "black"), check_only=arguments.get("check_only", False))
            return [TextContent(type="text", text=report.to_json())]

        raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
