#!/usr/bin/env python3
"""
AgentQMS MCP Server

Exposes artifact workflow tools and specs as MCP resources and tools.

Resources:
- agentqms://specs/index - Specs registry index
- agentqms://specs/artifact_types - Artifact types and locations
- agentqms://specs/workflows - Workflow requirements
- agentqms://templates/list - Available templates
- agentqms://config/settings - QMS settings
- agentqms://context/bundles - List of available context bundles
- agentqms://context/bundle/{name} - Specific context bundle file list

Tools:
- create_artifact - Create new artifact following specs
- validate_artifact - Validate artifact against specs
- list_artifact_templates - List available templates
- check_compliance - Check overall compliance status
- get_context_bundle - Get file paths for a specific task or bundle name
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


# Bootstrap import path when running as a script from source checkout.
for parent in Path(__file__).resolve().parents:
    if (parent / "AgentQMS").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break


def find_project_root() -> Path:
    """Resolve project root via canonical path resolver."""
    from AgentQMS.tools.utils.paths import get_project_root

    return get_project_root().resolve()


def find_framework_root() -> Path:
    """Resolve framework root via canonical path resolver."""
    from AgentQMS.tools.utils.paths import get_framework_root

    return get_framework_root().resolve()


PROJECT_ROOT = find_project_root()
AGENTQMS_DIR = find_framework_root()


from AgentQMS.tools.utils.config import YamlCacheLoader
from AgentQMS.tools.core.context.context_bundle import get_context_bundle, list_available_bundles
from AgentQMS.tools.core.mcp.handlers import HandlerContext, TOOL_HANDLERS, text_payload

# Initialize YAML/cache loader for MCP schema reads.
CONFIG_LOADER = YamlCacheLoader()

def load_mcp_schema() -> dict[str, list[dict]]:
    """Load MCP resources from schema."""
    schema_path = AGENTQMS_DIR / "mcp_schema.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(f"MCP Schema not found at: {schema_path}")

    return CONFIG_LOADER.get_config(schema_path, defaults={"resources": []})


# Load Schema
MCP_SCHEMA = load_mcp_schema()
RESOURCES = MCP_SCHEMA.get("resources", [])


# Resolve Resource Paths
for res in RESOURCES:
    if res.get("path"):
        res["path"] = PROJECT_ROOT / res["path"]


# Create MCP server
app = Server("agentqms")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List all available AgentQMS resources."""
    resources = [
        Resource(
            uri=res["uri"],
            name=res["name"],
            description=res["description"],
            mimeType=res["mimeType"],
        )
        for res in RESOURCES
    ]

    # Add dynamic context bundle resources
    resources.append(
        Resource(
            uri="agentqms://context/bundles",
            name="Context Bundles List",
            description="List of all available context bundles",
            mimeType="application/json",
        )
    )

    # Add individual bundles
    available_bundles = list_available_bundles()
    for bundle in available_bundles:
        resources.append(
            Resource(
                uri=f"agentqms://context/bundle/{bundle}",
                name=f"Context Bundle: {bundle}",
                description=f"File list for {bundle} context bundle",
                mimeType="application/json",
            )
        )

    return resources


@app.read_resource()
async def read_resource(uri: str) -> list[ReadResourceContents]:
    """Read content of an AgentQMS resource."""
    # Find matching resource
    uri = str(uri).strip()
    resource = next((r for r in RESOURCES if r["uri"] == uri), None)

    # Handle dynamic template list
    if uri == "agentqms://templates/list":
        content = await _get_template_list()
        return [ReadResourceContents(content=content, mime_type="application/json")]

    # Handle dynamic plugin artifact types
    if uri == "agentqms://plugins/artifact_types":
        content = await _get_plugin_artifact_types()
        return [ReadResourceContents(content=content, mime_type="application/json")]

    # Handle Context Bundles List
    if uri == "agentqms://context/bundles":
        bundles = list_available_bundles()
        return [
            ReadResourceContents(
                content=json.dumps({"bundles": bundles}, indent=2),
                mime_type="application/json",
            )
        ]

    # Handle Individual Context Bundle
    if uri.startswith("agentqms://context/bundle/"):
        bundle_name = uri.split("/")[-1]
        try:
            # We use get_context_bundle with task_type=bundle_name to bypass auto-detection
            files = get_context_bundle(task_description=f"Load {bundle_name}", task_type=bundle_name)
            return [
                ReadResourceContents(
                    content=json.dumps({"bundle": bundle_name, "files": files}, indent=2),
                    mime_type="application/json",
                )
            ]
        except Exception as e:
            raise ValueError(f"Failed to load bundle {bundle_name}: {str(e)}")

    if not resource:
        raise ValueError(f"Unknown resource URI: {uri}. Available: {[r['uri'] for r in RESOURCES]}")

    # Handle file-based resources
    path: Path = resource["path"]

    if not path or not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")

    content = path.read_text(encoding="utf-8")

    return [ReadResourceContents(content=content, mime_type=resource["mimeType"])]


async def _get_template_list() -> str:
    """Get list of available artifact templates with source metadata."""
    try:
        from AgentQMS.tools.core.artifacts.artifact_templates import ArtifactTemplates

        templates_obj = ArtifactTemplates()
        templates_with_metadata = templates_obj.get_available_templates_with_metadata()

        # Return as JSON with enhanced metadata
        return json.dumps(
            {
                "templates": templates_with_metadata,
                "summary": {
                    "total": len(templates_with_metadata),
                    "hardcoded": sum(1 for t in templates_with_metadata if t["source"] == "hardcoded"),
                    "plugin": sum(1 for t in templates_with_metadata if t["source"] == "plugin"),
                    "with_conflicts": sum(1 for t in templates_with_metadata if t.get("has_conflict", False)),
                },
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


async def _get_plugin_artifact_types() -> str:
    """Get all artifact types with comprehensive metadata including source, validation, and template info.

    Returns JSON with structure:
    {
        "artifact_types": [
            {
                "name": "audit",
                "source": "plugin",
                "description": "...",
                "metadata": {...},
                "validation": {...},
                "frontmatter": {...},
                "template_preview": {...},
                "plugin_info": {...},
                "conflicts": {...}
            },
            ...
        ],
        "summary": {
            "total": 11,
            "sources": {"hardcoded": 8, "plugin": 3},
            "validation_enabled": true,
            "last_updated": "2026-01-10T..."
        },
        "metadata": {
            "version": "1.0",
            "plugin_discovery_enabled": true,
            "conflict_detection": true
        }
    }
    """
    try:
        from datetime import datetime
        from AgentQMS.tools.core.artifacts.artifact_templates import ArtifactTemplates

        templates_obj = ArtifactTemplates()
        types_dict = templates_obj._get_available_artifact_types()

        artifact_types_list = []

        for type_name, type_info in sorted(types_dict.items()):
            template = type_info.get("template") or {}
            validation = type_info.get("validation")

            # Build template preview
            template_content = template.get("content_template", "")
            first_300 = template_content[:300] if template_content else ""
            sections = []

            # Extract markdown sections from template
            for line in template_content.split("\n"):
                if line.startswith("#"):
                    sections.append(line.strip())

            artifact_type_obj = {
                "name": type_name,
                "source": type_info.get("source", "unknown"),
                "description": type_info.get("description", ""),
                "category": template.get("frontmatter", {}).get("category", "development"),
                "version": template.get("frontmatter", {}).get("version", "1.0"),
                "metadata": {
                    "filename_pattern": template.get("filename_pattern", ""),
                    "directory": template.get("directory", ""),
                    "template_variables": template.get("_plugin_variables", {}),
                },
                "validation": validation,
                "frontmatter": template.get("frontmatter", {}),
                "template_preview": {
                    "first_300_chars": first_300,
                    "line_count": len(template_content.split("\n")),
                    "sections": sections[:5],  # First 5 sections
                },
                "plugin_info": {},
                "conflicts": {
                    "exists_in_multiple_sources": type_info.get("conflict", False),
                    "conflict_sources": type_info.get("_conflict_note", []),
                },
            }

            # Add plugin info if from plugin
            if type_info.get("source") == "plugin":
                # Try to find plugin path
                plugin_path = ".agentqms/plugins/artifact_types/"
                plugin_files = {
                    "audit": "audit.yaml",
                    "change_request": "change_request.yaml",
                    "ocr_experiment_report": "ocr_experiment.yaml",
                }
                plugin_file = plugin_files.get(type_name, f"{type_name}.yaml")

                artifact_type_obj["plugin_info"] = {
                    "plugin_name": type_name,
                    "plugin_path": f"{plugin_path}{plugin_file}",
                    "plugin_scope": "project",
                }

            artifact_types_list.append(artifact_type_obj)

        # Build summary statistics
        summary = {
            "total": len(artifact_types_list),
            "sources": {
                "hardcoded": sum(1 for t in artifact_types_list if t["source"] == "hardcoded"),
                "plugin": sum(1 for t in artifact_types_list if t["source"] == "plugin"),
                "hardcoded_with_plugin": sum(1 for t in artifact_types_list if t["source"] == "hardcoded (plugin available)"),
            },
            "validation_enabled": True,
            "last_updated": datetime.utcnow().isoformat() + "Z",
        }

        response = {
            "artifact_types": artifact_types_list,
            "summary": summary,
            "metadata": {
                "version": "1.0",
                "plugin_discovery_enabled": True,
                "conflict_detection": True,
                "schema_url": "docs/artifacts/design_documents/2026-01-10_design_plugin_artifact_types_mcp_resource.md",
            },
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        import traceback

        return json.dumps(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "artifact_types": [],
                "summary": {"total": 0},
            },
            indent=2,
        )


async def _get_available_artifact_types() -> list[str]:
    """Get list of available artifact types dynamically from templates and plugins."""
    try:
        from AgentQMS.tools.core.artifacts.artifact_templates import ArtifactTemplates

        templates_obj = ArtifactTemplates()
        types_list = templates_obj.get_available_templates()
        return sorted(types_list) if types_list else _get_fallback_artifact_types()
    except Exception:
        # Fallback to minimal standard types if discovery fails
        return _get_fallback_artifact_types()


def _get_fallback_artifact_types() -> list[str]:
    """
    Fallback artifact types when plugin system is unavailable.

    Returns only the core canonical types defined in validation rules.
    In normal operation, types are loaded dynamically from plugins.
    """
    return [
        "assessment",
        "bug_report",
        "design_document",
        "implementation_plan",
        "vlm_report",
        "walkthrough",
    ]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available AgentQMS tools."""
    artifact_types = await _get_available_artifact_types()

    return [
        Tool(
            name="create_artifact",
            description="Create a new artifact following project standards with auto-validation",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_type": {"type": "string", "enum": artifact_types, "description": "Type of artifact to create"},
                    "name": {"type": "string", "description": "Artifact name in kebab-case"},
                    "title": {"type": "string", "description": "Human-readable title"},
                    "description": {"type": "string", "description": "Optional description"},
                    "tags": {"type": "string", "description": "Comma-separated tags"},
                    "force": {"type": "boolean", "default": False}
                },
                "required": ["artifact_type", "name", "title"]
            }
        ),
        Tool(
            name="validate_artifact",
            description="Validate artifact(s) against naming and structure standards",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to specific artifact to validate (optional)"},
                    "validate_all": {"type": "boolean", "description": "Validate all artifacts instead of single file"},
                    "force": {"type": "boolean", "default": False}
                }
            }
        ),
        Tool(
            name="list_artifact_templates",
            description="List all available artifact templates with details",
            inputSchema={"type": "object", "properties": {"force": {"type": "boolean", "default": False}}}
        ),
        Tool(
            name="check_compliance",
            description="Check overall artifact compliance status and generate report",
            inputSchema={"type": "object", "properties": {"force": {"type": "boolean", "default": False}}}
        ),
        Tool(
            name="get_standard",
            description="Retrieve project standard or rule content by name (fuzzy match)",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the standard"},
                    "force": {"type": "boolean", "default": False}
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="get_context_bundle",
            description="Get relevant context files for a specific task or by bundle name",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string", "description": "Description of the task to get context for"},
                    "task_type": {
                        "type": "string",
                        "description": "Explicit task type/bundle name (optional). If not provided, will auto-detect from description.",
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Token budget limit (default: 32000).",
                    },
                },
                "required": ["task_description"]
            }
        )
    ]


HANDLER_CONTEXT = HandlerContext(agentqms_dir=AGENTQMS_DIR)


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute an AgentQMS tool."""
    try:
        from AgentQMS.tools.core.artifacts.workflow import ArtifactWorkflow

        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return text_payload({"error": f"Unknown tool: {name}"})
        workflow = ArtifactWorkflow(quiet=True)
        safe_arguments = arguments if isinstance(arguments, dict) else {}
        return await handler(workflow, safe_arguments, HANDLER_CONTEXT)
    except Exception as e:
        import traceback

        return text_payload(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "tool": name,
            }
        )


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
