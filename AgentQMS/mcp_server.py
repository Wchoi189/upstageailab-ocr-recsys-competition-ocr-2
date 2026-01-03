#!/usr/bin/env python3
"""
AgentQMS MCP Server

Exposes artifact workflow tools and standards as MCP resources and tools.

Resources:
- agentqms://standards/index - Standards hierarchy
- agentqms://standards/artifact_types - Artifact types and locations
- agentqms://standards/workflows - Workflow requirements
- agentqms://templates/list - Available templates
- agentqms://config/settings - QMS settings

Tools:
- create_artifact - Create new artifact following standards
- validate_artifact - Validate artifact against standards
- list_artifact_templates - List available templates
- check_compliance - Check overall compliance status
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import ReadResourceResult, Resource, TextContent, TextResourceContents, Tool
from mcp.server.lowlevel.helper_types import ReadResourceContents


# Auto-discover project root
def find_project_root() -> Path:
    """Find project root by locating AgentQMS/ directory."""
    current = Path(__file__).resolve().parent

    # We're already in AgentQMS/
    if current.name == "AgentQMS":
        return current.parent

    # Search upward
    for parent in current.parents:
        if (parent / "AgentQMS").exists():
            return parent

    raise RuntimeError("Cannot find project root with AgentQMS/")


PROJECT_ROOT = find_project_root()
AGENTQMS_DIR = PROJECT_ROOT / "AgentQMS"

# Add AgentQMS to Python path for imports
sys.path.insert(0, str(PROJECT_ROOT))


# Define available resources
RESOURCES = [
    {
        "uri": "agentqms://standards/index",
        "name": "Standards Index",
        "description": "Complete standards hierarchy with 4-tier system (SST, Framework, Agents, Workflows)",
        "path": AGENTQMS_DIR / "standards" / "INDEX.yaml",
        "mimeType": "application/x-yaml",
    },
    {
        "uri": "agentqms://standards/artifact_types",
        "name": "Artifact Types",
        "description": "Allowed artifact types, locations, naming conventions, and prohibited types",
        "path": AGENTQMS_DIR / "standards" / "tier1-sst" / "artifact-types.yaml",
        "mimeType": "application/x-yaml",
    },
    {
        "uri": "agentqms://standards/workflows",
        "name": "Workflow Requirements",
        "description": "Workflow validation protocols and requirements",
        "path": AGENTQMS_DIR / "standards" / "tier1-sst" / "workflow-requirements.yaml",
        "mimeType": "application/x-yaml",
    },
    {
        "uri": "agentqms://templates/list",
        "name": "Template Catalog",
        "description": "Available artifact templates (dynamically generated)",
        "path": None,  # Dynamic content
        "mimeType": "application/json",
    },
    {
        "uri": "agentqms://config/settings",
        "name": "QMS Settings",
        "description": "Framework configuration, paths, validation rules, tool mappings",
        "path": AGENTQMS_DIR / "config" / "settings.yaml",
        "mimeType": "application/x-yaml",
    },
]


# Create MCP server
app = Server("agentqms")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List all available AgentQMS resources."""
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
    """Read content of an AgentQMS resource."""
    # Find matching resource
    uri = str(uri).strip()
    resource = next((r for r in RESOURCES if r["uri"] == uri), None)

    if not resource:
        raise ValueError(f"Unknown resource URI: {uri}. Available: {[r['uri'] for r in RESOURCES]}")

    # Handle dynamic template list
    if uri == "agentqms://templates/list":
        content = await _get_template_list()
        return [
            ReadResourceContents(
                content=content,
                mime_type="application/json"
            )
        ]

    # Handle file-based resources
    path: Path = resource["path"]

    if not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")

    content = path.read_text(encoding="utf-8")

    return [
        ReadResourceContents(
            content=content,
            mime_type=resource["mimeType"]
        )
    ]


async def _get_template_list() -> str:
    """Get list of available artifact templates."""
    try:
        from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow

        workflow = ArtifactWorkflow(quiet=True)
        templates = workflow.get_available_templates()

        # Return as JSON
        return json.dumps({"templates": templates}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available AgentQMS tools."""
    return [
        Tool(
            name="create_artifact",
            description="Create a new artifact following project standards with auto-validation",
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
                        "description": "Type of artifact to create",
                    },
                    "name": {
                        "type": "string",
                        "description": "Artifact name in kebab-case (e.g., 'api-refactor')",
                    },
                    "title": {
                        "type": "string",
                        "description": "Human-readable title for the artifact",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of the artifact",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags (e.g., 'optimization,performance')",
                    },
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
                    "file_path": {
                        "type": "string",
                        "description": "Path to specific artifact to validate (optional)",
                    },
                    "validate_all": {
                        "type": "boolean",
                        "description": "Validate all artifacts instead of single file",
                    },
                },
            },
        ),
        Tool(
            name="list_artifact_templates",
            description="List all available artifact templates with details",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="check_compliance",
            description="Check overall artifact compliance status and generate report",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_standard",
            description="Retrieve project standard or rule content by name (fuzzy match)",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the standard (e.g. 'naming-conventions', 'artifact-types')",
                    },
                },
                "required": ["name"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute an AgentQMS tool."""
    try:
        from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow

        workflow = ArtifactWorkflow(quiet=True)

        if name == "create_artifact":
            artifact_type = arguments["artifact_type"]
            art_name = arguments["name"]
            title = arguments["title"]

            kwargs = {}
            if "description" in arguments:
                kwargs["description"] = arguments["description"]
            if "tags" in arguments:
                kwargs["tags"] = arguments["tags"]

            file_path = workflow.create_artifact(
                artifact_type,
                art_name,
                title,
                **kwargs
            )

            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "file_path": file_path,
                        "message": f"Created {artifact_type}: {file_path}",
                    }, indent=2)
                )
            ]

        elif name == "validate_artifact":
            if arguments.get("validate_all"):
                success = workflow.validate_all()
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "success": success,
                            "message": "Validation complete. Check output above.",
                        }, indent=2)
                    )
                ]
            elif "file_path" in arguments:
                file_path = arguments["file_path"]
                success = workflow.validate_artifact(file_path)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "success": success,
                            "file_path": file_path,
                        }, indent=2)
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "error": "Must specify either file_path or validate_all=true",
                        }, indent=2)
                    )
                ]

        elif name == "list_artifact_templates":
            templates = workflow.get_available_templates()
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "templates": templates,
                    }, indent=2)
                )
            ]

        elif name == "check_compliance":
            report = workflow.check_compliance()
            return [
                TextContent(
                    type="text",
                    text=json.dumps(report, indent=2)
                )
            ]

        elif name == "get_standard":
            query = arguments["name"].lower()
            standards_dir = AGENTQMS_DIR / "standards"
            matches = []

            # Recursive search for .yaml and .md files
            if standards_dir.exists():
                for path in standards_dir.rglob("*"):
                    if path.is_file() and path.suffix in [".yaml", ".md", ".json"]:
                        if query in path.stem.lower():
                            matches.append(path)

            if not matches:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": f"No standards found matching '{query}'"}, indent=2)
                    )
                ]

            if len(matches) == 1:
                 content = matches[0].read_text(encoding="utf-8")
                 return [
                     TextContent(
                         type="text",
                         text=f"Standard: {matches[0].name}\nLocation: {matches[0]}\n\n{content}"
                     )
                 ]

            # Multiple matches
            names = [str(p.relative_to(standards_dir)) for p in matches]
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "message": "Multiple matches found. Please specify:",
                        "matches": names
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
