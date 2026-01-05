#!/usr/bin/env python3
"""
Unified Project MCP Server

A single server that combines all project MCP functionality:
- Project Compass resources and tools
- AgentQMS artifact workflows
- Experiment Manager lifecycle tools

This server consolidates multiple MCP services into a single process
to reduce resource overhead and improve efficiency.
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent
from mcp.server.lowlevel.helper_types import ReadResourceContents


# Auto-discover project root
def find_project_root() -> Path:
    """Find project root by locating project_compass/ directory."""
    current = Path(__file__).resolve()
    
    # Search upward
    for parent in current.parents:
        if (parent / "project_compass").exists():
            return parent

    raise RuntimeError("Cannot find project root with project_compass/")

PROJECT_ROOT = find_project_root()


# Project Compass Resources and Tools
def get_compass_resources() -> List[Resource]:
    """Get Project Compass resources."""
    compass_dir = PROJECT_ROOT / "project_compass"
    
    resources = [
        Resource(
            uri="compass://compass.json",
            name="Project Compass State",
            description="Main project state and configuration"
        ),
        Resource(
            uri="compass://session_handover.md", 
            name="Session Handover",
            description="Current session handover document"
        ),
        Resource(
            uri="compass://current_session.yml",
            name="Active Session Context",
            description="Current active session context"
        ),
        Resource(
            uri="compass://uv_lock_state.yml",
            name="Environment Lock State",
            description="Environment lock state"
        ),
        Resource(
            uri="compass://agents.yaml",
            name="Agent Configuration",
            description="Agent configuration file"
        )
    ]
    return resources


def get_compass_tools() -> List[Tool]:
    """Get Project Compass tools."""
    tools = [
        Tool(
            name="env_check",
            description="Validate the uv environment, Python version, and CUDA status against the lock file",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="session_init",
            description="Atomically update the current session context to focus on a specific objective",
            input_schema={
                "type": "object",
                "properties": {
                    "objective": {"type": "string", "description": "The objective to focus on"}
                },
                "required": ["objective"]
            }
        ),
        Tool(
            name="reconcile",
            description="Perform a deep scan of experiment metadata and synchronize the state with actual disk content",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="ocr_convert",
            description="Launch the multi-threaded ETL pipeline to convert datasets to LMDB",
            input_schema={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to input dataset"},
                    "output_path": {"type": "string", "description": "Path for output LMDB"}
                },
                "required": ["input_path", "output_path"]
            }
        ),
        Tool(
            name="ocr_inspect",
            description="Verify the integrity of LMDB datasets",
            input_schema={
                "type": "object",
                "properties": {
                    "dataset_path": {"type": "string", "description": "Path to LMDB dataset"}
                },
                "required": ["dataset_path"]
            }
        )
    ]
    return tools


# AgentQMS Resources and Tools
def get_agentqms_resources() -> List[Resource]:
    """Get AgentQMS resources."""
    resources = [
        Resource(
            uri="agentqms://standards/index",
            name="Standards Hierarchy",
            description="Standards hierarchy and organization"
        ),
        Resource(
            uri="agentqms://standards/artifact_types",
            name="Artifact Types",
            description="Artifact types and their locations"
        ),
        Resource(
            uri="agentqms://standards/workflows", 
            name="Workflow Requirements",
            description="Workflow requirements and standards"
        ),
        Resource(
            uri="agentqms://templates/list",
            name="Available Templates",
            description="List of available artifact templates"
        ),
        Resource(
            uri="agentqms://config/settings",
            name="QMS Settings",
            description="QMS configuration settings"
        )
    ]
    return resources


def get_agentqms_tools() -> List[Tool]:
    """Get AgentQMS tools."""
    tools = [
        Tool(
            name="create_artifact",
            description="Create new artifact following standards",
            input_schema={
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Type of artifact to create"},
                    "name": {"type": "string", "description": "Name of the artifact"},
                    "title": {"type": "string", "description": "Title of the artifact"}
                },
                "required": ["type", "name", "title"]
            }
        ),
        Tool(
            name="validate_artifact",
            description="Validate artifact against standards",
            input_schema={
                "type": "object",
                "properties": {
                    "artifact_path": {"type": "string", "description": "Path to artifact to validate"}
                },
                "required": ["artifact_path"]
            }
        ),
        Tool(
            name="list_artifact_templates",
            description="List available templates",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="check_compliance",
            description="Check overall compliance status",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]
    return tools


# Experiment Manager Resources and Tools
def get_experiment_resources() -> List[Resource]:
    """Get Experiment Manager resources."""
    resources = [
        Resource(
            uri="experiments://agent_interface",
            name="Command Reference",
            description="Experiment manager command reference"
        ),
        Resource(
            uri="experiments://active_list",
            name="Active Experiments",
            description="List of active experiments"
        ),
        Resource(
            uri="experiments://schemas/manifest",
            name="Manifest Schema",
            description="Manifest JSON schema"
        ),
        Resource(
            uri="experiments://schemas/artifact",
            name="Artifact Schema",
            description="Artifact JSON schema"
        )
    ]
    return resources


def get_experiment_tools() -> List[Tool]:
    """Get Experiment Manager tools."""
    tools = [
        Tool(
            name="init_experiment",
            description="Initialize new experiment",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the experiment"},
                    "description": {"type": "string", "description": "Description of the experiment"}
                },
                "required": ["name", "description"]
            }
        ),
        Tool(
            name="get_experiment_status",
            description="Get experiment status",
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "ID of the experiment"}
                },
                "required": ["experiment_id"]
            }
        ),
        Tool(
            name="add_task",
            description="Add task to experiment",
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "ID of the experiment"},
                    "task": {"type": "string", "description": "Task description"}
                },
                "required": ["experiment_id", "task"]
            }
        ),
        Tool(
            name="log_insight",
            description="Log insight/decision/failure",
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "ID of the experiment"},
                    "type": {"type": "string", "description": "Type of insight (decision, failure, insight)"},
                    "content": {"type": "string", "description": "Content of the insight"}
                },
                "required": ["experiment_id", "type", "content"]
            }
        ),
        Tool(
            name="sync_experiment",
            description="Sync experiment to database",
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "ID of the experiment"}
                },
                "required": ["experiment_id"]
            }
        )
    ]
    return tools


# Create unified server
server = Server("unified-project-mcp", "Unified Project MCP Server for Compass, AgentQMS, and Experiments")


# Resource handler
@server.check_resource_exists()
async def check_resource_exists(resource: Resource) -> bool:
    """Check if a resource exists."""
    # Handle compass resources
    if resource.uri.startswith("compass://"):
        compass_dir = PROJECT_ROOT / "project_compass"
        resource_path = resource.uri.replace("compass://", "")
        full_path = compass_dir / resource_path
        return full_path.exists()
    
    # Handle agentqms resources
    elif resource.uri.startswith("agentqms://"):
        # These are virtual resources, always exist
        return True
    
    # Handle experiment resources
    elif resource.uri.startswith("experiments://"):
        # These are virtual resources, always exist
        return True
    
    return False


@server.read_resource()
async def read_resource(resource: Resource) -> List[TextContent]:
    """Read resource content."""
    # Handle compass resources
    if resource.uri.startswith("compass://"):
        compass_dir = PROJECT_ROOT / "project_compass"
        resource_path = resource.uri.replace("compass://", "")
        full_path = compass_dir / resource_path
        
        if full_path.exists():
            content = full_path.read_text()
            return [TextContent(
                uri=resource.uri,
                text=content,
                mime_type="text/plain"
            )]
        else:
            return [TextContent(
                uri=resource.uri,
                text=f"Resource not found: {resource_path}",
                mime_type="text/plain"
            )]
    
    # Handle agentqms resources
    elif resource.uri.startswith("agentqms://"):
        # Return placeholder content for virtual resources
        return [TextContent(
            uri=resource.uri,
            text=f"Virtual AgentQMS resource: {resource.uri}",
            mime_type="text/plain"
        )]
    
    # Handle experiment resources
    elif resource.uri.startswith("experiments://"):
        # Return placeholder content for virtual resources
        return [TextContent(
            uri=resource.uri,
            text=f"Virtual Experiment resource: {resource.uri}",
            mime_type="text/plain"
        )]
    
    # Unknown resource type
    return [TextContent(
        uri=resource.uri,
        text=f"Unknown resource type: {resource.uri}",
        mime_type="text/plain"
    )]


# Tool handlers
@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Handle tool calls."""
    # Compass tools
    if name == "env_check":
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "project_compass" / "env_check.py")
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "session_init":
        objective = arguments.get("objective", "")
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "project_compass" / "session_init.py"),
            objective
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "reconcile":
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "project_compass" / "reconcile.py")
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "ocr_convert":
        input_path = arguments.get("input_path", "")
        output_path = arguments.get("output_path", "")
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "project_compass" / "ocr_convert.py"),
            input_path, output_path
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "ocr_inspect":
        dataset_path = arguments.get("dataset_path", "")
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "project_compass" / "ocr_inspect.py"),
            dataset_path
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    # AgentQMS tools
    elif name == "create_artifact":
        artifact_type = arguments.get("type", "")
        name = arguments.get("name", "")
        title = arguments.get("title", "")
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "AgentQMS" / "bin" / "create-artifact.py"),
            "--type", artifact_type,
            "--name", name,
            "--title", title
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "validate_artifact":
        artifact_path = arguments.get("artifact_path", "")
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "AgentQMS" / "bin" / "validate-artifact.py"),
            artifact_path
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "list_artifact_templates":
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "AgentQMS" / "bin" / "list-templates.py")
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "check_compliance":
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "AgentQMS" / "bin" / "compliance-check.py")
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    # Experiment tools
    elif name == "init_experiment":
        name = arguments.get("name", "")
        description = arguments.get("description", "")
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "experiment_manager" / "etk.py"),
            "init", name, description
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "get_experiment_status":
        experiment_id = arguments.get("experiment_id", "")
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "experiment_manager" / "etk.py"),
            "status", experiment_id
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "add_task":
        experiment_id = arguments.get("experiment_id", "")
        task = arguments.get("task", "")
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "experiment_manager" / "etk.py"),
            "add-task", experiment_id, task
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "log_insight":
        experiment_id = arguments.get("experiment_id", "")
        insight_type = arguments.get("type", "")
        content = arguments.get("content", "")
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "experiment_manager" / "etk.py"),
            "log", experiment_id, insight_type, content
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    elif name == "sync_experiment":
        experiment_id = arguments.get("experiment_id", "")
        result = subprocess.run([
            "uv", "run", "python", 
            str(PROJECT_ROOT / "experiment_manager" / "etk.py"),
            "sync", experiment_id
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    
    # Unknown tool
    else:
        return json.dumps({
            "error": f"Unknown tool: {name}",
            "available_tools": [
                "env_check", "session_init", "reconcile", "ocr_convert", "ocr_inspect",  # compass
                "create_artifact", "validate_artifact", "list_artifact_templates", "check_compliance",  # agentqms
                "init_experiment", "get_experiment_status", "add_task", "log_insight", "sync_experiment"  # experiments
            ]
        })


async def main():
    """Run the unified MCP server."""
    # Register all resources
    for resource in get_compass_resources():
        server.get_resource(resource)
    
    for resource in get_agentqms_resources():
        server.get_resource(resource)
    
    for resource in get_experiment_resources():
        server.get_resource(resource)
    
    # Register all tools
    for tool in get_compass_tools():
        server.register_tool(tool)
    
    for tool in get_agentqms_tools():
        server.register_tool(tool)
    
    for tool in get_experiment_tools():
        server.register_tool(tool)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())