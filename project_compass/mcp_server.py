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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

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


def _read_resource_sync(uri: str) -> list[ReadResourceContents]:
    """Sync implementation of read_resource."""
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


@app.read_resource()
async def read_resource(uri: str) -> list[ReadResourceContents]:
    """Read content of a vessel resource."""
    uri = str(uri).strip()
    return await asyncio.to_thread(_read_resource_sync, uri)


# ============ Tools ============

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all pulse management tools."""
    return [
        Tool(
            name="compass_meta_pulse",
            description="Pulse management tools: init (initialize pulse), sync (register artifact), export (archive pulse), status (get status), checkpoint (update token burden)",
            inputSchema={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["init", "sync", "export", "status", "checkpoint"],
                        "description": "Pulse operation type",
                    },
                    "pulse_id": {
                        "type": "string",
                        "description": "Pulse ID in domain-action-target format (required for init)",
                    },
                    "objective": {
                        "type": "string",
                        "description": "Work objective (20-500 characters, required for init)",
                    },
                    "milestone_id": {
                        "type": "string",
                        "description": "Milestone ID from star-chart (required for init/sync)",
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["detection", "recognition", "kie", "integration"],
                        "description": "Active pipeline phase (optional for init)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Artifact path relative to pulse_staging/artifacts/ (required for sync)",
                    },
                    "artifact_type": {
                        "type": "string",
                        "enum": ["design", "research", "walkthrough", "implementation_plan", "bug_report", "audit"],
                        "description": "Type of artifact (required for sync)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Skip staging audit (NOT RECOMMENDED, for export)",
                    },
                    "token_burden": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Update token burden level (for checkpoint)",
                    },
                },
                "required": ["kind"],
            },
        ),
        Tool(
            name="compass_meta_spec",
            description="Spec Kit tools: constitution (establish principles), specify (create spec), plan (implementation plan), tasks (generate tasks)",
            inputSchema={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["constitution", "specify", "plan", "tasks"],
                        "description": "Spec operation type",
                    },
                    "principles": {
                        "type": "string",
                        "description": "Project principles and guidelines (required for constitution)",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Scope and context for specification (for specify)",
                    },
                    "requirements": {
                        "type": "string",
                        "description": "Key requirements to capture (for specify)",
                    },
                    "approach": {
                        "type": "string",
                        "description": "Implementation approach and strategy (for plan)",
                    },
                    "focus_area": {
                        "type": "string",
                        "description": "Specific area to focus task generation on (for tasks)",
                    },
                },
                "required": ["kind"],
            },
        ),
    ]


# ============ Tool Handlers ============

def get_tool_context():
    """Import and return necessary context objects/functions lazily."""
    try:
        from project_compass.src.core import PulseManager, VesselPaths
        from project_compass.src.pulse_exporter import export_pulse, register_artifact
    except ImportError:
        from src.core import PulseManager, VesselPaths
        from src.pulse_exporter import export_pulse, register_artifact

    paths = VesselPaths()
    manager = PulseManager(paths)
    return manager, paths, export_pulse, register_artifact


async def handle_pulse_init(arguments: Dict[str, Any], manager: Any, **kwargs) -> List[TextContent]:
    success, message = await asyncio.to_thread(
        manager.init_pulse,
        pulse_id=arguments["pulse_id"],
        objective=arguments["objective"],
        milestone_id=arguments["milestone_id"],
        phase=arguments.get("phase", "kie"),
    )
    result = {"success": success, "message": message}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_pulse_sync(arguments: Dict[str, Any], paths: Any, register_artifact: Any, **kwargs) -> List[TextContent]:
    success, message = await asyncio.to_thread(
        register_artifact,
        state_path=paths.vessel_state,
        artifact_path=arguments["path"],
        artifact_type=arguments["artifact_type"],
        milestone_id=arguments.get("milestone_id"),
    )
    result = {"success": success, "message": message}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_pulse_export(paths: Any, export_pulse: Any, **kwargs) -> List[TextContent]:
    result = await asyncio.to_thread(
        export_pulse,
        state_path=paths.vessel_state,
        staging_path=paths.staging_dir,
        history_path=paths.history_dir,
    )
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_pulse_status(manager: Any, **kwargs) -> List[TextContent]:
    status = await asyncio.to_thread(manager.get_pulse_status)
    return [TextContent(type="text", text=json.dumps(status, indent=2))]


async def handle_pulse_checkpoint(arguments: Dict[str, Any], manager: Any, **kwargs) -> List[TextContent]:
    state = await asyncio.to_thread(manager.load_state)
    if not state.active_pulse:
        return [TextContent(type="text", text=json.dumps({"error": "No active pulse"}))]

    if arguments.get("token_burden"):
        state.active_pulse.token_burden = arguments["token_burden"]
        await asyncio.to_thread(manager.save_state, state)

    assessment = {
        "pulse_id": state.active_pulse.pulse_id,
        "artifact_count": len(state.active_pulse.artifacts),
        "token_burden": state.active_pulse.token_burden,
        "recommendation": "export" if state.active_pulse.token_burden == "high" else "continue",
    }
    return [TextContent(type="text", text=json.dumps(assessment, indent=2))]


async def handle_spec_constitution(arguments: Dict[str, Any], paths: Any, register_artifact: Any, **kwargs) -> List[TextContent]:
    def _sync_impl():
        try:
            staging_dir = STAGING_DIR / "artifacts"
            constitution_file = staging_dir / "constitution.md"

            constitution_content = f"# Project Constitution\n\n## Principles\n{arguments['principles']}\n\n## Established\nDate: {datetime.now().isoformat()}\nTool: Project Compass v2\n"

            constitution_file.parent.mkdir(parents=True, exist_ok=True)
            constitution_file.write_text(constitution_content)
            
            success, message = register_artifact(
                state_path=paths.vessel_state,
                artifact_path="constitution.md",
                artifact_type="requirements",
                milestone_id=None,
            )
            return {
                "success": True, 
                "message": "Project constitution established and registered", 
                "file": str(constitution_file), 
                "artifact_registered": success
            }
        except Exception as e:
            return {"success": False, "message": f"Error establishing constitution: {str(e)}"}

    response = await asyncio.to_thread(_sync_impl)
    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def handle_spec_specify(arguments: Dict[str, Any], paths: Any, register_artifact: Any, **kwargs) -> List[TextContent]:
    def _sync_impl():
        try:
            staging_dir = STAGING_DIR / "artifacts"
            spec_file = staging_dir / "specification.md"
            
            scope_text = arguments.get("scope", "General project scope")
            requirements_text = arguments.get("requirements", "TBD")
            
            spec_content = f"# Project Specification\n\n## Scope\n{scope_text}\n\n## Requirements\n{requirements_text}\n\n## Status\n- Created: {datetime.now().isoformat()}\n- Tool: Project Compass v2\n- Status: Draft\n"

            spec_file.parent.mkdir(parents=True, exist_ok=True)
            spec_file.write_text(spec_content)
            
            success, message = register_artifact(
                state_path=paths.vessel_state,
                artifact_path="specification.md",
                artifact_type="specification",
                milestone_id=None,
            )
            return {
                "success": True, 
                "message": "Specification created and registered", 
                "file": str(spec_file), 
                "artifact_registered": success
            }
        except Exception as e:
            return {"success": False, "message": f"Error creating specification: {str(e)}"}

    response = await asyncio.to_thread(_sync_impl)
    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def handle_spec_plan(arguments: Dict[str, Any], paths: Any, register_artifact: Any, **kwargs) -> List[TextContent]:
    def _sync_impl():
        try:
            staging_dir = STAGING_DIR / "artifacts"
            plan_file = staging_dir / "implementation_plan.md"
            
            approach_text = arguments.get("approach", "Standard implementation approach")
            
            plan_content = f"# Implementation Plan\n\n## Approach\n{approach_text}\n\n## High-Level Steps\n1. **Analysis Phase**\n   - Requirements review\n   - Architecture design\n   - Risk assessment\n\n2. **Development Phase**\n   - Core implementation\n   - Testing strategy\n   - Integration planning\n\n3. **Validation Phase**\n   - Quality assurance\n   - Performance testing\n   - Deployment preparation\n\n## Success Criteria\n- All requirements met\n- Code quality standards maintained\n- Performance benchmarks achieved\n\n## Timeline\nTBD - To be determined based on scope and resources\n\n## Status\n- Created: {datetime.now().isoformat()}\n- Tool: Project Compass v2\n- Status: Draft\n"

            plan_file.parent.mkdir(parents=True, exist_ok=True)
            plan_file.write_text(plan_content)
            
            success, message = register_artifact(
                state_path=paths.vessel_state,
                artifact_path="implementation_plan.md",
                artifact_type="implementation_plan",
                milestone_id=None,
            )
            return {
                "success": True, 
                "message": "Implementation plan created and registered", 
                "file": str(plan_file), 
                "artifact_registered": success
            }
        except Exception as e:
            return {"success": False, "message": f"Error creating implementation plan: {str(e)}"}

    response = await asyncio.to_thread(_sync_impl)
    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def handle_spec_tasks(arguments: Dict[str, Any], paths: Any, register_artifact: Any, **kwargs) -> List[TextContent]:
    def _sync_impl():
        try:
            staging_dir = STAGING_DIR / "artifacts"
            tasks_file = staging_dir / "tasks.md"
            
            focus_text = arguments.get("focus_area", "General development tasks")
            
            tasks_content = f"# Actionable Tasks\n\n## Focus Area: {focus_text}\n\n## Task Breakdown\n\n### Phase 1: Foundation\n- [ ] Set up development environment\n- [ ] Initialize project structure\n- [ ] Configure CI/CD pipeline\n- [ ] Establish coding standards\n\n### Phase 2: Core Development\n- [ ] Implement core functionality\n- [ ] Write unit tests\n- [ ] Integration testing\n- [ ] Documentation\n\n### Phase 3: Validation & Deployment\n- [ ] Performance testing\n- [ ] Security review\n- [ ] User acceptance testing\n- [ ] Production deployment\n\n## Priority Matrix\n- **High Priority**: Environment setup, core functionality\n- **Medium Priority**: Testing, documentation\n- **Low Priority**: Optimization, advanced features\n\n## Status\n- Created: {datetime.now().isoformat()}\n- Tool: Project Compass v2\n- Status: Draft\n"

            tasks_file.parent.mkdir(parents=True, exist_ok=True)
            tasks_file.write_text(tasks_content)
            
            success, message = register_artifact(
                state_path=paths.vessel_state,
                artifact_path="tasks.md",
                artifact_type="implementation_plan",
                milestone_id=None,
            )
            return {
                "success": True, 
                "message": "Tasks generated and registered", 
                "file": str(tasks_file), 
                "artifact_registered": success
            }
        except Exception as e:
            return {"success": False, "message": f"Error generating tasks: {str(e)}"}

    response = await asyncio.to_thread(_sync_impl)
    return [TextContent(type="text", text=json.dumps(response, indent=2))]


TOOL_HANDLERS = {
    "pulse_init": handle_pulse_init,
    "pulse_sync": handle_pulse_sync,
    "pulse_export": handle_pulse_export,
    "pulse_status": handle_pulse_status,
    "pulse_checkpoint": handle_pulse_checkpoint,
    "spec_constitution": handle_spec_constitution,
    "spec_specify": handle_spec_specify,
    "spec_plan": handle_spec_plan,
    "spec_tasks": handle_spec_tasks,
}


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute a pulse tool."""
    
    # --- Meta-Tool Handlers (Router Pattern) ---
    if name in ["compass_meta_pulse", "compass_meta_spec"]:
        args = arguments.copy()  # Safe copy
        kind = args.pop("kind", "")
        
        try:
            if name == "compass_meta_pulse":
                 from project_compass.src.router import route_pulse
                 routing = route_pulse(kind, args)
            else:
                 from project_compass.src.router import route_spec
                 routing = route_spec(kind, args)
                 
            return await call_tool(routing["tool_name"], routing["arguments"])
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    # --- Individual Tool Handlers ---
    
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        raise ValueError(f"Unknown tool: {name}")
    
    manager, paths, export_pulse, register_artifact = get_tool_context()
    
    return await handler(
        arguments=arguments,
        manager=manager,
        paths=paths,
        export_pulse=export_pulse,
        register_artifact=register_artifact
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
