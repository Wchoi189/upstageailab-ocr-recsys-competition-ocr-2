"""MCP tool handlers for AgentQMS server dispatch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from mcp.types import TextContent


@dataclass(frozen=True)
class HandlerContext:
    """Static context shared by MCP tool handlers."""

    agentqms_dir: Path


ToolHandler = Callable[[Any, dict[str, Any], HandlerContext], Awaitable[list[TextContent]]]


def text_payload(payload: dict[str, Any]) -> list[TextContent]:
    """Return standard JSON text response payload."""
    return [TextContent(type="text", text=json.dumps(payload, indent=2))]


async def handle_create_artifact(workflow: Any, arguments: dict[str, Any], ctx: HandlerContext) -> list[TextContent]:
    _ = ctx
    artifact_type = arguments["artifact_type"]
    art_name = arguments["name"]
    title = arguments["title"]

    kwargs = {}
    if "description" in arguments:
        kwargs["description"] = arguments["description"]
    if "tags" in arguments:
        kwargs["tags"] = arguments["tags"]

    file_path = workflow.create_artifact(artifact_type, art_name, title, **kwargs)
    return text_payload(
        {
            "success": True,
            "file_path": file_path,
            "message": f"Created {artifact_type}: {file_path}",
        }
    )


async def handle_validate_artifact(workflow: Any, arguments: dict[str, Any], ctx: HandlerContext) -> list[TextContent]:
    _ = ctx
    if arguments.get("validate_all"):
        success = workflow.validate_all()
        return text_payload({"success": success, "message": "Validation complete. Check output above."})
    if "file_path" in arguments:
        file_path = arguments["file_path"]
        success = workflow.validate_artifact(file_path)
        return text_payload({"success": success, "file_path": file_path})
    return text_payload({"error": "Must specify either file_path or validate_all=true"})


async def handle_list_artifact_templates(
    workflow: Any, arguments: dict[str, Any], ctx: HandlerContext
) -> list[TextContent]:
    _ = arguments
    _ = ctx
    return text_payload({"templates": workflow.get_available_templates()})


async def handle_check_compliance(workflow: Any, arguments: dict[str, Any], ctx: HandlerContext) -> list[TextContent]:
    _ = arguments
    _ = ctx
    report = workflow.check_compliance()
    return [TextContent(type="text", text=json.dumps(report, indent=2))]


async def handle_get_standard(workflow: Any, arguments: dict[str, Any], ctx: HandlerContext) -> list[TextContent]:
    _ = workflow
    query = arguments["name"].lower()
    specs_dir = ctx.agentqms_dir / "specs"
    matches = []

    if specs_dir.exists():
        for path in specs_dir.rglob("*"):
            if path.is_file() and path.suffix in [".md", ".yaml", ".json"] and query in path.stem.lower():
                matches.append(path)

    if not matches:
        return text_payload({"error": f"No specs found matching '{query}'"})
    if len(matches) == 1:
        content = matches[0].read_text(encoding="utf-8")
        return [TextContent(type="text", text=f"Spec: {matches[0].name}\nLocation: {matches[0]}\n\n{content}")]

    names = [str(p.relative_to(ctx.agentqms_dir)) for p in matches]
    return text_payload({"message": "Multiple matches found. Please specify:", "matches": names})


async def handle_get_context_bundle(
    workflow: Any, arguments: dict[str, Any], ctx: HandlerContext
) -> list[TextContent]:
    _ = workflow
    _ = ctx
    task_description = arguments["task_description"]

    from AgentQMS.tools.core.context.context_bundle import auto_suggest_context

    if "budget" in arguments:
        from AgentQMS.tools.core.context.context_bundle import _ENGINE

        _ENGINE.max_tokens = int(arguments["budget"])

    suggestion = auto_suggest_context(task_description)
    return text_payload(
        {
            "task_description": task_description,
            "files": suggestion["bundle_files"],
            "detected": suggestion,
            "token_usage": suggestion.get("token_usage"),
            "stats": {
                "total_files": len(suggestion["bundle_files"]),
                "total_tokens": suggestion.get("token_usage", {}).get("total_tokens", 0),
            },
        }
    )


TOOL_HANDLERS: dict[str, ToolHandler] = {
    "create_artifact": handle_create_artifact,
    "validate_artifact": handle_validate_artifact,
    "list_artifact_templates": handle_list_artifact_templates,
    "check_compliance": handle_check_compliance,
    "get_standard": handle_get_standard,
    "get_context_bundle": handle_get_context_bundle,
}
