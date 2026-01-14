#!/usr/bin/env python3
"""
Unified Project MCP Server (SSE + Stdio)

A single server that combines all project MCP functionality:
- Project Compass resources and tools
- AgentQMS artifact workflows and standards
- Experiment Manager lifecycle tools
- Middleware enforcement (Telemetry, Compliance, Proactive Feedback)
- Context Auto-Loading

Supports both Stdio (for local desktop) and SSE (for cloud/remote).
"""

import asyncio
import json
import sys
import time
import argparse
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.lowlevel.helper_types import ReadResourceContents

# Add project root to path
# NOTE: We assume the environment is set up correctly (uv pip install -e .).
# If AgentQMS is not found, it means the environment is invalid.

# --- Imports ---
from AgentQMS.tools.utils.config_loader import ConfigLoader
from AgentQMS.tools.utils.paths import get_project_root
from AgentQMS.middleware.telemetry import TelemetryPipeline, PolicyViolation
from AgentQMS.middleware.policies import (
    RedundancyInterceptor,
    ComplianceInterceptor,
    FileOperationInterceptor,
    StandardsInterceptor,
    ProactiveFeedbackInterceptor
)

PROJECT_ROOT = get_project_root()
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Initialize Middleware
feedback_interceptor = ProactiveFeedbackInterceptor()
TELEMETRY_PIPELINE = TelemetryPipeline([
    RedundancyInterceptor(),
    ComplianceInterceptor(),
    FileOperationInterceptor(),
    StandardsInterceptor(),
    feedback_interceptor
])

app = Server("unified_project")
config_loader = ConfigLoader(cache_size=5)
SESSION_ID = str(uuid.uuid4())

# --- Telemetry ---
TELEMETRY_FILE = PROJECT_ROOT / "AgentQMS" / ".mcp-telemetry.jsonl"

def log_telemetry_event(event: dict) -> None:
    """Append a telemetry event to the JSONL log."""
    try:
        with open(TELEMETRY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass  # Don't crash on telemetry failure

# --- Aggregation Variables ---
TOOLS_DEFINITIONS: list[dict] = []
RESOURCES_CONFIG: list[dict] = []


# --- Resource Handling ---

async def load_resources_from_servers() -> list[dict]:
    """Aggregate resources from all MCP servers + unified config."""
    import importlib
    resources = []

    # Servers to aggregate
    server_modules = [
        ("AgentQMS.mcp_server", "agentqms"),
        ("project_compass.mcp_server", "compass"),
        ("experiment_manager.mcp_server", "experiments"),
    ]

    for mod_name, scheme in server_modules:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "list_resources"):
                server_res = await mod.list_resources()
                for res in server_res:
                    resources.append({
                        "uri": res.uri,
                        "name": res.name,
                        "description": res.description,
                        "mimeType": res.mimeType,
                        "path": None # Delegated
                    })
        except ImportError:
            pass # Skip missing modules

    # Unified config resources
    # Use PROJECT_ROOT based path for robustness
    config_path = PROJECT_ROOT / "scripts/mcp/config/resources.yaml"
    if config_path.exists():
        config_resources = config_loader.get_config(config_path, defaults=[])
        for res in config_resources:
            if res.get("path"):
                res["path"] = PROJECT_ROOT / res["path"]
            resources.append(res)

    # Dynamic bundles
    resources.append({
        "uri": "bundle://list",
        "name": "Context Bundles",
        "description": "List available context bundles for AI agents",
        "mimeType": "application/json",
        "path": None
    })

    # Telemetry resource
    resources.append({
        "uri": "mcp://telemetry/recent",
        "name": "Recent MCP Calls",
        "description": "Recent tool call telemetry (last 50 calls)",
        "mimeType": "application/json",
        "path": None
    })

    return resources

@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(uri=res["uri"], name=res["name"], description=res.get("description", ""), mimeType=res["mimeType"])
        for res in RESOURCES_CONFIG
    ]

@app.read_resource()
async def read_resource(uri: str) -> list[ReadResourceContents]:
    start_time = time.time()
    event = {
        "timestamp": datetime.now().isoformat(),
        # Use a distinguishable name for the resource read event
        "tool_name": f"read_resource:{uri}",
        "session_id": SESSION_ID,
        "input_tokens": len(uri) // 4,  # Approx input
    }

    try:
        result = await _read_resource_impl(uri)

        # Calculate size of content for token estimation
        output_len = sum(len(c.content) if isinstance(c.content, str) else len(str(c.content)) for c in result)

        duration_ms = (time.time() - start_time) * 1000
        event["status"] = "success"
        event["duration_ms"] = round(duration_ms, 2)
        event["output_tokens"] = output_len // 4
        log_telemetry_event(event)

        return result

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        event["status"] = "error"
        event["duration_ms"] = round(duration_ms, 2)
        event["error"] = str(e)[:200]
        event["output_tokens"] = 0
        log_telemetry_event(event)
        # Re-raise or return error content - MCP expects list of contents or error
        # Re-raising allows the client to see the error properly
        raise e

async def _read_resource_impl(uri: str) -> list[ReadResourceContents]:
    uri = str(uri).strip()
    import importlib

    # Route by scheme
    if "://" in uri:
        scheme = uri.split("://")[0]
        if scheme == "agentqms":
            mod = importlib.import_module("AgentQMS.mcp_server")
            return await mod.read_resource(uri)
        elif scheme == "compass":
            mod = importlib.import_module("project_compass.mcp_server")
            return await mod.read_resource(uri)
        elif scheme == "experiments":
            mod = importlib.import_module("experiment_manager.mcp_server")
            return await mod.read_resource(uri)
        elif scheme == "bundle":
             # Handle context bundles
            from AgentQMS.tools.core.context_bundle import list_available_bundles, get_context_bundle
            try:
                if uri == "bundle://list":
                    bundles = list_available_bundles()
                    return [ReadResourceContents(content=json.dumps(bundles, indent=2), mime_type="application/json")]
                else:
                    bundle_name = uri.replace("bundle://", "")
                    files = get_context_bundle(task_description=bundle_name, task_type=bundle_name)
                    return [ReadResourceContents(content=json.dumps({"bundle": bundle_name, "files": files}, indent=2), mime_type="application/json")]
            except Exception as e:
                return [ReadResourceContents(content=json.dumps({"error": str(e)}), mime_type="application/json")]
        elif scheme == "mcp":
            # Handle MCP internal resources
            if uri == "mcp://telemetry/recent":
                try:
                    if TELEMETRY_FILE.exists():
                        lines = TELEMETRY_FILE.read_text(encoding="utf-8").strip().split("\n")
                        # Get last 50 entries
                        recent = [json.loads(line) for line in lines[-50:] if line.strip()]
                        # Summary stats
                        success = sum(1 for e in recent if e.get("status") == "success")
                        errors = sum(1 for e in recent if e.get("status") == "error")
                        violations = sum(1 for e in recent if e.get("status") == "policy_violation")
                        avg_duration = sum(e.get("duration_ms", 0) for e in recent) / max(len(recent), 1)
                        result = {
                            "summary": {
                                "total": len(recent),
                                "success": success,
                                "errors": errors,
                                "policy_violations": violations,
                                "avg_duration_ms": round(avg_duration, 2)
                            },
                            "recent_calls": recent
                        }
                        return [ReadResourceContents(content=json.dumps(result, indent=2), mime_type="application/json")]
                    else:
                        return [ReadResourceContents(content=json.dumps({"summary": {"total": 0}, "recent_calls": []}), mime_type="application/json")]
                except Exception as e:
                    return [ReadResourceContents(content=json.dumps({"error": str(e)}), mime_type="application/json")]

    # Static resources
    resource = next((r for r in RESOURCES_CONFIG if r["uri"] == uri), None)
    if not resource:
        raise ValueError(f"Unknown resource URI: {uri}")

    file_path: Path = resource.get("path")
    if file_path and file_path.exists():
         return [ReadResourceContents(content=file_path.read_text(encoding="utf-8"), mime_type=resource["mimeType"])]

    raise ValueError("Resource content not found")

# --- Tool Handling ---

async def load_tools_from_servers() -> list[dict]:
    import importlib
    tools = []
    # Order matters for overriding
    servers = [
        ("AgentQMS.mcp_server", "agentqms"),
        ("project_compass.mcp_server", "compass"),
        ("experiment_manager.mcp_server", "experiments"),
        ("agent_debug_toolkit.mcp_server", "adt"),
    ]

    for module_name, _prefix in servers:
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "list_tools"):
                server_tools = await mod.list_tools()
                for tool in server_tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                        "implementation": {"module": module_name}
                    })
        except ImportError:
            pass
    return tools

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
        for t in TOOLS_DEFINITIONS
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    start_time = time.time()
    # Approx input tokens
    arg_str = json.dumps(arguments) if arguments else ""
    event = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": name,
        "args_hash": hashlib.md5(str(arguments).encode()).hexdigest()[:8],
        "session_id": SESSION_ID,
        "input_tokens": len(arg_str) // 4,
    }
    try:
        # 1. Validation Logic
        TELEMETRY_PIPELINE.validate(name, arguments)

        # 2. Execution Logic
        tool_def = next((t for t in TOOLS_DEFINITIONS if t["name"] == name), None)
        if not tool_def:
            raise ValueError(f"Unknown tool: {name}")

        module_name = tool_def["implementation"]["module"]
        import importlib
        module = importlib.import_module(module_name)
        result = await module.call_tool(name, arguments)

        # Calculate output tokens
        res_str = ""
        for content in result:
            if hasattr(content, "text") and content.text:
                res_str += content.text
            elif hasattr(content, "data"):
                # Binary data - rough estimate base64
                res_str += str(content.data)
        event["output_tokens"] = len(res_str) // 4

        # 3. Post-Execution & Proactive Feedback (Fix #3)
        duration_ms = (time.time() - start_time) * 1000
        feedback_msg = feedback_interceptor.after_execution(name, duration_ms)

        # Log success
        event["status"] = "success"
        event["duration_ms"] = round(duration_ms, 2)
        event["module"] = module_name

        # FEASIBILITY UPDATE: Capture safe metadata for analytics (Bundle, Type, Query)
        # We only whitelist specific keys to avoid PII/bloat.
        safe_keys = ["bundle_name", "context_bundle", "artifact_type", "type", "category", "query", "action"]
        metadata = {}
        if isinstance(arguments, dict):
            for k in safe_keys:
                if k in arguments and isinstance(arguments[k], (str, int, bool)):
                    metadata[k] = arguments[k]

        if metadata:
            event["metadata"] = metadata

        log_telemetry_event(event)

        if feedback_msg:
            # Append feedback to result
            result.append(TextContent(type="text", text=f"\n\n{feedback_msg}"))

        # 4. Context Auto-Suggestion (Fix #5)
        if "context" not in name and "bundle" not in name:
            try:
                from AgentQMS.tools.core.context_bundle import auto_suggest_context
                # Use tool name + args as task desc
                task_desc = f"{name}: {str(arguments)[:200]}"
                suggestions = auto_suggest_context(task_desc)

                # If a specific, non-generic bundle is suggested
                bundle_name = suggestions.get("context_bundle")
                if bundle_name and bundle_name not in ("general", "development"):
                    sug_text = (
                        f"\n💡 **Context Suggestion**: The '{bundle_name}' bundle may be relevant.\n"
                        f"   Files: {len(suggestions.get('bundle_files', []))}\n"
                        f"   Access: read_resource('bundle://{bundle_name}')"
                    )
                    result.append(TextContent(type="text", text=sug_text))
            except Exception:
                pass

        return result

    except PolicyViolation as e:
        duration_ms = (time.time() - start_time) * 1000
        event["status"] = "policy_violation"
        event["duration_ms"] = round(duration_ms, 2)
        event["policy"] = e.message
        log_telemetry_event(event)
        return [TextContent(type="text", text=f"⚠️ FEEDBACK TRIGGERED: {e.feedback_to_ai}")]
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        event["status"] = "error"
        event["duration_ms"] = round(duration_ms, 2)
        event["error"] = str(e)[:200]
        log_telemetry_event(event)
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]

# --- Server Start Logic ---

async def main():
    global TOOLS_DEFINITIONS, RESOURCES_CONFIG

    parser = argparse.ArgumentParser(description="Unified MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"], help="Transport mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE")
    parser.add_argument("--host", default="0.0.0.0", help="Host for SSE")
    args = parser.parse_args()

    print(f"Loading resources/tools... (Transport: {args.transport})", file=sys.stderr)
    TOOLS_DEFINITIONS = await load_tools_from_servers()
    RESOURCES_CONFIG = await load_resources_from_servers()

    if args.transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.middleware import Middleware
        from starlette.middleware.cors import CORSMiddleware
        import uvicorn

        sse = SseServerTransport("/messages")

        async def handle_sse(request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await app.run(streams[0], streams[1], app.create_initialization_options())

        async def handle_messages(request):
            await sse.handle_post_message(request.scope, request.receive, request._send)

        starlette_app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Route("/messages", endpoint=handle_messages, methods=["POST"])
            ],
            middleware=[
                Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
            ]
        )

        # Disable uvicorn logs to clear stdio if needed, but for background it's fine
        config = uvicorn.Config(starlette_app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    else:
        # Standard Stdio
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
