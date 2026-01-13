#!/usr/bin/env python3
"""
Agent Debug Toolkit MCP Server

Provides AST-based analysis tools for AI agents to debug complex
Hydra/OmegaConf configuration issues.

"""
import sys
import asyncio
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import from AgentQMS (installed as package in pyproject.toml)
from AgentQMS.tools.utils.paths import get_project_root as _get_project_root

try:
    from AgentQMS.tools.utils.config_loader import ConfigLoader
    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False

# Create MCP server
app = Server("agent_debug_toolkit")

# Get project root using the path utility (this is the canonical project root)
PROJECT_ROOT = _get_project_root()

def load_tools_config() -> list[dict]:
    """Load tools configuration from YAML file."""
    if not HAS_CONFIG_LOADER:
        # Fallback if AgentQMS not available (should be rare given usage)
        print("Warning: ConfigLoader not found. Using empty tools list.", file=sys.stderr)
        return []

    config_loader = ConfigLoader(cache_size=5)
    config_path = Path(__file__).parent / "config/tools.yaml"

    tools = config_loader.get_config(config_path, defaults=[])

    return tools if isinstance(tools, list) else []


# Load tools from configuration
TOOLS = load_tools_config()


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    # Return meta-tools first for prominence
    return [
        Tool(name=tool["name"], description=tool["description"], inputSchema=tool["inputSchema"])
        for tool in TOOLS
    ]


def resolve_path(path_str: str) -> Path:
    """Resolve a path string to an absolute Path."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_analyzer(analyzer_class, path: Path, recursive: bool = True):
    """Run an analyzer and return the report."""
    analyzer = analyzer_class()
    if path.is_file():
        return analyzer.analyze_file(path)
    elif path.is_dir():
        return analyzer.analyze_directory(path, recursive=recursive)
    else:
        raise FileNotFoundError(f"Path not found: {path}")


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute a tool."""

    # --- Meta-Tool Handlers (Router Pattern) ---

    if name == "adt_meta_query":
        from agent_debug_toolkit.router import route_query

        kind = arguments.get("kind", "")
        target = arguments.get("target", "")
        options = arguments.get("options", {})

        try:
            routing = route_query(kind, target, options)
            # Recursively call the routed tool
            return await call_tool(routing["tool_name"], routing["arguments"])
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "adt_meta_edit":
        from agent_debug_toolkit.router import route_edit

        kind = arguments.get("kind", "")
        target = arguments.get("target", "")
        options = arguments.get("options", {})

        try:
            routing = route_edit(kind, target, options)
            # Recursively call the routed tool
            return await call_tool(routing["tool_name"], routing["arguments"])
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    # --- Individual Tool Handlers (backward compatibility) ---

    if name == "analyze_config_access":
        from agent_debug_toolkit.analyzers.config_access import ConfigAccessAnalyzer

        path = resolve_path(arguments.get("path", ""))
        component = arguments.get("component")
        output_format = arguments.get("output", "json")

        report = run_analyzer(ConfigAccessAnalyzer, path)

        # Filter by component if specified
        if component:
            report.results = report.filter_by_component(component)
            report.summary["filtered_by"] = component

        content = report.to_json() if output_format == "json" else report.to_markdown()

        return [TextContent(type="text", text=content)]

    elif name == "trace_merge_order":
        from agent_debug_toolkit.analyzers.merge_order import MergeOrderTracker

        path = resolve_path(arguments.get("file", ""))
        explain = arguments.get("explain", True)
        output_format = arguments.get("output", "markdown")

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        analyzer = MergeOrderTracker()
        report = analyzer.analyze_file(path)

        if output_format == "markdown" and explain:
            explanation = analyzer.explain_precedence()
            markdown = f"{explanation}\n\n---\n\n{report.to_markdown()}"
            content = markdown
        elif output_format == "json":
            content = report.to_json()
        else:
            content = report.to_markdown()

        return [TextContent(type="text", text=content)]

    elif name == "find_hydra_usage":
        from agent_debug_toolkit.analyzers.hydra_usage import HydraUsageAnalyzer

        path = resolve_path(arguments.get("path", ""))
        output_format = arguments.get("output", "json")

        report = run_analyzer(HydraUsageAnalyzer, path)
        content = report.to_json() if output_format == "json" else report.to_markdown()

        return [TextContent(type="text", text=content)]

    elif name == "find_component_instantiations":
        from agent_debug_toolkit.analyzers.instantiation import ComponentInstantiationTracker

        path = resolve_path(arguments.get("path", ""))
        component = arguments.get("component")
        output_format = arguments.get("output", "json")

        report = run_analyzer(ComponentInstantiationTracker, path)

        # Filter by component if specified
        if component:
            report.results = [
                r
                for r in report.results
                if component.lower() in (r.metadata.get("component_type") or "").lower()
                or component.lower() in r.pattern.lower()
            ]
            report.summary["filtered_by"] = component

        content = report.to_json() if output_format == "json" else report.to_markdown()

        return [TextContent(type="text", text=content)]

    elif name == "explain_config_flow":
        from agent_debug_toolkit.analyzers.config_access import ConfigAccessAnalyzer
        from agent_debug_toolkit.analyzers.merge_order import MergeOrderTracker
        from agent_debug_toolkit.analyzers.hydra_usage import HydraUsageAnalyzer
        from agent_debug_toolkit.analyzers.instantiation import ComponentInstantiationTracker

        path = resolve_path(arguments.get("file", ""))

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        # Run all analyzers
        config_report = ConfigAccessAnalyzer().analyze_file(path)
        merge_tracker = MergeOrderTracker()
        merge_report = merge_tracker.analyze_file(path)
        hydra_report = HydraUsageAnalyzer().analyze_file(path)
        inst_report = ComponentInstantiationTracker().analyze_file(path)

        # Build summary
        summary_parts = [
            f"# Configuration Flow Analysis: {path.name}",
            "",
            "## Overview",
            f"- **File**: `{path}`",
            f"- **Config Accesses**: {len(config_report.results)}",
            f"- **Merge Operations**: {len(merge_report.results)}",
            f"- **Hydra Patterns**: {len(hydra_report.results)}",
            f"- **Component Instantiations**: {len(inst_report.results)}",
            "",
        ]

        # Merge precedence explanation
        if merge_report.results:
            summary_parts.extend(
                [
                    "## Merge Precedence",
                    merge_tracker.explain_precedence(),
                    "",
                ]
            )

        # Entry points
        entry_points = [r for r in hydra_report.results if r.category == "entry_point"]
        if entry_points:
            summary_parts.append("## Entry Points")
            for ep in entry_points:
                summary_parts.append(f"- Line {ep.line}: `{ep.pattern}`")
            summary_parts.append("")

        # Component breakdown
        comp_types = inst_report.summary.get("by_component_type", {})
        if comp_types:
            summary_parts.append("## Component Breakdown")
            for comp_type, count in comp_types.items():
                summary_parts.append(f"- **{comp_type}**: {count} instantiation(s)")
            summary_parts.append("")

        # Config access by category
        access_cats = config_report.summary.get("by_category", {})
        if access_cats:
            summary_parts.append("## Config Access Categories")
            for cat, count in access_cats.items():
                summary_parts.append(f"- {cat.replace('_', ' ').title()}: {count}")
            summary_parts.append("")

        content = "\n".join(summary_parts)

        return [TextContent(type="text", text=content)]

    elif name == "analyze_dependency_graph":
        from agent_debug_toolkit.analyzers.dependency_graph import DependencyGraphAnalyzer

        path = resolve_path(arguments.get("path", ""))
        include_stdlib = arguments.get("include_stdlib", False)
        output_format = arguments.get("output", "json")

        analyzer = DependencyGraphAnalyzer(include_stdlib=include_stdlib)
        if path.is_file():
            report = analyzer.analyze_file(path)
        elif path.is_dir():
            report = analyzer.analyze_directory(path)
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        if output_format == "mermaid":
            content = analyzer.to_mermaid()
        elif output_format == "json":
            content = report.to_json()
        else:
            content = report.to_markdown()

        return [TextContent(type="text", text=content)]

    elif name == "analyze_imports":
        from agent_debug_toolkit.analyzers.import_tracker import ImportTracker

        path = resolve_path(arguments.get("path", ""))
        show_unused = arguments.get("show_unused", True)
        output_format = arguments.get("output", "json")

        report = run_analyzer(ImportTracker, path)

        if not show_unused and "unused_imports" in report.summary:
            del report.summary["unused_imports"]

        content = report.to_json() if output_format == "json" else report.to_markdown()

        return [TextContent(type="text", text=content)]

    elif name == "analyze_complexity":
        from agent_debug_toolkit.analyzers.complexity_metrics import ComplexityMetricsAnalyzer

        path = resolve_path(arguments.get("path", ""))
        threshold = arguments.get("threshold", 10)
        output_format = arguments.get("output", "json")

        analyzer = ComplexityMetricsAnalyzer(complexity_threshold=threshold)
        if path.is_file():
            report = analyzer.analyze_file(path)
        elif path.is_dir():
            report = analyzer.analyze_directory(path)
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        content = report.to_json() if output_format == "json" else report.to_markdown()

        return [TextContent(type="text", text=content)]

    elif name == "context_tree":
        from agent_debug_toolkit.analyzers.context_tree import ContextTreeAnalyzer, format_tree_markdown

        path = resolve_path(arguments.get("path", ""))
        depth = arguments.get("depth", 3)
        output_format = arguments.get("output", "markdown")

        if not path.is_dir():
            raise ValueError(f"Path must be a directory: {path}")

        analyzer = ContextTreeAnalyzer(max_depth=depth)
        report = analyzer.analyze_directory(str(path))

        if "error" in report.summary:
            raise RuntimeError(f"Analysis error: {report.summary['error']}")

        if output_format == "markdown":
            content = format_tree_markdown(report)
        else:
            content = report.to_json()

        return [TextContent(type="text", text=content)]

    elif name == "intelligent_search":
        from agent_debug_toolkit.analyzers.intelligent_search import (
            IntelligentSearcher,
            format_search_results_markdown,
        )

        query = arguments.get("query", "")
        root = resolve_path(arguments.get("root", PROJECT_ROOT))
        fuzzy = arguments.get("fuzzy", True)
        threshold = arguments.get("threshold", 0.6)
        output_format = arguments.get("output", "markdown")

        searcher = IntelligentSearcher(str(root), str(PROJECT_ROOT))
        results = searcher.search(query, fuzzy=fuzzy, threshold=threshold)

        if output_format == "json":
            import json
            content = json.dumps([r.to_dict() for r in results], indent=2)
        else:
            content = format_search_results_markdown(results, query)

        return [TextContent(type="text", text=content)]

    # --- Edit Tools ---

    elif name == "apply_unified_diff":
        from agent_debug_toolkit.edits import apply_unified_diff

        diff = arguments.get("diff", "")
        strategy = arguments.get("strategy", "fuzzy")
        dry_run = arguments.get("dry_run", False)

        report = apply_unified_diff(
            diff=diff,
            strategy=strategy,
            project_root=PROJECT_ROOT,
            dry_run=dry_run,
        )

        return [TextContent(type="text", text=report.to_json())]

    elif name == "smart_edit":
        from agent_debug_toolkit.edits import smart_edit

        path = resolve_path(arguments.get("file", ""))
        search = arguments.get("search", "")
        replace = arguments.get("replace", "")
        mode = arguments.get("mode", "exact")
        all_occurrences = arguments.get("all_occurrences", False)
        dry_run = arguments.get("dry_run", False)

        report = smart_edit(
            file=path,
            search=search,
            replace=replace,
            mode=mode,
            all_occurrences=all_occurrences,
            dry_run=dry_run,
        )

        return [TextContent(type="text", text=report.to_json())]

    elif name == "read_file_slice":
        from agent_debug_toolkit.edits import read_file_slice

        path = resolve_path(arguments.get("file", ""))
        start_line = int(arguments.get("start_line", 1))
        end_line = int(arguments.get("end_line", start_line + 50))
        context_lines = int(arguments.get("context_lines", 0))

        content = read_file_slice(path, start_line, end_line, context_lines)

        return [TextContent(type="text", text=content)]

    elif name == "format_code":
        from agent_debug_toolkit.edits import format_code

        path = resolve_path(arguments.get("path", ""))
        style = arguments.get("style", "black")
        check_only = arguments.get("check_only", False)

        report = format_code(path, style=style, check_only=check_only)

        return [TextContent(type="text", text=report.to_json())]

    raise ValueError(f"Unknown tool: {name}")


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
