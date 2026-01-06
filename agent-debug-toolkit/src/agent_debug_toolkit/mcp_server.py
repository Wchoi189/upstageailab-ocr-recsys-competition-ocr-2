#!/usr/bin/env python3
"""
Agent Debug Toolkit MCP Server

Provides AST-based analysis tools for AI agents to debug complex
Hydra/OmegaConf configuration issues.

Tools exposed:
- analyze_config_access: Find cfg.X, config.X patterns in Python code
- trace_merge_order: Trace OmegaConf.merge() call precedence
- find_hydra_usage: Find Hydra framework patterns (@hydra.main, instantiate)
- find_component_instantiations: Track component factory patterns
- explain_config_flow: Generate high-level config flow summary
- analyze_dependency_graph: Build module dependency graphs (Phase 1)
- analyze_imports: Categorize and track imports (Phase 1)
- analyze_complexity: Calculate code complexity metrics (Phase 1)
"""

import asyncio
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Create MCP server
app = Server("agent_debug_toolkit")


def get_project_root() -> Path:
    """Get project root from environment or current working directory."""
    import os

    # Check for PROJECT_ROOT environment variable
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root)

    # Default to current working directory
    return Path.cwd()


# Define available tools
TOOLS = [
    {
        "name": "analyze_config_access",
        "description": "Analyze Python code for configuration access patterns (cfg.X, self.cfg.X, config['key']). Essential for understanding how configuration flows through the codebase.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Python file or directory to analyze (relative to project root or absolute)",
                },
                "component": {
                    "type": "string",
                    "description": "Optional: Filter results by component name (e.g., 'decoder', 'encoder')",
                },
                "output": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "json",
                    "description": "Output format",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "trace_merge_order",
        "description": "Trace OmegaConf.merge() operations and their precedence order. Critical for debugging configuration override issues where later merges overwrite earlier values.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to Python file to analyze"},
                "explain": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include detailed precedence explanation",
                },
                "output": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "markdown",
                    "description": "Output format",
                },
            },
            "required": ["file"],
        },
    },
    {
        "name": "find_hydra_usage",
        "description": "Find Hydra framework usage patterns including @hydra.main decorators, hydra.utils.instantiate() calls, and _target_ config patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Python file or directory to analyze",
                },
                "output": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "json",
                    "description": "Output format",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "find_component_instantiations",
        "description": "Track component instantiation patterns: get_*_by_cfg() factory calls, registry.create() patterns, and direct class instantiation. Helps trace where components are created and what config sources them.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Python file or directory to analyze",
                },
                "component": {
                    "type": "string",
                    "description": "Optional: Filter by component type (e.g., 'decoder', 'encoder', 'head')",
                },
                "output": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "json",
                    "description": "Output format",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "explain_config_flow",
        "description": "Generate a high-level summary of configuration flow through a file: imports, entry points, merge operations, and instantiations. Provides a bird's-eye view for understanding complex config handling.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to Python file to analyze"}
            },
            "required": ["file"],
        },
    },
    {
        "name": "analyze_dependency_graph",
        "description": "Build module dependency graph showing imports, class inheritance, and call relationships. Detects circular dependencies. Outputs JSON, markdown, or Mermaid diagram.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Python file or directory to analyze",
                },
                "include_stdlib": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include standard library imports in graph",
                },
                "output": {
                    "type": "string",
                    "enum": ["json", "markdown", "mermaid"],
                    "default": "json",
                    "description": "Output format",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "analyze_imports",
        "description": "Categorize imports as stdlib, third-party, or local. Detects potentially unused imports. Useful for dependency auditing and dead code detection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Python file or directory to analyze",
                },
                "show_unused": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include potentially unused imports in output",
                },
                "output": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "json",
                    "description": "Output format",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "analyze_complexity",
        "description": "Calculate code complexity metrics: cyclomatic complexity, nesting depth, LOC, parameter count. Identifies functions exceeding thresholds for refactoring candidates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Python file or directory to analyze",
                },
                "threshold": {
                    "type": "integer",
                    "default": 10,
                    "description": "Complexity threshold for warnings",
                },
                "output": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "json",
                    "description": "Output format",
                },
            },
            "required": ["path"],
        },
    },
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(name=tool["name"], description=tool["description"], inputSchema=tool["inputSchema"])
        for tool in TOOLS
    ]


def resolve_path(path_str: str) -> Path:
    """Resolve a path string to an absolute Path."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_project_root() / path


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
