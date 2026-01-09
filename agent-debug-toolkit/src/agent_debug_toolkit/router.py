"""
Agent Debug Toolkit - Router Module

Implements the Router/Meta-Tool Pattern to reduce tool proliferation.
Instead of exposing 14+ individual tools, we expose 2 meta-tools:
- adt_meta_query: Routes to analysis tools
- adt_meta_edit: Routes to edit tools

This reduces model context burden while maintaining full functionality.
"""

from __future__ import annotations

from typing import Any


# Define available query kinds and their mappings
QUERY_KINDS = {
    "config_access": "analyze_config_access",
    "merge_order": "trace_merge_order",
    "hydra_usage": "find_hydra_usage",
    "component_instantiations": "find_component_instantiations",
    "config_flow": "explain_config_flow",
    "dependency_graph": "analyze_dependency_graph",
    "imports": "analyze_imports",
    "complexity": "analyze_complexity",
    "context_tree": "context_tree",
    "symbol_search": "intelligent_search",
}

# Define available edit kinds and their mappings
EDIT_KINDS = {
    "apply_diff": "apply_unified_diff",
    "smart_edit": "smart_edit",
    "read_slice": "read_file_slice",
    "format": "format_code",
}


def get_query_kinds() -> list[str]:
    """Get list of available query kinds."""
    return list(QUERY_KINDS.keys())


def get_edit_kinds() -> list[str]:
    """Get list of available edit kinds."""
    return list(EDIT_KINDS.keys())


def route_query(kind: str, target: str, options: dict[str, Any]) -> dict[str, Any]:
    """
    Route a query kind to its corresponding tool arguments.

    Args:
        kind: Query kind (e.g., "config_access", "symbol_search")
        target: Target path or query string
        options: Additional options to pass through

    Returns:
        Dict with tool_name and arguments
    """
    if kind not in QUERY_KINDS:
        raise ValueError(f"Unknown query kind: {kind}. Valid kinds: {list(QUERY_KINDS.keys())}")

    tool_name = QUERY_KINDS[kind]
    arguments: dict[str, Any] = {}

    # Map common parameters
    if kind in ("config_access", "hydra_usage", "component_instantiations", "dependency_graph", "imports", "complexity"):
        arguments["path"] = target
    elif kind in ("merge_order", "config_flow"):
        arguments["file"] = target
    elif kind == "context_tree":
        arguments["path"] = target
        if "depth" in options:
            arguments["depth"] = options["depth"]
    elif kind == "symbol_search":
        arguments["query"] = target
        if "root" in options:
            arguments["root"] = options["root"]
        if "fuzzy" in options:
            arguments["fuzzy"] = options["fuzzy"]
        if "threshold" in options:
            arguments["threshold"] = options["threshold"]

    # Pass through common options
    if "output" in options:
        arguments["output"] = options["output"]
    if "component" in options:
        arguments["component"] = options["component"]
    if "include_stdlib" in options:
        arguments["include_stdlib"] = options["include_stdlib"]
    if "threshold" in options and kind == "complexity":
        arguments["threshold"] = options["threshold"]

    return {"tool_name": tool_name, "arguments": arguments}


def route_edit(kind: str, target: str, options: dict[str, Any]) -> dict[str, Any]:
    """
    Route an edit kind to its corresponding tool arguments.

    Args:
        kind: Edit kind (e.g., "apply_diff", "smart_edit")
        target: Target file or diff content
        options: Additional options (diff, search, replace, etc.)

    Returns:
        Dict with tool_name and arguments
    """
    if kind not in EDIT_KINDS:
        raise ValueError(f"Unknown edit kind: {kind}. Valid kinds: {list(EDIT_KINDS.keys())}")

    tool_name = EDIT_KINDS[kind]
    arguments: dict[str, Any] = {}

    if kind == "apply_diff":
        # For diffs, target can be the diff content or we get it from options
        arguments["diff"] = options.get("diff", target)
        if "strategy" in options:
            arguments["strategy"] = options["strategy"]
        if "dry_run" in options:
            arguments["dry_run"] = options["dry_run"]

    elif kind == "smart_edit":
        arguments["file"] = target
        arguments["search"] = options.get("search", "")
        arguments["replace"] = options.get("replace", "")
        if "mode" in options:
            arguments["mode"] = options["mode"]
        if "all_occurrences" in options:
            arguments["all_occurrences"] = options["all_occurrences"]
        if "dry_run" in options:
            arguments["dry_run"] = options["dry_run"]

    elif kind == "read_slice":
        arguments["file"] = target
        arguments["start_line"] = options.get("start_line", 1)
        arguments["end_line"] = options.get("end_line", 50)
        if "context_lines" in options:
            arguments["context_lines"] = options["context_lines"]

    elif kind == "format":
        arguments["path"] = target
        if "style" in options:
            arguments["style"] = options["style"]
        if "check_only" in options:
            arguments["check_only"] = options["check_only"]

    return {"tool_name": tool_name, "arguments": arguments}


# Meta-tool schema definitions for MCP
META_QUERY_SCHEMA = {
    "name": "adt_meta_query",
    "description": "Unified analysis tool for code understanding. Routes to specialized analyzers based on 'kind' parameter. Use this instead of individual analysis tools to reduce context overhead.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(QUERY_KINDS.keys()),
                "description": "Type of analysis: config_access (cfg.X patterns), merge_order (OmegaConf.merge precedence), hydra_usage (@hydra.main, instantiate), component_instantiations (factory patterns), config_flow (high-level summary), dependency_graph (imports/calls), imports (categorized imports), complexity (cyclomatic/nesting), context_tree (directory structure), symbol_search (fuzzy symbol lookup)",
            },
            "target": {
                "type": "string",
                "description": "Target path (file/directory) or search query depending on kind",
            },
            "options": {
                "type": "object",
                "description": "Kind-specific options: output (json|markdown), component (filter), depth (context_tree), threshold (complexity/search), root (symbol_search), fuzzy (symbol_search)",
                "additionalProperties": True,
            },
        },
        "required": ["kind", "target"],
    },
}

META_EDIT_SCHEMA = {
    "name": "adt_meta_edit",
    "description": "Unified edit tool for code modification. Routes to specialized editors based on 'kind' parameter. Use this instead of individual edit tools.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(EDIT_KINDS.keys()),
                "description": "Type of edit: apply_diff (unified diff with fuzzy matching), smart_edit (search/replace with modes), read_slice (targeted line reading), format (code formatting)",
            },
            "target": {
                "type": "string",
                "description": "Target file path (for edits) or diff content (for apply_diff)",
            },
            "options": {
                "type": "object",
                "description": "Kind-specific options: diff (apply_diff), strategy (apply_diff), search/replace/mode (smart_edit), start_line/end_line (read_slice), style (format), dry_run (preview only)",
                "additionalProperties": True,
            },
        },
        "required": ["kind", "target"],
    },
}
