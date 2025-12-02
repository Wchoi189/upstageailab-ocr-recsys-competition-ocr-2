from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from AgentQMS.agent_tools.utils.paths import get_docs_dir
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

"""Lookup utility for documentation context bundles.

This is the canonical implementation in agent_tools.

Usage examples:
    # YAML context bundles (recommended)
    PYTHONPATH=. python AgentQMS/agent_tools/utilities/get_context.py --task "implement new feature"
    PYTHONPATH=. python AgentQMS/agent_tools/utilities/get_context.py --type development
    PYTHONPATH=. python AgentQMS/agent_tools/utilities/get_context.py --list-context-bundles

    # Legacy handbook index bundles (deprecated)
    PYTHONPATH=. python AgentQMS/agent_tools/utilities/get_context.py --bundle bundle-name
    PYTHONPATH=. python AgentQMS/agent_tools/utilities/get_context.py --list-bundles
"""

ensure_project_root_on_sys_path()

try:
    from AgentQMS.agent_tools.core.context_bundle import (
        get_context_bundle,
        list_available_bundles,
        print_context_bundle,
    )
    CONTEXT_BUNDLES_AVAILABLE = True
except ImportError:
    # Graceful fallback if context_bundle module not available
    get_context_bundle = None  # type: ignore
    list_available_bundles = None  # type: ignore
    print_context_bundle = None  # type: ignore
    CONTEXT_BUNDLES_AVAILABLE = False


def _get_doc_index_path() -> Path:
    """Get the documentation index path, handling deprecated docs location."""
    docs_dir = get_docs_dir()
    # Try new location first
    new_path = docs_dir / "ai_handbook" / "index.json"
    if new_path.exists():
        return new_path
    # Fall back to deprecated location
    from AgentQMS.agent_tools.utils.paths import get_project_root
    deprecated_path = get_project_root() / "docs_deprecated" / "ai_handbook" / "index.json"
    if deprecated_path.exists():
        return deprecated_path
    return new_path  # Return expected path for error messages


def load_index() -> dict[str, Any]:
    """Load the handbook index JSON file."""
    index_path = _get_doc_index_path()
    if not index_path.exists():
        msg = f"Handbook index not found: {index_path}"
        raise SystemExit(msg)

    with index_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        if not isinstance(data, dict):
            raise SystemExit("Handbook index must be a JSON object")
        return data


def print_bundle(index: dict[str, Any], bundle_name: str) -> None:
    """Print entries for a specific handbook bundle."""
    bundles: dict[str, Any] = index.get("bundles", {})
    entries: list[dict[str, Any]] = index.get("entries", [])

    if bundle_name not in bundles:
        available = ", ".join(sorted(bundles)) or "<none>"
        raise SystemExit(f"Unknown bundle '{bundle_name}'. Available: {available}")

    bundle = bundles[bundle_name]
    print(f"Bundle: {bundle.get('title', bundle_name)}")
    if description := bundle.get("description"):
        print(f"Description: {description}")
    print()

    entry_map = {entry.get("id"): entry for entry in entries}

    for entry_id in bundle.get("entries", []):
        entry = entry_map.get(entry_id)
        if not entry:
            print(f"- {entry_id} (missing from entries)")
            continue

        title = entry.get("title", entry_id)
        path = entry.get("path")
        priority = entry.get("priority", "unknown")
        summary = entry.get("summary", "")
        print(f"- {title}")
        if path:
            print(f"    path: {path}")
        print(f"    priority: {priority}")
        if summary:
            print(f"    summary: {summary}")
        tags = entry.get("tags") or []
        if tags:
            print(f"    tags: {', '.join(tags)}")
        print()


def list_bundles(index: dict[str, Any]) -> None:
    """List all available handbook bundles."""
    bundles: dict[str, Any] = index.get("bundles", {})
    if not bundles:
        print("No bundles defined.")
        return

    for name, meta in sorted(bundles.items()):
        title = meta.get("title", name)
        description = meta.get("description", "")
        print(f"- {name}: {title}")
        if description:
            print(f"    {description}")


def lookup_entry(index: dict[str, Any], entry_id: str) -> None:
    """Look up and print a specific entry by ID."""
    entries: list[dict[str, Any]] = index.get("entries", [])
    for entry in entries:
        if entry.get("id") == entry_id:
            print(json.dumps(entry, indent=2))
            return
    raise SystemExit(f"Entry '{entry_id}' not found.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Print documentation bundles or entries for agents."
    )

    # Context bundle arguments (recommended)
    context_group = parser.add_argument_group("Context Bundles (YAML) - Recommended")
    context_group.add_argument(
        "--task",
        type=str,
        help="Task description - will auto-detect bundle type and generate context.",
    )
    context_group.add_argument(
        "--type",
        type=str,
        choices=["development", "documentation", "debugging", "planning", "general"],
        help="Explicit bundle type (development, documentation, debugging, planning, general).",
    )
    context_group.add_argument(
        "--list-context-bundles",
        action="store_true",
        help="List available context bundles (YAML-based).",
    )

    # Legacy arguments (handbook index bundles)
    legacy_group = parser.add_argument_group("Legacy (Handbook Index) - Deprecated")
    legacy_group.add_argument("--bundle", help="Handbook bundle identifier to print.")
    legacy_group.add_argument("--entry", help="Entry identifier to print details for.")
    legacy_group.add_argument(
        "--list-bundles",
        action="store_true",
        help="List available handbook bundles.",
    )

    args = parser.parse_args(argv)

    # Validate that at least one mutually exclusive group is provided
    legacy_args = [args.bundle, args.entry, args.list_bundles]
    context_args = [args.task, args.type, args.list_context_bundles]

    if not any(legacy_args) and not any(context_args):
        parser.error(
            "At least one argument required. Use --help to see available options."
        )

    return args


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Handle context bundle commands (recommended)
    if args.list_context_bundles:
        if not CONTEXT_BUNDLES_AVAILABLE or list_available_bundles is None:
            print(
                "ERROR: Context bundle module not available.",
                file=sys.stderr,
            )
            return 1
        bundles = list_available_bundles()
        if bundles:
            print("Available context bundles:")
            for bundle in bundles:
                print(f"  - {bundle}")
        else:
            print("No context bundles found.")
            print("Create YAML bundle definitions in AgentQMS/knowledge/context_bundles/")
        return 0

    if args.task or args.type:
        if not CONTEXT_BUNDLES_AVAILABLE or print_context_bundle is None:
            print(
                "ERROR: Context bundle module not available.",
                file=sys.stderr,
            )
            return 1

        try:
            if args.task:
                print_context_bundle(args.task, args.type)
            elif args.type:
                # Use a generic description if only type is provided
                print_context_bundle(f"task type: {args.type}", args.type)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        return 0

    # Handle legacy handbook index commands
    try:
        index = load_index()
    except SystemExit as e:
        print(f"WARNING: {e}", file=sys.stderr)
        print("Legacy handbook index is deprecated. Use --task or --type instead.")
        return 1

    if args.list_bundles:
        list_bundles(index)
    elif args.entry:
        lookup_entry(index, args.entry)
    elif args.bundle:
        print_bundle(index, args.bundle)

    return 0


if __name__ == "__main__":
    sys.exit(main())

