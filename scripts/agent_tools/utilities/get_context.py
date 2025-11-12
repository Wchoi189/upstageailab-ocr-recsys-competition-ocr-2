from __future__ import annotations

"""Lookup utility for documentation context bundles.

Usage examples:
    # Old style (handbook index bundles)
    uv run python scripts/agent_tools/get_context.py --bundle streamlit-maintenance
    uv run python scripts/agent_tools/get_context.py --list-bundles

    # New style (YAML context bundles)
    uv run python scripts/agent_tools/get_context.py --task "implement new feature"
    uv run python scripts/agent_tools/get_context.py --type development
    uv run python scripts/agent_tools/get_context.py --list-context-bundles
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_bootstrap():
    if "scripts._bootstrap" in sys.modules:
        return sys.modules["scripts._bootstrap"]

    current_dir = Path(__file__).resolve().parent
    for directory in (current_dir, *tuple(current_dir.parents)):
        candidate = directory / "_bootstrap.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(
                "scripts._bootstrap", candidate
            )
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


_BOOTSTRAP = _load_bootstrap()
setup_project_paths = _BOOTSTRAP.setup_project_paths
get_path_resolver = _BOOTSTRAP.get_path_resolver

setup_project_paths()
_RESOLVER = get_path_resolver()

try:
    from scripts.agent_tools.core.context_bundle import (
        get_context_bundle,
        list_available_bundles,
        print_context_bundle,
    )
except ImportError:
    # Graceful fallback if context_bundle module not available
    get_context_bundle = None
    list_available_bundles = None
    print_context_bundle = None

DOC_INDEX_PATH = _RESOLVER.config.docs_dir / "ai_handbook" / "index.json"


def load_index() -> dict[str, Any]:
    if not DOC_INDEX_PATH.exists():
        msg = f"Handbook index not found: {DOC_INDEX_PATH}"
        raise SystemExit(msg)

    with DOC_INDEX_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        if not isinstance(data, dict):
            raise SystemExit("Handbook index must be a JSON object")
        return data


def print_bundle(index: dict[str, Any], bundle_name: str) -> None:
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
    entries: list[dict[str, Any]] = index.get("entries", [])
    for entry in entries:
        if entry.get("id") == entry_id:
            print(json.dumps(entry, indent=2))
            return
    raise SystemExit(f"Entry '{entry_id}' not found.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print documentation bundles or entries for agents."
    )

    # Legacy arguments (handbook index bundles)
    legacy_group = parser.add_argument_group("Legacy (Handbook Index)")
    legacy_group.add_argument("--bundle", help="Handbook bundle identifier to print.")
    legacy_group.add_argument("--entry", help="Entry identifier to print details for.")
    legacy_group.add_argument(
        "--list-bundles",
        action="store_true",
        help="List available handbook bundles.",
    )

    # New arguments (YAML context bundles)
    context_group = parser.add_argument_group("Context Bundles (YAML)")
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

    args = parser.parse_args(argv)

    # Validate that at least one mutually exclusive group is provided
    legacy_args = [args.bundle, args.entry, args.list_bundles]
    context_args = [args.task, args.type, args.list_context_bundles]

    if not any(legacy_args) and not any(context_args):
        parser.error(
            "At least one argument required. Use --help to see available options."
        )

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Handle new context bundle commands
    if args.list_context_bundles:
        if list_available_bundles is None:
            print(
                "ERROR: Context bundle module not available. "
                "Ensure scripts.agent_tools.core.context_bundle can be imported.",
                file=sys.stderr,
            )
            sys.exit(1)
        bundles = list_available_bundles()
        print("Available context bundles:")
        for bundle in bundles:
            print(f"  - {bundle}")
        return

    if args.task or args.type:
        if print_context_bundle is None:
            print(
                "ERROR: Context bundle module not available. "
                "Ensure scripts.agent_tools.core.context_bundle can be imported.",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.task:
            print_context_bundle(args.task, args.type)
        elif args.type:
            # Use a generic description if only type is provided
            print_context_bundle(f"task type: {args.type}", args.type)
        return

    # Handle legacy handbook index commands
    index = load_index()

    if args.list_bundles:
        list_bundles(index)
    elif args.entry:
        lookup_entry(index, args.entry)
    elif args.bundle:
        print_bundle(index, args.bundle)


if __name__ == "__main__":
    main()
