"""
Plugin CLI Module

Command-line interface for the plugin system.
Provides commands to list, validate, and inspect plugins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .loader import PluginLoader
from .registry import PluginRegistry
from .snapshot import SnapshotWriter


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the plugin CLI."""
    parser = argparse.ArgumentParser(
        description="AgentQMS Plugin Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m AgentQMS.agent_tools.core.plugins --list
  python -m AgentQMS.agent_tools.core.plugins --validate
  python -m AgentQMS.agent_tools.core.plugins --show change_request
  python -m AgentQMS.agent_tools.core.plugins --json
        """,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all discovered plugins",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all plugins and show errors",
    )
    parser.add_argument(
        "--artifact-types",
        action="store_true",
        help="List artifact type plugins only",
    )
    parser.add_argument(
        "--context-bundles",
        action="store_true",
        help="List context bundle plugins only",
    )
    parser.add_argument(
        "--validators",
        action="store_true",
        help="Show validator configuration",
    )
    parser.add_argument(
        "--show",
        type=str,
        metavar="NAME",
        help="Show details for a specific plugin",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--write-snapshot",
        action="store_true",
        help="Write runtime snapshot to .agentqms/state/plugins.yaml",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        help="Project root directory (default: auto-detect)",
    )

    return parser


def format_human_output(
    registry: PluginRegistry,
    args: argparse.Namespace,
    discovery_paths: Dict[str, str],
) -> str:
    """Format registry data for human-readable output."""
    lines = [
        "=" * 60,
        "AGENTQMS PLUGIN REGISTRY",
        "=" * 60,
    ]

    # Validation results (always show if errors, or if --validate)
    if args.validate or registry.has_errors():
        lines.append("\n📋 Validation Results:")
        if registry.validation_errors:
            for error in registry.validation_errors:
                lines.append(f"   ❌ {error.plugin_path}")
                lines.append(f"      Type: {error.plugin_type}")
                lines.append(f"      Error: {error.error_message}")
        else:
            lines.append("   ✅ All plugins validated successfully")

    # Artifact types
    if args.list or args.artifact_types:
        lines.append("\n📦 Artifact Types:")
        if registry.artifact_types:
            for name, data in registry.artifact_types.items():
                meta = registry.get_metadata_for_plugin(name, "artifact_type")
                source = meta.source if meta else "unknown"
                version = data.get("version", "?")
                lines.append(f"   • {name} (v{version}) [{source}]")
                if data.get("description"):
                    desc = data["description"].split("\n")[0][:60]
                    lines.append(f"     {desc}...")
        else:
            lines.append("   (none)")

    # Context bundles
    if args.list or args.context_bundles:
        lines.append("\n📚 Context Bundles:")
        if registry.context_bundles:
            for name, data in registry.context_bundles.items():
                meta = registry.get_metadata_for_plugin(name, "context_bundle")
                source = meta.source if meta else "unknown"
                lines.append(f"   • {name} [{source}]")
                if data.get("title"):
                    lines.append(f"     {data['title']}")
        else:
            lines.append("   (none)")

    # Validators
    if args.list or args.validators:
        lines.append("\n⚙️  Validators:")
        if registry.validators:
            v = registry.validators
            lines.append(f"   Prefixes: {len(v.get('prefixes', {}))}")
            lines.append(f"   Types: {len(v.get('types', []))}")
            lines.append(f"   Categories: {len(v.get('categories', []))}")
            lines.append(f"   Custom validators: {len(v.get('custom_validators', []))}")
            if args.validators:
                # Show details
                if v.get("prefixes"):
                    lines.append("   Prefix mappings:")
                    for prefix, directory in v["prefixes"].items():
                        lines.append(f"     {prefix} → {directory}")
        else:
            lines.append("   (no extensions)")

    # Show specific plugin
    if args.show:
        lines.append(f"\n🔍 Plugin Details: {args.show}")
        if args.show in registry.artifact_types:
            lines.append(yaml.dump(
                registry.artifact_types[args.show],
                default_flow_style=False
            ))
        elif args.show in registry.context_bundles:
            lines.append(yaml.dump(
                registry.context_bundles[args.show],
                default_flow_style=False
            ))
        else:
            lines.append(f"   Plugin '{args.show}' not found")

    # Footer
    lines.extend([
        "",
        "=" * 60,
        f"Loaded at: {registry.loaded_at}",
        f"Framework: {discovery_paths.get('framework', 'N/A')}",
        f"Project: {discovery_paths.get('project', 'N/A')}",
        "=" * 60,
    ])

    return "\n".join(lines)


def format_json_output(
    registry: PluginRegistry,
    args: argparse.Namespace,
) -> str:
    """Format registry data as JSON."""
    output: Dict[str, Any]

    if args.artifact_types:
        output = {"artifact_types": registry.artifact_types}
    elif args.context_bundles:
        output = {"context_bundles": registry.context_bundles}
    elif args.validators:
        output = {"validators": registry.validators}
    elif args.show:
        if args.show in registry.artifact_types:
            output = registry.artifact_types[args.show]
        elif args.show in registry.context_bundles:
            output = registry.context_bundles[args.show]
        else:
            output = {"error": f"Plugin '{args.show}' not found"}
    else:
        output = registry.to_summary_dict()

    return json.dumps(output, indent=2)


def main(argv: Optional[list[str]] = None) -> int:
    """
    Main entry point for the plugin CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Determine project root
    project_root: Optional[Path] = None
    if args.project_root:
        project_root = Path(args.project_root)

    # Load plugins
    loader = PluginLoader(project_root=project_root)
    registry = loader.load()

    # Write snapshot if requested
    if args.write_snapshot:
        from AgentQMS.agent_tools.utils.paths import get_project_root
        state_dir = (project_root or get_project_root()) / ".agentqms" / "state"
        writer = SnapshotWriter(state_dir)
        snapshot_path = writer.write(registry, loader.get_discovery_paths())
        if not args.json:
            print(f"Snapshot written to: {snapshot_path}")

    # Default to --list if no specific action
    if not any([
        args.list, args.validate, args.artifact_types,
        args.context_bundles, args.validators, args.show
    ]):
        args.list = True

    # Format output
    if args.json:
        print(format_json_output(registry, args))
    else:
        print(format_human_output(registry, args, loader.get_discovery_paths()))

    return 0 if not registry.has_errors() else 1


if __name__ == "__main__":
    sys.exit(main())

