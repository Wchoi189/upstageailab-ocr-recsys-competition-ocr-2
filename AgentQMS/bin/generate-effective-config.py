#!/usr/bin/env python3
"""
Generate effective.yaml with path-aware standard discovery.

This script demonstrates the new dynamic context injection capability where
standards are automatically resolved based on the current working directory.

Usage:
    python generate-effective-config.py
    python generate-effective-config.py --path ocr/inference
    python generate-effective-config.py --output custom-effective.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from AgentQMS.tools.utils.config_loader import ConfigLoader

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available. Install with: pip install pyyaml")


def main():
    parser = argparse.ArgumentParser(
        description="Generate effective.yaml with path-aware standard discovery"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Current working path for standard discovery (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="AgentQMS/.agentqms/effective.yaml",
        help="Output path for effective.yaml (default: AgentQMS/.agentqms/effective.yaml)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="AgentQMS/standards/registry.yaml",
        help="Path to registry.yaml (default: AgentQMS/standards/registry.yaml)",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="AgentQMS/.agentqms/settings.yaml",
        help="Path to settings.yaml (default: AgentQMS/.agentqms/settings.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print to stdout instead of writing file",
    )

    args = parser.parse_args()

    # Create loader
    loader = ConfigLoader()

    # Generate effective config with path-aware discovery
    effective = loader.generate_effective_config(
        settings_path=args.settings,
        registry_path=args.registry,
        current_path=args.path,
    )

    # Add tool_mappings from original settings if present
    settings = loader.get_config(args.settings, defaults={})
    if "tool_mappings" in settings.get("resolved", {}):
        effective["resolved"]["tool_mappings"] = settings["resolved"]["tool_mappings"]

    # Output
    if not YAML_AVAILABLE:
        print("Error: PyYAML required to write YAML files")
        print("Install with: pip install pyyaml")
        sys.exit(1)

    yaml_output = yaml.dump(effective, sort_keys=False, default_flow_style=False)

    if args.dry_run:
        print("=" * 60)
        print("Generated effective.yaml (dry-run):")
        print("=" * 60)
        print(yaml_output)
        print("=" * 60)
        print(f"\nActive Standards ({len(effective['resolved']['context_integration'].get('active_standards', []))}):")
        for std in effective["resolved"]["context_integration"].get("active_standards", []):
            print(f"  - {std}")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(yaml_output)

        print(f"✅ Generated: {output_path}")
        print(f"📁 Current path: {args.path or 'current directory'}")
        print(f"📋 Active standards: {len(effective['resolved']['context_integration'].get('active_standards', []))}")

        active_standards = effective["resolved"]["context_integration"].get("active_standards", [])
        if active_standards:
            print("\nActive Standards:")
            for std in active_standards[:5]:  # Show first 5
                print(f"  - {std}")
            if len(active_standards) > 5:
                print(f"  ... and {len(active_standards) - 5} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
