#!/usr/bin/env python3
"""
Validate registry.yaml for broken file paths and integrity.

This script ensures all file references in registry.yaml point to existing files.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("Error: PyYAML not available. Install with: pip install pyyaml")
    sys.exit(1)


def validate_registry_paths(registry_path: str = "AgentQMS/standards/registry.yaml"):
    """Validate all file paths referenced in registry.yaml."""

    registry_file = Path(registry_path)

    if not registry_file.exists():
        print(f"❌ Error: Registry file not found: {registry_path}")
        return False

    with registry_file.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    if not registry:
        print(f"❌ Error: Registry file is empty or invalid")
        return False

    print(f"📋 Validating registry: {registry_path}")
    print("=" * 60)

    all_paths = []
    broken_paths = []

    # Extract paths from task_mappings
    if "task_mappings" in registry:
        for task_name, task_config in registry["task_mappings"].items():
            standards = task_config.get("standards", [])
            for std_path in standards:
                all_paths.append((f"task_mappings.{task_name}.standards", std_path))

    # Extract paths from tier indices
    for tier_name in ["tier1_sst", "tier2_framework", "tier3_agents", "tier4_workflows"]:
        if tier_name in registry:
            _extract_paths_recursive(registry[tier_name], tier_name, all_paths)

    # Validate each path
    print(f"\n🔍 Checking {len(all_paths)} file references...")

    for location, file_path in all_paths:
        full_path = Path(file_path)

        if not full_path.exists():
            broken_paths.append((location, file_path))
            print(f"  ❌ BROKEN: {file_path}")
            print(f"     Referenced in: {location}")
        else:
            print(f"  ✅ {file_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Total paths checked: {len(all_paths)}")
    print(f"Valid paths: {len(all_paths) - len(broken_paths)}")
    print(f"Broken paths: {len(broken_paths)}")

    if broken_paths:
        print("\n❌ VALIDATION FAILED")
        print("\nBroken references:")
        for location, file_path in broken_paths:
            print(f"  - {file_path}")
            print(f"    (in {location})")
        return False
    else:
        print("\n✅ VALIDATION PASSED - All file paths are valid!")
        return True


def _extract_paths_recursive(data, prefix, all_paths):
    """Recursively extract file paths from nested dict."""
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}"
            if isinstance(value, str) and (value.endswith(".yaml") or value.endswith(".py")):
                all_paths.append((new_prefix, value))
            elif isinstance(value, dict):
                _extract_paths_recursive(value, new_prefix, all_paths)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and (item.endswith(".yaml") or item.endswith(".py")):
                        all_paths.append((new_prefix, item))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate registry.yaml file paths"
    )
    parser.add_argument(
        "--registry",
        default="AgentQMS/standards/registry.yaml",
        help="Path to registry.yaml file"
    )

    args = parser.parse_args()

    success = validate_registry_paths(args.registry)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
