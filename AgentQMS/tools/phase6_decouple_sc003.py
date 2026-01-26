#!/usr/bin/env python3
"""
Phase 6: Decouple SC-003 Dependencies

Remove SC-003 from all Tier 2, 3, and 4 standards.
Tier 1 standards are now globally injected by the resolver.
"""

import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDARDS_DIR = PROJECT_ROOT / "AgentQMS" / "standards"

def remove_sc003_dependency(yaml_path: Path) -> bool:
    """Remove SC-003 from dependencies list in a YAML file.

    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
            data = yaml.safe_load(content)

        if not isinstance(data, dict):
            return False

        # Check if dependencies field exists and contains SC-003
        dependencies = data.get("dependencies", [])
        if not dependencies or "SC-003" not in dependencies:
            return False

        # Remove SC-003
        dependencies.remove("SC-003")

        # If dependencies list is now empty, remove it entirely
        if not dependencies:
            del data["dependencies"]
        else:
            data["dependencies"] = dependencies

        # Write back to file
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )

        return True

    except Exception as e:
        print(f"❌ Error processing {yaml_path.name}: {e}", file=sys.stderr)
        return False


def main():
    """Remove SC-003 dependencies from all non-Tier1 standards."""
    print("🔧 Phase 6: Decoupling SC-003 Dependencies")
    print("=" * 60)

    # Find all YAML files (excluding tier1-sst and registry.yaml)
    yaml_files = []
    for tier_dir in ["tier2-framework", "tier3-agents", "tier4-workflows"]:
        tier_path = STANDARDS_DIR / tier_dir
        if tier_path.exists():
            yaml_files.extend(tier_path.rglob("*.yaml"))

    print(f"\n📂 Found {len(yaml_files)} YAML files to process")

    modified_count = 0
    for yaml_path in yaml_files:
        if remove_sc003_dependency(yaml_path):
            print(f"   ✓ Removed SC-003 from {yaml_path.relative_to(STANDARDS_DIR)}")
            modified_count += 1

    print(f"\n✅ Modified {modified_count} files")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
