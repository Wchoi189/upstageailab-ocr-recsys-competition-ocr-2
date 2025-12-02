#!/usr/bin/env python3
"""
Regenerate documentation index and validate structure.
This script is designed to be run from pre-commit hooks and manual workflows.
"""

import subprocess
import sys

from AgentQMS.toolkit.utils.paths import get_docs_dir, get_project_root
from AgentQMS.toolkit.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

PROJECT_ROOT = get_project_root()
HANDBOOK_DIR = get_docs_dir() / "ai_handbook"
INDEX_PATH = HANDBOOK_DIR / "index.json"


def regenerate_index() -> bool:
    """Regenerate the documentation index."""
    print("🔄 Regenerating AI handbook index...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "AgentQMS/agent_tools/documentation/auto_generate_index.py",
                "--handbook-dir",
                str(HANDBOOK_DIR),
                "--output",
                str(INDEX_PATH),
                "--validate",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Index regenerated successfully")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print("❌ Index regeneration failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error regenerating index: {e}")
        return False


def validate_manifest() -> bool:
    """Validate the documentation manifest."""
    print("🔍 Validating documentation manifest...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "AgentQMS/agent_tools/documentation/validate_manifest.py",
                str(INDEX_PATH),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Manifest validation passed")
            if result.stdout and "WARNING:" in result.stdout:
                print(result.stdout)
            return True
        else:
            print("❌ Manifest validation failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error validating manifest: {e}")
        return False


def main() -> None:
    """Main regeneration workflow."""
    print("📚 Documentation Regeneration Workflow")
    print("=" * 50)

    # Check if we're in the right directory
    if not (PROJECT_ROOT / "docs" / "ai_handbook").exists():
        print("❌ Not in project root or AI handbook not found")
        sys.exit(1)

    # Regenerate index
    if not regenerate_index():
        print("❌ Failed to regenerate index")
        sys.exit(1)

    # Validate manifest
    if not validate_manifest():
        print("❌ Manifest validation failed")
        sys.exit(1)

    print("\n✅ Documentation regeneration complete!")
    print(f"📄 Index file: {INDEX_PATH}")
    print(f"📁 Handbook directory: {HANDBOOK_DIR}")


if __name__ == "__main__":
    main()
