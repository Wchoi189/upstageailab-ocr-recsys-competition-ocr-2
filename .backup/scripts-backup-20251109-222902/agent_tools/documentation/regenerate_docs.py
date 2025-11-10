#!/usr/bin/env python3
"""
Regenerate documentation index and validate structure.
This script is designed to be run from pre-commit hooks and manual workflows.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path


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

PROJECT_ROOT = _RESOLVER.project_root
HANDBOOK_DIR = _RESOLVER.config.docs_dir / "ai_handbook"
INDEX_PATH = HANDBOOK_DIR / "index.json"


def regenerate_index() -> bool:
    """Regenerate the documentation index."""
    print("🔄 Regenerating AI handbook index...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/agent_tools/documentation/auto_generate_index.py",
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
                "scripts/agent_tools/documentation/validate_manifest.py",
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
