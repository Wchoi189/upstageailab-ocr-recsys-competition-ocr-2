#!/usr/bin/env python3
"""
Wrapper script for environment check.
Delegates to etk.compass.EnvironmentChecker
"""
import sys
from pathlib import Path

# Add project root to sys.path
from ocr.core.utils.path_utils import PROJECT_ROOT
# Add dev_tools to sys.path to allow project_compass import
sys.path.insert(0, str(PROJECT_ROOT / "dev_tools"))

from project_compass.src.core import EnvironmentChecker

def main():
    print("🔒 Environment Guard: Checking against Project Compass lock state...\n")
    checker = EnvironmentChecker()
    passed, errors, warnings = checker.check_all()

    if warnings:
        for warning in warnings:
            print(f"⚠️  {warning}")
        print()

    if errors:
        print("❌ ENVIRONMENT BREACH DETECTED\n")
        for error in errors:
            print(f"  ✗ {error}\n")
        print("\n🔧 Path Restoration Instructions:")
        print("   1. Ensure you are using the correct UV binary")
        print("   2. Run: uv sync")
        print('   3. Verify with: uv run python -c "import torch; print(torch.__version__)"')
        sys.exit(1)
    else:
        print("✅ Environment validated against Compass lock state")
        sys.exit(0)

if __name__ == "__main__":
    main()
