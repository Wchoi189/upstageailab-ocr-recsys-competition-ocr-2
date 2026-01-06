#!/usr/bin/env python3
"""
Wrapper script for environment check.
Delegates to etk.compass.EnvironmentChecker
"""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from etk.compass import EnvironmentChecker

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
