#!/usr/bin/env python3
"""
Environment validation script - Run this after environment setup
to catch issues early before starting work.

Usage:
    python scripts/validate_environment.py
"""

import importlib
import sys
from pathlib import Path


def validate_environment() -> bool:
    """Validate environment is correctly set up."""
    errors = []

    print("🔍 Validating environment...\n")

    # Check 1: Python version
    print("✓ Python version:", sys.version.split()[0])
    if sys.version_info < (3, 11):
        errors.append("Python 3.11+ required")

    # Check 2: Critical imports
    required_modules = [
        "torch",
        "numpy",
        "cv2",
        "lightning",
        "wandb",
        "mypy",
        "ruff",
    ]

    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError:
            errors.append(f"Missing: {module}")
            print(f"✗ {module}")

    # Check 3: PYTHONPATH
    # We expect the project root to be importable or in path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        # It's okay if it's installed as a package, but let's warn if strictly missing from path
        # and checking if 'ocr' import works is the real test.
        # But for development, having root in path is often useful if not installed.
        print(f"ℹ️  Note: {project_root} not in PYTHONPATH (normal if installed as package)")
        # print(f"   Current PYTHONPATH: {sys.path}")

    # Check 4: Project modules
    try:
        import ocr
        print(f"✓ ocr module (from {ocr.__file__})")
    except ImportError as e:
        errors.append(f"Cannot import ocr: {e}")
        print("✗ ocr module")

    try:
        import AgentQMS
        print(f"✓ AgentQMS module (from {AgentQMS.__file__})")
    except ImportError as e:
        errors.append(f"Cannot import AgentQMS: {e}")
        print("✗ AgentQMS module")

    # Check 5: GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available (CPU only) - This is expected in standard Codespaces")
    except Exception as e:
        print(f"⚠️  Cannot check CUDA: {e}")

    # Results
    print("\n" + "=" * 60)
    if errors:
        print("❌ VALIDATION FAILED\n")
        for error in errors:
            print(f"  • {error}")
        print("\nFix these issues before starting work.")
        return False
    else:
        print("✅ ENVIRONMENT VALID")
        print("\nAll checks passed! Ready to work.")
        return True


if __name__ == "__main__":
    sys.exit(0 if validate_environment() else 1)
