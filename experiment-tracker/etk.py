#!/usr/bin/env python3
"""
ETK (Experiment Tracker Kit) - CLI Wrapper for EDS v1.0
Thin wrapper around the modular experiment_tracker package.
"""

import sys
from pathlib import Path

# Ensure the src directory is in sys.path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from experiment_tracker.cli import main
except ImportError:
    print("Error: Could not find experiment_tracker package in src directory.")
    sys.exit(1)

if __name__ == "__main__":
    main()
