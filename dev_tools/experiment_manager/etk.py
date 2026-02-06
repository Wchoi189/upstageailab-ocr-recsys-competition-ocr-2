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
    from etk.cli import main
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import ETK core components.\nDetail: {e}")
    sys.exit(1)

if __name__ == "__main__":
    main()
