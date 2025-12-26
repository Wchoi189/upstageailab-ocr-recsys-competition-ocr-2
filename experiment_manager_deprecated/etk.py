#!/usr/bin/env python3
"""
ETK (Experiment Tracker Kit) - CLI for EDS v1.0
"""

import sys
from pathlib import Path

# Add parent directory to path for experiment_manager package
parent = Path(__file__).parent.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

from experiment_manager.cli import main

if __name__ == "__main__":
    main()
