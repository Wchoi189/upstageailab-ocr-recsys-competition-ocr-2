#!/usr/bin/env python3
"""
Wrapper script for artifact validation.
Delegates to AgentQMS/tools/compliance/validate_artifacts.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from AgentQMS.tools.compliance.validate_artifacts import main

if __name__ == "__main__":
    sys.exit(main())
