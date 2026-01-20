"""
Legacy import path compatibility shim.

This module provides backward compatibility for tests importing from:
    from etk.compass import CompassPaths, EnvironmentChecker, ...

The actual implementation is in:
    project_compass.src.core

This shim will be deprecated in a future release.
"""

import sys
from pathlib import Path

# Add project_compass/src to path
_project_root = Path(__file__).parent.parent.parent.parent
_compass_src = _project_root / "project_compass" / "src"
if str(_compass_src) not in sys.path:
    sys.path.insert(0, str(_compass_src))

# Import from the actual location
from core import (  # noqa: E402
    CompassPaths,
    EnvironmentChecker,
    SessionManager,
    atomic_json_write,
    atomic_yaml_write,
    format_uv_command,
)

__all__ = [
    "CompassPaths",
    "EnvironmentChecker",
    "SessionManager",
    "atomic_json_write",
    "atomic_yaml_write",
    "format_uv_command",
]
