"""\nDEPRECATED: 
Legacy import path compatibility shim.

This module provides backward compatibility for tests importing from:
    from etk.compass import CompassPaths, EnvironmentChecker, ...

The actual implementation is in:
⚠️  WARNING: This compatibility shim will be REMOVED in v0.4.0
⚠️  Update your imports to use the new path

"""

import warnings

warnings.warn(
    "Importing from 'etk.compass' is deprecated. "
    "Use 'project_compass.src.core' instead. "
    "This compatibility shim will be removed in v0.4.0.",
    DeprecationWarning,
    stacklevel=2
)


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
