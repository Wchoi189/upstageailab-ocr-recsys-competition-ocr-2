"""Runtime helpers for initializing AgentQMS tooling."""

from __future__ import annotations

import sys
from pathlib import Path

from .paths import get_project_root


def ensure_project_root_on_sys_path() -> Path:
    """Ensure the project root is importable and return it."""
    project_root = get_project_root().resolve()
    project_str = str(project_root)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)
    return project_root
