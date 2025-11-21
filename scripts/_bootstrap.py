#!/usr/bin/env python3
"""
Bootstrap module for scripts in the scripts/ directory.

This module provides path setup utilities for scripts that need to import
from the project modules. It ensures the project root is in sys.path and
provides access to path resolution utilities.
"""

import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Find the project root directory."""
    current_path = Path(__file__).resolve().parent

    # Look for common project markers
    project_markers = ["pyproject.toml", "requirements.txt", "setup.py", ".git", "agent_qms"]

    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in project_markers):
            return parent

    # Fallback: assume parent of scripts/ is project root
    return current_path.parent


def setup_project_paths():
    """Setup project paths by adding project root to sys.path."""
    project_root = _find_project_root()
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    return project_root


def get_path_resolver():
    """Get path resolver from ocr.utils.path_utils if available."""
    try:
        from ocr.utils.path_utils import get_path_resolver as _get_path_resolver

        return _get_path_resolver()
    except ImportError:
        # Fallback: return a simple resolver
        project_root = _find_project_root()

        class SimplePathResolver:
            def __init__(self, root: Path):
                self.project_root = root

        return SimplePathResolver(project_root)


# Auto-setup paths when module is imported
setup_project_paths()
