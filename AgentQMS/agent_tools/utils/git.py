#!/usr/bin/env python3
"""
Git Utilities for AgentQMS

Provides functions to detect and manage git branch information for artifact metadata.

Usage:
    from AgentQMS.agent_tools.utils.git import get_current_branch, validate_branch_name

    branch = get_current_branch()  # Returns current branch or "main" as fallback
    is_valid = validate_branch_name(branch)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from AgentQMS.agent_tools.utils.paths import get_project_root


def get_current_branch(project_root: Path | None = None) -> str:
    """
    Get the current git branch name.

    Falls back to "main" if not in a git repository or git is unavailable.

    Args:
        project_root: Project root path (auto-detected if not provided)

    Returns:
        Current branch name or "main" as fallback
    """
    if project_root is None:
        project_root = get_project_root()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0:
            branch = result.stdout.strip()
            if branch and validate_branch_name(branch):
                return branch
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    # Fallback to "main"
    return "main"


def get_default_branch(config: dict | None = None) -> str:
    """
    Get the default branch from configuration.

    Args:
        config: Configuration dictionary with optional 'default_branch' key

    Returns:
        Default branch name from config or "main"
    """
    if config and isinstance(config, dict):
        return config.get("default_branch", "main")
    return "main"


def validate_branch_name(branch_name: str) -> bool:
    """
    Validate branch name format.

    Allows alphanumeric characters, underscores, hyphens, and forward slashes.
    Pattern: ^[a-zA-Z0-9_/-]+$

    Args:
        branch_name: Branch name to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(branch_name, str) or not branch_name:
        return False

    pattern = r"^[a-zA-Z0-9_/-]+$"
    return bool(re.match(pattern, branch_name))


def get_default_branch_from_remote(project_root: Path | None = None) -> str:
    """
    Attempt to detect default branch from remote (main, master, trunk, etc).

    Args:
        project_root: Project root path (auto-detected if not provided)

    Returns:
        Detected default branch or "main" as fallback
    """
    if project_root is None:
        project_root = get_project_root()

    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0:
            # Output is like "refs/remotes/origin/main"
            parts = result.stdout.strip().split("/")
            if parts:
                branch = parts[-1]
                if validate_branch_name(branch):
                    return branch
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return "main"


def is_in_git_repository(project_root: Path | None = None) -> bool:
    """
    Check if project root is in a git repository.

    Args:
        project_root: Project root path (auto-detected if not provided)

    Returns:
        True if in git repository, False otherwise
    """
    if project_root is None:
        project_root = get_project_root()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def get_git_commit_for_file(file_path: Path, project_root: Path | None = None) -> str | None:
    """
    Get the git commit hash that created or last modified a file.

    Useful for retroactively assigning branch information to legacy artifacts.

    Args:
        file_path: Path to file
        project_root: Project root path (auto-detected if not provided)

    Returns:
        Commit hash or None if not found
    """
    if project_root is None:
        project_root = get_project_root()

    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(file_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return None
