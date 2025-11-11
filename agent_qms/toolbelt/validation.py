"""
Validation helpers to prevent manual artifact creation.

This module provides validation functions that should be called
BEFORE creating any artifact files manually.
"""

from pathlib import Path
from typing import Optional


class ManualCreationError(Exception):
    """Raised when manual artifact creation is detected."""
    pass


def validate_artifact_path(file_path: str | Path, raise_error: bool = True) -> tuple[bool, Optional[str]]:
    """
    Validate that a file path is NOT being created manually in artifact directories.

    This function should be called BEFORE using the `write` tool to create
    files in artifact directories. It checks if the path is in a managed
    artifact directory and raises an error if manual creation is detected.

    Args:
        file_path: The file path to validate
        raise_error: If True, raise ManualCreationError on violation

    Returns:
        Tuple of (is_valid, error_message)

    Raises:
        ManualCreationError: If manual creation is detected and raise_error=True
    """
    file_path = Path(file_path)

    # Normalize path to check if it's in artifact directories
    path_str = str(file_path)

    # Check for artifact directories
    artifact_dirs = [
        "artifacts/",
        "docs/bug_reports/",
    ]

    for artifact_dir in artifact_dirs:
        if artifact_dir in path_str:
            error_msg = (
                f"❌ FORBIDDEN: Manual creation of artifact file detected: {file_path}\n"
                f"   Artifact files in '{artifact_dir}' MUST be created using:\n"
                f"   1. AgentQMS toolbelt (preferred):\n"
                f"      from agent_qms.toolbelt import AgentQMSToolbelt\n"
                f"      toolbelt = AgentQMSToolbelt()\n"
                f"      toolbelt.create_artifact(...)\n"
                f"   2. Artifact workflow script:\n"
                f"      python scripts/agent_tools/core/artifact_workflow.py create --type [TYPE] --name [NAME] --title \"[TITLE]\"\n"
                f"   Manual creation bypasses validation and violates project rules."
            )

            if raise_error:
                raise ManualCreationError(error_msg)

            return False, error_msg

    return True, None


def check_before_write(file_path: str | Path) -> None:
    """
    Check if a file path is in an artifact directory before writing.

    This is a convenience function that always raises an error if manual
    creation is detected. Use this before calling the `write` tool.

    Args:
        file_path: The file path to check

    Raises:
        ManualCreationError: If manual creation is detected
    """
    validate_artifact_path(file_path, raise_error=True)

