#!/usr/bin/env python3
"""
AgentQMS Migration Package Script

This script migrates the AgentQMS framework into an existing project.
It copies the necessary directories, adjusts paths, and sets up artifact directories.

Usage:
    python migration_package.py <target_directory>

Example:
    python migration_package.py /path/to/your/project
"""

import shutil
import os
import sys
from pathlib import Path

def copy_framework(target_dir):
    """Copy AgentQMS framework directories to target."""
    source_agentqms = Path(__file__).parent.parent
    source_dot_agentqms = Path(__file__).parent.parent.parent / ".agentqms"

    target_agentqms = Path(target_dir) / "AgentQMS"
    target_dot_agentqms = Path(target_dir) / ".agentqms"

    if source_agentqms.exists():
        shutil.copytree(source_agentqms, target_agentqms, dirs_exist_ok=True)
    if source_dot_agentqms.exists():
        shutil.copytree(source_dot_agentqms, target_dot_agentqms, dirs_exist_ok=True)

def scan_and_adjust_paths(target_dir):
    """Scan for relative paths and adjust them if necessary."""
    # This is a placeholder; implement path scanning logic as needed
    # For example, check for paths in config files and adjust relative to target_dir
    pass

def create_artifacts_dirs(target_dir):
    """Create artifact subdirectories in docs/artifacts/."""
    artifacts_dir = Path(target_dir) / "docs" / "artifacts"
    subdirs = ["assessments", "bug_reports", "completed_plans", "design_documents", "implementation_plans", "research"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (artifacts_dir / subdir).mkdir(exist_ok=True)

def print_instructions():
    """Print usage instructions."""
    print("""
AgentQMS Migration Complete!

Next steps:
1. Review the copied AgentQMS/ and .agentqms/ directories in your project.
2. Run 'cd AgentQMS/interface && make validate' to check compliance.
3. Refer to MIGRATION.md for detailed usage instructions.
4. Integrate AgentQMS tools into your development workflow.
""")

def main():
    if len(sys.argv) != 2:
        print("Usage: python migration_package.py <target_directory>")
        sys.exit(1)

    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory.")
        sys.exit(1)

    copy_framework(target_dir)
    scan_and_adjust_paths(target_dir)
    create_artifacts_dirs(target_dir)
    print_instructions()

if __name__ == "__main__":
    main()
