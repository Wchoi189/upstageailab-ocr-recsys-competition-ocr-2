#!/usr/bin/env python3
"""
Initialize AgentQMS state with current project data.

This script scans the project for artifacts, gets current git branch,
and initializes the state tracking system with current project status.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_qms.toolbelt import StateManager, StateError


def get_git_branch() -> Optional[str]:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=True
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_artifact_type_from_path(path: Path) -> str:
    """Determine artifact type from path."""
    if "implementation_plans" in str(path):
        return "implementation_plan"
    elif "assessments" in str(path):
        return "assessment"
    elif "data_contracts" in str(path):
        return "data_contract"
    elif "bug_reports" in str(path):
        return "bug_report"
    else:
        return "unknown"


def get_artifact_status_from_frontmatter(path: Path) -> str:
    """Extract status from artifact frontmatter."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read(2000)  # Read first 2KB

        # Look for status in frontmatter
        status_match = re.search(r'^status:\s*(.+)$', content, re.MULTILINE)
        if status_match:
            status = status_match.group(1).strip().strip('"\'')
            # Normalize status values
            status_lower = status.lower()
            if status_lower in ['draft', 'in_progress', 'validated', 'deployed', 'deprecated', 'completed']:
                return status_lower
            elif status_lower in ['complete', 'done']:
                return 'completed'

        return "draft"
    except Exception:
        return "draft"


def scan_artifacts(artifacts_dir: Path) -> list[Dict]:
    """Scan artifacts directory and return artifact metadata."""
    artifacts = []

    if not artifacts_dir.exists():
        return artifacts

    # Scan all markdown files in artifacts directory
    for md_file in artifacts_dir.rglob("*.md"):
        # Skip index files
        if md_file.name == "INDEX.md" or md_file.name == "MASTER_INDEX.md":
            continue

        # Get relative path from project root
        rel_path = md_file.relative_to(project_root)
        artifact_type = get_artifact_type_from_path(md_file)
        status = get_artifact_status_from_frontmatter(md_file)

        # Get file timestamps
        stat = md_file.stat()

        artifacts.append({
            "path": str(rel_path),
            "type": artifact_type,
            "status": status,
            "created_at": None,  # Will use file mtime if no frontmatter date
            "last_updated": None,
            "metadata": {
                "file_size": stat.st_size,
                "file_mtime": stat.st_mtime
            }
        })

    return artifacts


def determine_current_phase(state_manager: StateManager) -> str:
    """Determine current project phase based on README and artifacts."""
    # Try to read README to determine phase
    readme_path = project_root / "README.md"
    if readme_path.exists():
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()

            # Check for "In Progress" phases in status table
            if "Phase 4: Testing & Quality Assurance" in readme_content and "🟡 In Progress" in readme_content:
                if "Phase 5" in readme_content and "🟡 In Progress" in readme_content:
                    return "phase-4-5"  # Both phases active
                return "phase-4"
            elif "Phase 5: Next.js Console Migration" in readme_content and "🟡 In Progress" in readme_content:
                return "phase-5"
        except Exception:
            pass

    # Default to current branch or phase-4
    return "phase-4"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Initialize AgentQMS state with current project data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-initialization even if state exists"
    )
    parser.add_argument(
        "--phase",
        type=str,
        help="Set current phase (default: auto-detect)"
    )
    args = parser.parse_args()

    try:
        # Initialize state manager
        state_manager = StateManager()

        # Check if state already exists and has artifacts
        if state_manager.state['artifacts']['total_count'] > 0 and not args.force:
            print(f"State already initialized with {state_manager.state['artifacts']['total_count']} artifacts.")
            print("Use --force to re-initialize.")
            return

        print("Initializing state tracking...")

        # Get current git branch
        current_branch = get_git_branch()
        if current_branch:
            print(f"Current git branch: {current_branch}")
            state_manager.set_current_branch(current_branch)

        # Determine current phase
        if args.phase:
            current_phase = args.phase
        else:
            current_phase = determine_current_phase(state_manager)
        print(f"Current phase: {current_phase}")
        state_manager.set_current_phase(current_phase)

        # Scan artifacts
        artifacts_dir = project_root / "artifacts"
        print(f"Scanning artifacts in {artifacts_dir}...")
        artifacts = scan_artifacts(artifacts_dir)

        if not args.force:
            # Only add artifacts that aren't already in the index
            existing_paths = {a['path'] for a in state_manager.get_all_artifacts()}
            new_artifacts = [a for a in artifacts if a['path'] not in existing_paths]
        else:
            # Force mode: clear existing artifacts and re-add all
            new_artifacts = artifacts
            state_manager.state['artifacts']['index'] = []
            state_manager.state['artifacts']['total_count'] = 0
            state_manager.state['artifacts']['by_type'] = {}
            state_manager.state['artifacts']['by_status'] = {}

        # Add artifacts to state
        print(f"Adding {len(new_artifacts)} artifacts to state...")
        for artifact in new_artifacts:
            state_manager.add_artifact(
                artifact_path=artifact['path'],
                artifact_type=artifact['type'],
                status=artifact['status'],
                metadata=artifact['metadata']
            )

        # Print summary
        stats = state_manager.get_statistics()
        health = state_manager.get_state_health()

        print("\n✅ State initialization complete!")
        print(f"   Total artifacts: {health['total_artifacts']}")
        print(f"   Current branch: {state_manager.state['current_context']['current_branch']}")
        print(f"   Current phase: {state_manager.state['current_context']['current_phase']}")
        print(f"   Artifacts by type: {state_manager.state['artifacts']['by_type']}")
        print(f"   Artifacts by status: {state_manager.state['artifacts']['by_status']}")

    except StateError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

