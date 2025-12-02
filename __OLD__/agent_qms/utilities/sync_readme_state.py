#!/usr/bin/env python3
"""
Synchronize README.md project status section with StateManager state.

This script reads the current state and updates the README's project status
table and related sections to reflect the actual project state.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_qms.toolbelt import StateManager, StateError


def parse_readme_status_table(content: str) -> Tuple[str, int, int]:
    """Parse the project status table from README.

    Returns:
        Tuple of (content_before_table, table_start_line, table_end_line)
    """
    lines = content.split('\n')

    # Find the project status table
    table_start = -1
    table_end = -1

    for i, line in enumerate(lines):
        if '## 📊 Project Status' in line:
            # Look for the table start (after the <div align="center">)
            for j in range(i + 1, min(i + 10, len(lines))):
                if '| Phase | Status | Progress |' in lines[j]:
                    table_start = j
                    break

            # Find the table end (look for closing </div> or next section)
            if table_start >= 0:
                for j in range(table_start + 1, len(lines)):
                    if lines[j].strip() == '</div>' or lines[j].strip().startswith('##'):
                        table_end = j
                        break
                break

    return '\n'.join(lines), table_start, table_end


def calculate_phase_progress_from_artifacts(
    state_manager: StateManager,
    phase_name: str
) -> Optional[int]:
    """Calculate phase progress based on artifacts related to that phase.

    This is a heuristic - in the future, this could be more sophisticated
    by looking at specific implementation plan progress trackers.
    """
    # For now, we'll use a simple heuristic based on artifact status
    # In a more sophisticated system, we'd parse implementation plan progress

    # Map phase names to artifact patterns
    phase_patterns = {
        'Phase 4': ['test', 'testing', 'quality', 'e2e', 'unit'],
        'Phase 5': ['nextjs', 'console', 'migration', 'chakra'],
    }

    artifacts = state_manager.get_all_artifacts()
    phase_artifacts = []

    for pattern in phase_patterns.get(phase_name, []):
        for artifact in artifacts:
            if pattern.lower() in artifact['path'].lower() or pattern.lower() in artifact.get('type', '').lower():
                phase_artifacts.append(artifact)

    if not phase_artifacts:
        return None

    # Count completed vs total
    completed_statuses = ['completed', 'validated', 'deployed']
    completed = sum(1 for a in phase_artifacts if a['status'] in completed_statuses)

    # Rough progress estimate
    if completed == 0:
        return 0
    elif completed == len(phase_artifacts):
        return 100
    else:
        # Estimate based on completion ratio (with some buffer for in-progress)
        in_progress = sum(1 for a in phase_artifacts if a['status'] == 'in_progress')
        estimated = int((completed / len(phase_artifacts)) * 100)
        # Add some credit for in-progress items
        if in_progress > 0:
            estimated += int((in_progress / len(phase_artifacts)) * 30)  # Max 30% for in-progress
        return min(estimated, 100)


def update_status_table(
    content: str,
    state_manager: StateManager
) -> str:
    """Update the project status table in README content."""
    lines = content.split('\n')

    # Find and update the overall progress line
    overall_progress_pattern = r'\*\*Overall Progress:\s*(\d+)%\*\*'

    # Calculate overall progress based on current state
    # This is a simple calculation - could be improved
    context = state_manager.get_current_context()
    current_phase = context.get('current_phase', 'phase-4')

    # Rough estimate based on phases
    phase_progress = {
        'phase-1': 100,
        'phase-2': 100,
        'phase-3': 100,
        'phase-4': 40,
        'phase-5': 75,
        'phase-4-5': 57,  # Average of 4 and 5
    }

    # Get a more accurate estimate if we have phase information
    overall = phase_progress.get(current_phase, 55)  # Default from README

    # Update overall progress
    content = re.sub(
        overall_progress_pattern,
        f'**Overall Progress: {overall}%**',
        content
    )

    # Note: For now, we don't automatically update individual phase rows
    # because that would require parsing implementation plan progress trackers.
    # This could be enhanced in the future.

    return content


def update_current_work_section(
    content: str,
    state_manager: StateManager
) -> str:
    """Update the 'Current Work' section based on active artifacts."""
    lines = content.split('\n')

    # Find the "### 🔥 Current Work" section
    section_start = -1
    section_end = -1

    for i, line in enumerate(lines):
        if '### 🔥 Current Work' in line or '### Current Work' in line:
            section_start = i
            # Find end of section (next ### or ##)
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith('##'):
                    section_end = j
                    break
            break

    if section_start < 0:
        return content  # Section not found, return unchanged

    # Get active artifacts
    active_artifacts = state_manager.get_active_artifacts()
    recent_artifacts = sorted(
        state_manager.get_all_artifacts(),
        key=lambda x: x.get('last_updated', x.get('created_at', '')),
        reverse=True
    )[:5]

    # For now, we'll just add a note about state tracking
    # A more sophisticated version would parse artifact contents
    # and generate detailed status

    # The section is probably fine as-is, so we'll leave it mostly unchanged
    # but could add a note about state tracking

    return content


def sync_readme(readme_path: Path, state_manager: StateManager, dry_run: bool = False):
    """Synchronize README with current state."""
    if not readme_path.exists():
        print(f"❌ README not found: {readme_path}", file=sys.stderr)
        sys.exit(1)

    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Update status table
    content = update_status_table(content, state_manager)

    # Update current work section (optional - more complex)
    # content = update_current_work_section(content, state_manager)

    # Check if changes were made
    if content == original_content:
        print("✅ README is already in sync with state.")
        return

    if dry_run:
        print("🔍 Dry run - would update README with following changes:")
        print("\n" + "="*60)
        # Simple diff - just show if overall progress changed
        original_overall = re.search(r'\*\*Overall Progress:\s*(\d+)%\*\*', original_content)
        new_overall = re.search(r'\*\*Overall Progress:\s*(\d+)%\*\*', content)
        if original_overall and new_overall:
            if original_overall.group(1) != new_overall.group(1):
                print(f"Overall Progress: {original_overall.group(1)}% → {new_overall.group(1)}%")
        print("="*60)
    else:
        # Write updated content
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ README synchronized with current state.")
        print(f"   Updated: {readme_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Synchronize README.md with StateManager state"
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=project_root / "README.md",
        help="Path to README.md (default: project_root/README.md)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually updating the file"
    )

    args = parser.parse_args()

    try:
        state_manager = StateManager()

        # Check state health
        health = state_manager.get_state_health()
        if not health['is_valid']:
            print("⚠️  Warning: State may be invalid", file=sys.stderr)

        print(f"📊 Current State:")
        print(f"   Total artifacts: {health['total_artifacts']}")
        print(f"   Current branch: {state_manager.state['current_context']['current_branch']}")
        print(f"   Current phase: {state_manager.state['current_context']['current_phase']}")
        print()

        # Sync README
        sync_readme(args.readme, state_manager, dry_run=args.dry_run)

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

