#!/usr/bin/env python3
"""
Display current project state in a readable format.

This CLI tool shows:
- Current context (branch, phase, active artifacts)
- Artifact statistics
- Recent activity
- Project health metrics
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_qms.toolbelt import StateManager, StateError


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp to readable format."""
    if not ts:
        return "Never"
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M KST")
    except Exception:
        return ts


def print_section(title: str, content: str):
    """Print a formatted section."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(content)


def show_current_context(state_manager: StateManager):
    """Display current context information."""
    context = state_manager.get_current_context()

    lines = []
    lines.append(f"Active Session:     {context['active_session_id'] or 'None'}")
    lines.append(f"Current Branch:     {context['current_branch'] or 'Unknown'}")
    lines.append(f"Current Phase:      {context['current_phase'] or 'Unknown'}")

    active_artifacts = context['active_artifacts']
    if active_artifacts:
        lines.append(f"\nActive Artifacts ({len(active_artifacts)}):")
        for artifact in active_artifacts[:5]:  # Show max 5
            lines.append(f"  • {artifact}")
        if len(active_artifacts) > 5:
            lines.append(f"  ... and {len(active_artifacts) - 5} more")
    else:
        lines.append("\nActive Artifacts:   None")

    pending_tasks = context['pending_tasks']
    if pending_tasks:
        lines.append(f"\nPending Tasks ({len(pending_tasks)}):")
        for task in pending_tasks[:5]:
            lines.append(f"  • {task}")
        if len(pending_tasks) > 5:
            lines.append(f"  ... and {len(pending_tasks) - 5} more")
    else:
        lines.append("\nPending Tasks:      None")

    print_section("Current Context", "\n".join(lines))


def show_artifact_statistics(state_manager: StateManager):
    """Display artifact statistics."""
    state = state_manager.state

    lines = []
    lines.append(f"Total Artifacts: {state['artifacts']['total_count']}")

    # By type
    by_type = state['artifacts']['by_type']
    if by_type:
        lines.append(f"\nBy Type:")
        for artifact_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"  {artifact_type:20s}: {count:3d}")

    # By status
    by_status = state['artifacts']['by_status']
    if by_status:
        lines.append(f"\nBy Status:")
        status_order = ['draft', 'in_progress', 'validated', 'completed', 'deployed', 'deprecated']
        for status in status_order:
            if status in by_status:
                lines.append(f"  {status:20s}: {by_status[status]:3d}")
        # Show any other statuses
        for status, count in sorted(by_status.items(), key=lambda x: -x[1]):
            if status not in status_order:
                lines.append(f"  {status:20s}: {count:3d}")

    print_section("Artifact Statistics", "\n".join(lines))


def show_recent_artifacts(state_manager: StateManager, limit: int = 10):
    """Display recent artifacts."""
    all_artifacts = state_manager.get_all_artifacts()

    # Sort by last_updated (most recent first)
    sorted_artifacts = sorted(
        all_artifacts,
        key=lambda x: x.get('last_updated', x.get('created_at', '')),
        reverse=True
    )[:limit]

    if not sorted_artifacts:
        print_section("Recent Artifacts", "No artifacts found.")
        return

    lines = []
    for artifact in sorted_artifacts:
        status_icon = {
            'draft': '📝',
            'in_progress': '🔄',
            'validated': '✅',
            'completed': '✅',
            'deployed': '🚀',
            'deprecated': '🗑️'
        }.get(artifact['status'], '•')

        lines.append(
            f"{status_icon} [{artifact['type']:20s}] {artifact['path']}"
        )
        lines.append(f"    Status: {artifact['status']:15s} | "
                    f"Updated: {format_timestamp(artifact.get('last_updated', artifact.get('created_at', '')))}")
        lines.append("")

    print_section(f"Recent Artifacts (Top {limit})", "\n".join(lines).rstrip())


def show_project_health(state_manager: StateManager):
    """Display project health metrics."""
    health = state_manager.get_state_health()
    stats = state_manager.get_statistics()

    lines = []
    lines.append(f"State Valid:        {'✅ Yes' if health['is_valid'] else '❌ No'}")
    lines.append(f"Schema Version:     {health['schema_version']}")
    lines.append(f"Last Updated:       {format_timestamp(health['last_updated'])}")
    lines.append(f"State File Size:    {health['state_file_size_bytes']:,} bytes")
    lines.append("")
    lines.append("Statistics:")
    lines.append(f"  Total Sessions:        {stats['total_sessions']}")
    lines.append(f"  Artifacts Created:     {stats['total_artifacts_created']}")
    lines.append(f"  Artifacts Validated:   {stats['total_artifacts_validated']}")
    lines.append(f"  Artifacts Deployed:    {stats['total_artifacts_deployed']}")

    last_session = stats.get('last_session_timestamp')
    if last_session:
        lines.append(f"  Last Session:          {format_timestamp(last_session)}")

    print_section("Project Health", "\n".join(lines))


def show_phase_progress(state_manager: StateManager):
    """Display phase progress based on README."""
    readme_path = project_root / "README.md"

    if not readme_path.exists():
        print_section("Phase Progress", "README.md not found.")
        return

    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract phase table
        lines = []
        lines.append("Phase Status (from README.md):")
        lines.append("")
        lines.append(f"{'Phase':<40} {'Status':<15} {'Progress':<10}")
        lines.append("-" * 65)

        # Parse phase table from README
        import re
        phase_pattern = r'\|\s*\*\*(Phase \d+[^\*]+)\*\*\s*\|\s*([✅🟡⚪]+)\s+([^\|]+)\s*\|\s*(\d+)%'
        matches = re.findall(phase_pattern, content)

        for phase, status_icon, status_text, progress in matches:
            phase_name = phase.strip()
            status = status_text.strip()
            progress_pct = progress.strip()
            lines.append(f"{phase_name:<40} {status:<15} {progress_pct}%")

        # Get overall progress
        overall_match = re.search(r'\*\*Overall Progress:\s*(\d+)%\*\*', content)
        if overall_match:
            lines.append("")
            lines.append(f"Overall Progress: {overall_match.group(1)}%")

        print_section("Phase Progress", "\n".join(lines))

    except Exception as e:
        print_section("Phase Progress", f"Error reading README: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Display current project state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Show all information
  %(prog)s --context          # Show only current context
  %(prog)s --artifacts 5      # Show only artifacts (top 5)
        """
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Show only current context"
    )
    parser.add_argument(
        "--artifacts",
        type=int,
        metavar="N",
        help="Show only artifacts (limit to N recent)"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Show only project health"
    )
    parser.add_argument(
        "--phases",
        action="store_true",
        help="Show only phase progress"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show only artifact statistics"
    )

    args = parser.parse_args()

    try:
        state_manager = StateManager()

        # If no specific filter, show everything
        show_all = not any([args.context, args.artifacts, args.health, args.phases, args.stats])

        print("\n" + "="*60)
        print("  PROJECT STATE OVERVIEW")
        print("="*60)

        if show_all or args.context:
            show_current_context(state_manager)

        if show_all or args.stats:
            show_artifact_statistics(state_manager)

        if show_all or args.artifacts:
            limit = args.artifacts if args.artifacts else 10
            show_recent_artifacts(state_manager, limit=limit)

        if show_all or args.phases:
            show_phase_progress(state_manager)

        if show_all or args.health:
            show_project_health(state_manager)

        print("\n" + "="*60)
        print()

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

