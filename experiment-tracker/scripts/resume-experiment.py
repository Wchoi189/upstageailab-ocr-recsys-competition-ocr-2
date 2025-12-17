#!/usr/bin/env python3
"""
Resume or switch to an existing experiment.

This script allows you to:
- Resume an experiment by ID
- Resume the latest experiment of a specific type
- List available experiments
- Show current active experiment
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_tracker.core import ExperimentTracker


def list_experiments(tracker: ExperimentTracker):
    """List all available experiments."""
    experiments = tracker.list_experiments()

    if not experiments:
        print("No experiments found.")
        return

    print("\nAvailable Experiments:")
    print("=" * 80)
    print(f"{'ID':<40} {'Type':<25} {'Status':<12} {'Created'}")
    print("-" * 80)

    for exp in experiments:
        exp_id = exp["id"]
        exp_type = exp.get("type", "unknown")
        status = exp.get("status", "UNKNOWN")
        timestamp = exp.get("timestamp", "")

        # Format timestamp for display
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            created = dt.strftime("%Y-%m-%d %H:%M")
        except:
            created = timestamp[:16] if len(timestamp) > 16 else timestamp

        print(f"{exp_id:<40} {exp_type:<25} {status:<12} {created}")

    print("=" * 80)


def show_current(tracker: ExperimentTracker):
    """Show the current active experiment."""
    current_id = tracker._get_current_experiment_id()

    if current_id:
        print(f"\nCurrent Active Experiment: {current_id}")

        try:
            state = tracker._load_state(current_id)
            print(f"  Type: {state.get('type', 'unknown')}")
            print(f"  Status: {state.get('status', 'unknown')}")
            print(f"  Intention: {state.get('intention', 'N/A')}")

            # Load metadata
            paths = tracker._get_paths(current_id)
            state_file = paths.get_context_file("state")
            if state_file.exists():
                import yaml

                with open(state_file) as f:
                    meta = yaml.safe_load(f) or {}
                    if "current_phase" in meta:
                        print(f"  Phase: {meta['current_phase']}")
        except Exception as e:
            print(f"  (Could not load details: {e})")
    else:
        print("\nNo active experiment.")


def resume_by_id(tracker: ExperimentTracker, experiment_id: str):
    """Resume an experiment by its ID."""
    try:
        # Verify experiment exists
        state = tracker._load_state(experiment_id)

        # Set as current
        tracker._set_current_experiment_id(experiment_id)

        print(f"\n✓ Resumed experiment: {experiment_id}")
        print(f"  Type: {state.get('type', 'unknown')}")
        print(f"  Status: {state.get('status', 'unknown')}")
        print(f"  Intention: {state.get('intention', 'N/A')}")

        # Load and show metadata
        paths = tracker._get_paths(experiment_id)
        state_file = paths.get_context_file("state")
        if state_file.exists():
            import yaml

            with open(state_file) as f:
                meta = yaml.safe_load(f) or {}
                if "current_phase" in meta:
                    print(f"  Current Phase: {meta['current_phase']}")
                if "last_updated" in meta:
                    print(f"  Last Updated: {meta['last_updated']}")

        return True
    except FileNotFoundError:
        print(f"\n✗ Experiment not found: {experiment_id}")
        print("  Use --list to see available experiments.")
        return False
    except Exception as e:
        print(f"\n✗ Error resuming experiment: {e}")
        return False


def resume_by_type(tracker: ExperimentTracker, experiment_type: str):
    """Resume the latest experiment of a specific type."""
    experiments = tracker.list_experiments()

    # Filter by type and get latest
    matching = [e for e in experiments if e.get("type") == experiment_type]

    if not matching:
        print(f"\n✗ No experiments found for type: {experiment_type}")
        print("  Use --list to see available experiments.")
        return False

    # Get latest (first in sorted list)
    latest = matching[0]
    experiment_id = latest["id"]

    print(f"\nFound latest {experiment_type} experiment: {experiment_id}")
    return resume_by_id(tracker, experiment_id)


def main():
    parser = argparse.ArgumentParser(
        description="Resume or switch to an existing experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume by ID
  resume-experiment.py --id 20251122_172313_perspective_correction

  # Resume latest of type
  resume-experiment.py --type perspective_correction

  # List all experiments
  resume-experiment.py --list

  # Show current experiment
  resume-experiment.py --current
        """,
    )

    parser.add_argument("--id", help="Experiment ID to resume (e.g., 20251122_172313_perspective_correction)")
    parser.add_argument("--type", help="Resume latest experiment of this type")
    parser.add_argument("--list", action="store_true", help="List all available experiments")
    parser.add_argument("--current", action="store_true", help="Show current active experiment")

    args = parser.parse_args()

    tracker = ExperimentTracker()

    # Handle different actions
    if args.list:
        list_experiments(tracker)
    elif args.current:
        show_current(tracker)
    elif args.id:
        resume_by_id(tracker, args.id)
    elif args.type:
        resume_by_type(tracker, args.type)
    else:
        # Default: show current
        show_current(tracker)
        print("\nUse --help for usage information.")


if __name__ == "__main__":
    main()
