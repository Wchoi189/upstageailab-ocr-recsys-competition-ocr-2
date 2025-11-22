#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_tracker.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Complete a task in the current experiment")
    parser.add_argument("task_id", help="Task ID (e.g., task_001)")
    parser.add_argument("--notes", help="Completion notes")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    tracker.complete_task(args.task_id, args.notes)


if __name__ == "__main__":
    main()
