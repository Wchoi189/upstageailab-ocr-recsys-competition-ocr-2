#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiment_manager.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Add a task to the current experiment")
    parser.add_argument("description", help="Task description")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--no-context", action="store_true", help="Skip context display")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    tracker.add_task(args.description, show_context=not args.no_context, confirm=not args.no_confirm)


if __name__ == "__main__":
    main()
