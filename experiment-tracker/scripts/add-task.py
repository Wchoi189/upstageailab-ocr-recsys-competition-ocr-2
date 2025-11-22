#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_tracker.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Add a task to the current experiment")
    parser.add_argument("description", help="Task description")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    tracker.add_task(args.description)


if __name__ == "__main__":
    main()
