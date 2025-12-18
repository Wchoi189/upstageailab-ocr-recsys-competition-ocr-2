#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_tracker.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Stash the current experiment")

    parser.parse_args()

    tracker = ExperimentTracker()
    experiment_id = tracker._get_current_experiment_id()

    if not experiment_id:
        print("No active experiment found.")
        sys.exit(1)

    tracker.stash_incomplete(experiment_id)


if __name__ == "__main__":
    main()
