#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_tracker.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Export an experiment")
    parser.add_argument("--id", help="Experiment ID (defaults to current)")
    parser.add_argument("--format", default="archive", help="Export format")
    parser.add_argument("--destination", default="./exports", help="Destination directory")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    experiment_id = args.id or tracker._get_current_experiment_id()

    if not experiment_id:
        print("No experiment ID provided and no active experiment found.")
        sys.exit(1)

    tracker.export_experiment(experiment_id, args.format, args.destination)


if __name__ == "__main__":
    main()
