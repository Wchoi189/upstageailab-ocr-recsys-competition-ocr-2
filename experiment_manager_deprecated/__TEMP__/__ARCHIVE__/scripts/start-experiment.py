#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_manager.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Start a new experiment")
    parser.add_argument("--type", required=True, help="Type of experiment")
    parser.add_argument("--intention", required=True, help="Intention of the experiment")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    tracker.start_experiment(args.intention, args.type)


if __name__ == "__main__":
    main()
