#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_manager.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Migrate an experiment to v2 structure")
    parser.add_argument("--id", help="Experiment ID (defaults to current)")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    experiment_id = args.id or tracker._get_current_experiment_id()

    if not experiment_id:
        print("No experiment ID provided and no active experiment found.")
        sys.exit(1)

    print(f"Migrating experiment: {experiment_id}")

    paths = tracker._get_paths(experiment_id)
    paths.ensure_structure()

    state = tracker._load_state(experiment_id)
    tracker._init_metadata(paths, state)

    print("Migration complete.")


if __name__ == "__main__":
    main()
