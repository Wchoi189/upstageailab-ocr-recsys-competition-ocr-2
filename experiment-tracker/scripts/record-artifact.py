#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_tracker.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Record an artifact for the current experiment")
    parser.add_argument("--path", required=True, help="Path to the artifact file")
    parser.add_argument("--type", default="unknown", help="Type of artifact")
    parser.add_argument("--metadata", help="JSON string of metadata")

    args = parser.parse_args()

    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Error: Metadata must be a valid JSON string")
            sys.exit(1)

    metadata["type"] = args.type

    tracker = ExperimentTracker()
    tracker.record_artifact(args.path, metadata)


if __name__ == "__main__":
    main()
