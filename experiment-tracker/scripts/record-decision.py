#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_tracker.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Record a decision for the current experiment")
    parser.add_argument("decision", help="The decision made")
    parser.add_argument("--rationale", required=True, help="Rationale for the decision")
    parser.add_argument("--alternatives", nargs="*", help="Alternatives considered")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    tracker.record_decision(args.decision, args.rationale, args.alternatives)


if __name__ == "__main__":
    main()
