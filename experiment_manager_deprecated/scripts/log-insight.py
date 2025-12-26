#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiment_manager.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Log an insight for the current experiment")
    parser.add_argument("insight", help="The insight to log")
    parser.add_argument("--category", default="general", help="Insight category")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    tracker.log_insight(args.insight, args.category)


if __name__ == "__main__":
    main()
