#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_tracker.core import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Generate an assessment for the current experiment")
    parser.add_argument("--template", required=True, help="Template name")
    parser.add_argument("--verbose", default="minimal", help="Verbosity level")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    # Logic to generate assessment from template would go here
    # For now, we'll just create a placeholder file

    experiment_id = tracker._get_current_experiment_id()
    if not experiment_id:
        print("No active experiment found.")
        sys.exit(1)

    output_path = args.output
    if not output_path:
        output_path = tracker.experiments_dir / experiment_id / "assessments" / f"{args.template}_assessment.md"

    with open(output_path, "w") as f:
        f.write(f"# Assessment: {args.template}\n")
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Verbosity: {args.verbose}\n")
        f.write("\n## Findings\n\n(Auto-generated placeholder)\n")

    print(f"Generated assessment at {output_path}")


if __name__ == "__main__":
    main()
