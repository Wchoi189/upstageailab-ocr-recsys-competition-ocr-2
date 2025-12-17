#!/usr/bin/env python3
"""
Append run data to metrics artifact table.

Usage:
    python append-run.py \\
        --experiment 20251217_024343_image_enhancements_implementation \\
        --metrics-file 20251217_1800_report_run-metrics-phase1.md \\
        --run-id 004 \\
        --params "kernel=5,threshold=0.8" \\
        --metrics "0.89,44,98" \\
        --status "✅" \\
        --notes "Kernel tuning experiment"
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


def find_experiment_root() -> Path:
    """Find experiment-tracker root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "experiment-tracker").exists():
            return current / "experiment-tracker"
        if (current / ".ai-instructions").exists() and (current / "experiments").exists():
            return current
        current = current.parent

    raise FileNotFoundError("Could not find experiment-tracker root directory")


def append_run(metrics_file: Path, run_data: dict) -> bool:
    """
    Append new run row to metrics table.

    Args:
        metrics_file: Path to metrics artifact
        run_data: Dict with run_id, date, params, metrics, status, notes

    Returns:
        True if successful, False otherwise
    """

    if not metrics_file.exists():
        print(f"❌ ERROR: Metrics file not found: {metrics_file}", file=sys.stderr)
        return False

    # Read current file
    with open(metrics_file, encoding='utf-8') as f:
        content = f.read()

    # Find Run History table
    # Pattern: ## Run History header, followed by table header, followed by separator, followed by rows
    table_pattern = r'(## Run History.*?\n\|[^\n]+\n\|[^\n]+\n)(.*?)(\n\n##|\Z)'
    match = re.search(table_pattern, content, re.DOTALL)

    if not match:
        print(f"❌ ERROR: Run History table not found in {metrics_file.name}", file=sys.stderr)
        print("Expected table structure:", file=sys.stderr)
        print("## Run History", file=sys.stderr)
        print("| Run | Date | ... |", file=sys.stderr)
        print("|-----|------|-----|", file=sys.stderr)
        return False

    # Build new row
    # Split metrics by comma and format with pipes
    metrics_formatted = ' | '.join(run_data['metrics'].split(','))

    new_row = (
        f"| {run_data['run_id']} "
        f"| {run_data['date']} "
        f"| {run_data['params']} "
        f"| {metrics_formatted} "
        f"| {run_data['status']} "
        f"| {run_data['notes']} |\n"
    )

    # Insert new row at end of table (before next section or EOF)
    table_start = match.start(2)
    table_end = match.end(2)

    # Append to existing rows
    new_content = content[:table_end] + new_row + content[table_end:]

    # Update frontmatter timestamp
    new_content = re.sub(
        r'updated: "[^"]*"',
        f'updated: "{datetime.utcnow().isoformat()}Z"',
        new_content
    )

    # Write back
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Append run data to metrics artifact table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python append-run.py \\
    --experiment 20251217_024343_image_enhancements_implementation \\
    --metrics-file 20251217_1800_report_run-metrics-phase1.md \\
    --run-id 004 \\
    --params "kernel=5,threshold=0.8" \\
    --metrics "0.089,44,98.5" \\
    --status "✅" \\
    --notes "Kernel tuning"
        """
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment ID (e.g., 20251217_024343_image_enhancements_implementation)"
    )
    parser.add_argument(
        "--metrics-file",
        required=True,
        help="Metrics artifact filename (e.g., 20251217_1800_report_run-metrics-phase1.md)"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID (e.g., 004)"
    )
    parser.add_argument(
        "--params",
        required=True,
        help="Parameters (comma-separated key=value pairs)"
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Metric values (comma-separated, e.g., '0.089,44,98.5')"
    )
    parser.add_argument(
        "--status",
        required=True,
        help="Status indicator (✅/⚠️/❌/🔄/⏸️)"
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Notes (optional)"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date override (default: today, format: YYYY-MM-DD)"
    )

    args = parser.parse_args()

    try:
        # Find experiment tracker root
        tracker_root = find_experiment_root()

        # Build run data
        run_data = {
            'run_id': args.run_id,
            'date': args.date or datetime.now().strftime('%Y-%m-%d'),
            'params': args.params,
            'metrics': args.metrics,
            'status': args.status,
            'notes': args.notes
        }

        # Find metrics file
        metrics_path = tracker_root / "experiments" / args.experiment / args.metrics_file

        # Append run
        success = append_run(metrics_path, run_data)

        if success:
            print(f"✅ Appended run {args.run_id} to {args.metrics_file}")
            print(f"📊 File: {metrics_path}")
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"❌ ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
