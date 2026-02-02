#!/usr/bin/env python3
"""Dashboard CLI tool for viewing AgentQMS middleware statistics.

Displays policy enforcement rates, exception summary, and health status
from collected middleware telemetry and metrics.

Usage:
    uv run python AgentQMS/tools/middleware/dashboard.py
    uv run python AgentQMS/tools/middleware/dashboard.py --json
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def load_metrics(metrics_path: Path) -> dict | None:
    """Load metrics from JSON file.

    Args:
        metrics_path: Path to metrics JSON file

    Returns:
        Metrics dictionary or None if file doesn't exist
    """
    if not metrics_path.exists():
        return None

    try:
        with metrics_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}", file=sys.stderr)
        return None


def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to human-readable format.

    Args:
        iso_timestamp: ISO format timestamp string

    Returns:
        Human-readable timestamp
    """
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return iso_timestamp


def display_metrics_table(metrics: dict) -> None:
    """Display metrics in formatted table.

    Args:
        metrics: Metrics dictionary
    """
    print("\n" + "=" * 80)
    print("=== Middleware Statistics ===")
    print("=" * 80)

    # Display timestamp and period
    timestamp = format_timestamp(metrics.get("timestamp", "Unknown"))
    period = metrics.get("period", "Unknown")
    print(f"\nPeriod: {period}")
    print(f"Last Updated: {timestamp}\n")

    # Policy Enforcement Table
    policies = metrics.get("policies", {})
    if policies:
        print("Policy Enforcement:")
        print("┌─────────────────────────┬────────┬────────────┬────────────┬──────────────┐")
        print("│ Policy                  │ Checks │ Violations │ Exceptions │ Avg Time (ms)│")
        print("├─────────────────────────┼────────┼────────────┼────────────┼──────────────┤")

        for policy_name, stats in policies.items():
            # Truncate policy name if too long
            display_name = policy_name[:23] if len(policy_name) > 23 else policy_name
            checks = stats.get("checks", 0)
            violations = stats.get("violations", 0)
            exceptions = stats.get("exceptions", 0)
            avg_duration = stats.get("avg_duration_ms", 0.0)

            print(
                f"│ {display_name:<23} │ {checks:>6} │ {violations:>10} │ {exceptions:>10} │ {avg_duration:>12.2f} │"
            )

        print("└─────────────────────────┴────────┴────────────┴────────────┴──────────────┘")
    else:
        print("No policy data available.")

    # Exception Summary
    exceptions = metrics.get("exceptions", {})
    if exceptions:
        print("\nException Summary:")
        for exc_type, count in exceptions.items():
            print(f"  - {exc_type}: {count}")
    else:
        print("\nNo exceptions recorded.")

    print("\n" + "=" * 80 + "\n")


def display_health_status() -> None:
    """Display health status of middleware."""
    try:
        from AgentQMS.middleware.health import get_health_status

        health = get_health_status()

        print("\n" + "=" * 80)
        print("=== Health Status ===")
        print("=" * 80)

        status = health.get("status", "unknown")
        status_symbol = "✅" if status == "healthy" else "⚠️"
        print(f"\nOverall Status: {status_symbol} {status.upper()}\n")

        summary = health.get("summary", {})
        print(f"Total Checks: {summary.get('total_checks', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Failed: {summary.get('failed', 0)}\n")

        details = health.get("details", {})
        if details:
            print("Component Status:")
            for component, result in details.items():
                if result == "ok":
                    print(f"  ✅ {component}: OK")
                else:
                    error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                    print(f"  ❌ {component}: FAILED - {error_msg}")

        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"Error checking health status: {e}", file=sys.stderr)


def main():
    """Main entry point for the dashboard CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="View AgentQMS middleware statistics and health status"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted display"
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="Path to metrics file (default: outputs/metrics/middleware_stats.json)",
    )
    parser.add_argument(
        "--health-only", action="store_true", help="Show only health status"
    )

    args = parser.parse_args()

    # Determine metrics path
    if args.metrics_file:
        metrics_path = args.metrics_file
    else:
        try:
            from AgentQMS.tools.utils.paths import get_project_root

            metrics_path = (
                get_project_root() / "outputs" / "metrics" / "middleware_stats.json"
            )
        except Exception:
            metrics_path = Path("outputs/metrics/middleware_stats.json")

    # Display health status if requested
    if args.health_only:
        display_health_status()
        return

    # Load and display metrics
    metrics = load_metrics(metrics_path)

    if metrics is None:
        print(f"No metrics file found at: {metrics_path}", file=sys.stderr)
        print("\nTip: Run some middleware operations first to generate metrics.", file=sys.stderr)

        # Still show health status
        print("\nAttempting to check health status...")
        display_health_status()
        sys.exit(1)

    if args.json:
        # Output raw JSON
        print(json.dumps(metrics, indent=2))
    else:
        # Display formatted tables
        display_metrics_table(metrics)
        display_health_status()


if __name__ == "__main__":
    main()
