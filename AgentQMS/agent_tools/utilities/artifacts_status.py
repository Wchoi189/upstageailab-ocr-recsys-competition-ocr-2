#!/usr/bin/env python3
"""
Artifact Status Dashboard
Shows aging status, lifecycle state, and version information for artifacts.

Usage:
    python artifacts_status.py [OPTIONS]

Examples:
    python artifacts_status.py --dashboard
    python artifacts_status.py --json
    python artifacts_status.py --aging-only
    python artifacts_status.py --threshold 30  # Show artifacts > 30 days old
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from AgentQMS.agent_tools.utilities.versioning import (
    ArtifactAgeDetector,
    ArtifactLifecycle,
    VersionManager,
)


@dataclass
class ArtifactStatus:
    """Status of a single artifact."""

    path: str
    age_days: int | None
    age_category: str
    lifecycle_state: str
    version: str | None
    warning_status: str


def get_artifact_status(artifact_path: Path) -> ArtifactStatus:
    """Get status for a single artifact."""
    try:
        # Get age information
        detector = ArtifactAgeDetector()
        age_days = detector.get_artifact_age(artifact_path)
        age_category = detector.get_age_category(age_days) if age_days else "unknown"

        # Get lifecycle state
        lifecycle = ArtifactLifecycle()
        lifecycle_state = lifecycle.get_current_state(artifact_path)

        # Get version
        version_mgr = VersionManager()
        version = version_mgr.extract_version_from_frontmatter(artifact_path)
        version_str = f"{version.major}.{version.minor}" if version else None

        # Determine warning status
        warning_status = "✅ OK"
        if age_category == "warning":
            warning_status = "⚠️  WARNING (90+ days old)"
        elif age_category == "stale":
            warning_status = "🚨 STALE (180+ days old)"
        elif age_category == "archive":
            warning_status = "📦 ARCHIVE (365+ days old)"

        return ArtifactStatus(
            path=str(artifact_path.relative_to(artifact_path.parents[3])),
            age_days=age_days,
            age_category=age_category,
            lifecycle_state=lifecycle_state,
            version=version_str,
            warning_status=warning_status,
        )
    except Exception as e:
        return ArtifactStatus(
            path=str(artifact_path.relative_to(artifact_path.parents[3])),
            age_days=None,
            age_category="error",
            lifecycle_state="unknown",
            version=None,
            warning_status=f"❌ ERROR: {str(e)[:50]}",
        )  # noqa: E501


def find_all_artifacts(root_dir: Path) -> list[Path]:
    """Find all artifact markdown files."""
    artifacts = []
    dirs = ["design_documents", "assessments", "audits", "bug_reports", "research", "templates"]
    for artifact_dir in dirs:
        target_dir = root_dir / artifact_dir
        if target_dir.exists():
            artifacts.extend(sorted(target_dir.glob("*.md")))
    return artifacts


def print_dashboard(statuses: list[ArtifactStatus]) -> None:
    """Print dashboard view."""
    print("\n" + "=" * 100)
    print("📊 ARTIFACT STATUS DASHBOARD")
    print("=" * 100)

    # Summary statistics
    total = len(statuses)
    ok_count = sum(1 for s in statuses if s.age_category == "ok")
    warning_count = sum(1 for s in statuses if s.age_category == "warning")
    stale_count = sum(1 for s in statuses if s.age_category == "stale")
    archive_count = sum(1 for s in statuses if s.age_category == "archive")
    error_count = sum(1 for s in statuses if s.age_category == "error")

    print("\n📈 SUMMARY:")
    print(f"   Total Artifacts:  {total}")
    pct = ok_count * 100 // total if total else 0
    print(f"   ✅ Healthy:       {ok_count} ({pct}%)")
    pct = warning_count * 100 // total if total else 0
    print(f"   ⚠️  Warning:       {warning_count} ({pct}%)")
    pct = stale_count * 100 // total if total else 0
    print(f"   🚨 Stale:         {stale_count} ({pct}%)")
    pct = archive_count * 100 // total if total else 0
    print(f"   📦 Archive:       {archive_count} ({pct}%)")
    if error_count:
        print(f"   ❌ Errors:        {error_count}")

    # Lifecycle state summary
    states = {}
    for status in statuses:
        states[status.lifecycle_state] = states.get(status.lifecycle_state, 0) + 1

    print("\n🔄 LIFECYCLE STATES:")
    for state, count in sorted(states.items()):
        print(f"   {state.upper()}: {count}")

    # Age distribution
    print("\n📅 AGE DISTRIBUTION:")
    age_ranges = {
        "0-30 days": 0,
        "31-90 days": 0,
        "91-180 days": 0,
        "181-365 days": 0,
        "365+ days": 0,
    }
    for status in statuses:
        if status.age_days is not None:
            if status.age_days <= 30:
                age_ranges["0-30 days"] += 1
            elif status.age_days <= 90:
                age_ranges["31-90 days"] += 1
            elif status.age_days <= 180:
                age_ranges["91-180 days"] += 1
            elif status.age_days <= 365:
                age_ranges["181-365 days"] += 1
            else:
                age_ranges["365+ days"] += 1

    for range_label, count in age_ranges.items():
        print(f"   {range_label}: {count}")

    # Detailed table (only show items needing attention)
    print("\n⚠️  ITEMS NEEDING ATTENTION:")
    print("-" * 100)

    attention_items = [
        s for s in statuses if s.age_category in ["warning", "stale", "archive"]
    ]
    if attention_items:
        print(f"{'ARTIFACT':<60} {'AGE':<12} {'STATE':<12} {'VERSION':<10}")
        print("-" * 100)
        sorted_items = sorted(attention_items, key=lambda x: x.age_days or 0, reverse=True)
        for status in sorted_items:
            age_str = f"{status.age_days}d" if status.age_days else "unknown"
            print(
                f"{status.path:<60} {age_str:<12} "
                f"{status.lifecycle_state:<12} {status.version or 'N/A':<10}"
            )
    else:
        print("✅ All artifacts are in good health!")

    print("\n" + "=" * 100 + "\n")


def print_compact(statuses: list[ArtifactStatus]) -> None:
    """Print compact table view."""
    print("\n" + "-" * 120)
    print(
        f"{'ARTIFACT':<60} {'AGE':<10} {'CATEGORY':<12} "
        f"{'LIFECYCLE':<12} {'VERSION':<10} {'STATUS':<20}"
    )
    print("-" * 120)

    sorted_statuses = sorted(statuses, key=lambda x: x.age_days or 0, reverse=True)
    for status in sorted_statuses:
        age_str = f"{status.age_days}d" if status.age_days else "N/A"
        print(
            f"{status.path:<60} {age_str:<10} {status.age_category:<12} "
            f"{status.lifecycle_state:<12} {status.version or 'N/A':<10} "
            f"{status.warning_status:<20}"
        )

    print("-" * 120 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Display artifact status dashboard with aging and lifecycle information"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show full dashboard view (default)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Show compact table view",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--aging-only",
        action="store_true",
        help="Show aging information only",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Only show artifacts older than THRESHOLD days (default: 0, show all)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory containing artifacts (default: auto-detect from script location)",
    )

    args = parser.parse_args()

    # Auto-detect artifacts root if not provided
    if args.root is None:
        # Try to find docs/artifacts from current location
        script_path = Path(__file__)
        artifacts_root = script_path.parent.parent.parent.parent / "docs" / "artifacts"

        # Fallback: try relative to current directory
        if not artifacts_root.exists():
            artifacts_root = Path.cwd() / "docs" / "artifacts"

        # Fallback: try one level up
        if not artifacts_root.exists():
            artifacts_root = Path.cwd().parent / "docs" / "artifacts"

        args.root = artifacts_root

    # Find artifacts
    if not args.root.exists():
        print(f"❌ Artifacts directory not found: {args.root}")
        sys.exit(1)

    artifacts = find_all_artifacts(args.root)
    if not artifacts:
        print(f"⚠️  No artifacts found in {args.root}")
        return

    # Get status for each artifact
    statuses = [get_artifact_status(artifact) for artifact in artifacts]

    # Filter by threshold
    if args.threshold > 0:
        statuses = [s for s in statuses if s.age_days and s.age_days >= args.threshold]

    # Output
    if args.json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "total": len(statuses),
            "artifacts": [asdict(s) for s in statuses],
        }
        print(json.dumps(output, indent=2, default=str))
    elif args.aging_only:
        print(f"\n{'ARTIFACT':<60} {'AGE (DAYS)':<12} {'CATEGORY':<12}")
        print("-" * 84)
        sorted_statuses = sorted(statuses, key=lambda x: x.age_days or 0, reverse=True)
        for status in sorted_statuses:
            age_str = f"{status.age_days}" if status.age_days else "N/A"
            print(f"{status.path:<60} {age_str:<12} {status.age_category:<12}")
        print()
    elif args.compact:
        print_compact(statuses)
    else:
        # Default: dashboard view
        print_dashboard(statuses)


if __name__ == "__main__":
    main()
