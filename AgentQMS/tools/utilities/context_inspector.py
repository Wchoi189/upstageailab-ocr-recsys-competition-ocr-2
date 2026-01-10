#!/usr/bin/env python3
"""
Context Inspector for Observability and Debugging

Provides tools to inspect loaded context bundles, analyze memory footprint,
detect stale content, and collect AI feedback on context quality.

Key Features:
- Visualize exact context being loaded into AI memory
- Calculate precise memory footprint per bundle and tier
- Detect stale files and outdated documentation
- Track AI feedback on context quality and relevance
- Report system status; does not modify global enable/disable state

Relationship:
- Pairs with `context_control.py`: this module is read-only and focuses on measurements
    (inventory, memory, staleness, feedback summaries). `context_control.py` is responsible
    for state changes (enable/disable, maintenance window, feedback ingestion).

Usage:
        python context_inspector.py --inspect ocr-text-detection
        python context_inspector.py --memory --output json
        python context_inspector.py --stale --threshold 7d
        python context_inspector.py --feedback --recent
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from AgentQMS.tools.core.plugins import get_plugin_registry
from AgentQMS.tools.utils.paths import get_project_root
from AgentQMS.tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


@dataclass
class FileInfo:
    """Information about a context file."""
    path: str
    tier: str
    size_bytes: int
    modified_days_ago: float
    is_stale: bool

    def size_kb(self) -> float:
        """Return size in kilobytes."""
        return self.size_bytes / 1024

    def size_mb(self) -> float:
        """Return size in megabytes."""
        return self.size_bytes / (1024 * 1024)


@dataclass
class BundleMetrics:
    """Memory and quality metrics for a bundle."""
    name: str
    description: str
    tier1_size_kb: float
    tier2_size_kb: float
    tier3_size_kb: float
    total_size_kb: float
    file_count: int
    stale_file_count: int
    average_file_age_days: float

    @property
    def total_size_mb(self) -> float:
        return self.total_size_kb / 1024

    @property
    def is_healthy(self) -> bool:
        """Bundle is healthy if <10% of files are stale."""
        if self.file_count == 0:
            return True
        return (self.stale_file_count / self.file_count) < 0.1


@dataclass
class ContextSnapshot:
    """Full context load snapshot for debugging."""
    timestamp: str
    enabled: bool
    bundles: list[str]
    total_context_mb: float
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ContextInspector:
    """Inspects and analyzes loaded context bundles."""

    STALE_THRESHOLD_DAYS = 7
    TOKENS_PER_MB = 200000  # Rough estimate for LLM context

    def __init__(self, project_root: Path | None = None):
        """Initialize the context inspector.

        Args:
            project_root: Project root path (auto-detected if not provided)
        """
        if project_root is None:
            project_root = get_project_root()

        self.project_root = Path(project_root)
        self.registry = get_plugin_registry()
        self.bundles = self.registry.get_context_bundles()
        self._state_file = self.project_root / ".agentqms" / "context_state.json"
        self._feedback_dir = self.project_root / ".agentqms" / "context_feedback"
        self._feedback_dir.mkdir(parents=True, exist_ok=True)

    def inspect_bundle(self, bundle_name: str, verbose: bool = False) -> BundleMetrics:
        """Inspect a single bundle and return metrics.

        Args:
            bundle_name: Name of the bundle to inspect
            verbose: If True, print detailed file list

        Returns:
            BundleMetrics with bundle statistics

        Raises:
            ValueError: If bundle not found
        """
        if bundle_name not in self.bundles:
            raise ValueError(f"Bundle '{bundle_name}' not found")

        bundle = self.bundles[bundle_name]
        now = datetime.now()

        tier_files: dict[str, list[Any]] = {"tier1": [], "tier2": [], "tier3": []}
        total_size = 0
        stale_count = 0
        ages = []

        # Collect files from all tiers
        for tier_name in ["tier1", "tier2", "tier3"]:
            if tier_name not in bundle:
                continue

            for file_pattern in bundle[tier_name]:
                files = self._expand_glob(file_pattern)
                for filepath in files:
                    try:
                        stat = filepath.stat()
                        size = stat.st_size
                        modified = datetime.fromtimestamp(stat.st_mtime)
                        days_ago = (now - modified).days
                        is_stale = days_ago > self.STALE_THRESHOLD_DAYS

                        total_size += size
                        if is_stale:
                            stale_count += 1
                        ages.append(days_ago)

                        file_info = FileInfo(
                            path=str(filepath.relative_to(self.project_root)),
                            tier=tier_name,
                            size_bytes=size,
                            modified_days_ago=days_ago,
                            is_stale=is_stale
                        )
                        tier_files[tier_name].append(file_info)

                        if verbose:
                            status = "⚠️ STALE" if is_stale else "✓"
                            print(f"  {status} {tier_name.upper()}: {file_info.path} ({file_info.size_kb():.1f} KB, {days_ago}d)")
                    except (OSError, FileNotFoundError):
                        pass

        # Calculate metrics
        total_files = sum(len(files) for files in tier_files.values())
        avg_age = sum(ages) / len(ages) if ages else 0

        tier_sizes = {
            "tier1": sum(f.size_bytes for f in tier_files["tier1"]) / 1024,
            "tier2": sum(f.size_bytes for f in tier_files["tier2"]) / 1024,
            "tier3": sum(f.size_bytes for f in tier_files["tier3"]) / 1024,
        }

        return BundleMetrics(
            name=bundle_name,
            description=bundle.get("description", ""),
            tier1_size_kb=tier_sizes["tier1"],
            tier2_size_kb=tier_sizes["tier2"],
            tier3_size_kb=tier_sizes["tier3"],
            total_size_kb=total_size / 1024,
            file_count=total_files,
            stale_file_count=stale_count,
            average_file_age_days=avg_age
        )

    def inspect_all_bundles(self, verbose: bool = False) -> dict[str, BundleMetrics]:
        """Inspect all bundles and return metrics.

        Args:
            verbose: If True, print detailed information

        Returns:
            Dictionary of bundle metrics indexed by bundle name
        """
        metrics = {}
        for bundle_name in self.bundles:
            try:
                metrics[bundle_name] = self.inspect_bundle(bundle_name, verbose)
            except ValueError:
                continue
        return metrics

    def get_memory_footprint(self, bundles_to_load: list[str] | None = None) -> dict[str, Any]:
        """Calculate memory footprint for a set of bundles.

        Args:
            bundles_to_load: List of bundle names. If None, inspect all.

        Returns:
            Dictionary with memory analysis
        """
        if bundles_to_load is None:
            bundles_to_load = list(self.bundles.keys())

        metrics = self.inspect_all_bundles()

        total_tier1_kb = 0.0
        total_tier2_kb = 0.0
        total_tier3_kb = 0.0
        total_files = 0
        unhealthy_bundles = []

        for bundle_name in bundles_to_load:
            if bundle_name not in metrics:
                continue
            m = metrics[bundle_name]
            total_tier1_kb += m.tier1_size_kb
            total_tier2_kb += m.tier2_size_kb
            total_tier3_kb += m.tier3_size_kb
            total_files += m.file_count

            if not m.is_healthy:
                unhealthy_bundles.append((bundle_name, m.stale_file_count, m.file_count))

        # Tier1 is always loaded; tier2/3 are optional
        loaded_tier1_mb = total_tier1_kb / 1024
        optional_tier23_mb = (total_tier2_kb + total_tier3_kb) / 1024

        return {
            "bundles": bundles_to_load,
            "summary": {
                "tier1_mb": round(loaded_tier1_mb, 2),
                "tier2_mb": round(total_tier2_kb / 1024, 2),
                "tier3_mb": round(total_tier3_kb / 1024, 2),
                "total_mb": round(loaded_tier1_mb + optional_tier23_mb, 2),
                "total_files": total_files,
                "estimated_tokens": int((loaded_tier1_mb + optional_tier23_mb) * self.TOKENS_PER_MB),
            },
            "tier_breakdown": {
                "tier1": round(total_tier1_kb / 1024, 2),
                "tier2": round(total_tier2_kb / 1024, 2),
                "tier3": round(total_tier3_kb / 1024, 2),
            },
            "health": {
                "healthy_bundles": len(bundles_to_load) - len(unhealthy_bundles),
                "unhealthy_bundles": [
                    {"name": name, "stale": stale, "total": total}
                    for name, stale, total in unhealthy_bundles
                ]
            }
        }

    def get_stale_files(self, days_threshold: int | None = None) -> dict[str, list[FileInfo]]:
        """Find stale files across all bundles.

        Args:
            days_threshold: Consider files older than this many days as stale

        Returns:
            Dictionary mapping bundle names to list of stale files
        """
        if days_threshold is None:
            days_threshold = self.STALE_THRESHOLD_DAYS

        now = datetime.now()
        stale_by_bundle = {}

        for bundle_name in self.bundles:
            bundle = self.bundles[bundle_name]
            stale_files = []

            for tier_name in ["tier1", "tier2", "tier3"]:
                if tier_name not in bundle:
                    continue

                for file_pattern in bundle[tier_name]:
                    files = self._expand_glob(file_pattern)
                    for filepath in files:
                        try:
                            stat = filepath.stat()
                            modified = datetime.fromtimestamp(stat.st_mtime)
                            days_ago = (now - modified).days

                            if days_ago > days_threshold:
                                stale_files.append(FileInfo(
                                    path=str(filepath.relative_to(self.project_root)),
                                    tier=tier_name,
                                    size_bytes=stat.st_size,
                                    modified_days_ago=days_ago,
                                    is_stale=True
                                ))
                        except (OSError, FileNotFoundError):
                            pass

            if stale_files:
                stale_by_bundle[bundle_name] = sorted(
                    stale_files,
                    key=lambda f: f.modified_days_ago,
                    reverse=True
                )

        return stale_by_bundle

    def save_context_snapshot(self, bundles: list[str] | None = None, enabled: bool = True) -> Path:
        """Save a snapshot of loaded context for debugging.

        Args:
            bundles: List of loaded bundles. If None, all are assumed.
            enabled: Whether context-bundling is enabled

        Returns:
            Path to saved snapshot
        """
        if bundles is None:
            bundles = list(self.bundles.keys())

        footprint = self.get_memory_footprint(bundles)

        snapshot = ContextSnapshot(
            timestamp=datetime.now().isoformat(),
            enabled=enabled,
            bundles=bundles,
            total_context_mb=footprint["summary"]["total_mb"],
            token_estimate=footprint["summary"]["estimated_tokens"]
        )

        snapshot_file = self._state_file.parent / f"snapshot_{snapshot.timestamp.replace(':', '-')}.json"
        snapshot_file.write_text(json.dumps(snapshot.to_dict(), indent=2))

        return snapshot_file

    def collect_feedback(self, bundle_name: str, feedback: str,
                        relevance_score: int, token_count: int) -> Path:
        """Collect AI feedback on context quality.

        Args:
            bundle_name: Name of the bundle being evaluated
            feedback: Qualitative feedback text
            relevance_score: 1-10 relevance score (1=irrelevant, 10=perfect)
            token_count: Approximate tokens used

        Returns:
            Path to saved feedback
        """
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "bundle": bundle_name,
            "relevance_score": relevance_score,
            "token_count": token_count,
            "feedback": feedback,
        }

        feedback_file = self._feedback_dir / f"{bundle_name}_{datetime.now().isoformat().replace(':', '-')}.json"
        feedback_file.write_text(json.dumps(feedback_data, indent=2))

        return feedback_file

    def get_feedback_summary(self, bundle_name: str | None = None, days: int = 7) -> dict[str, Any]:
        """Get summary of feedback for bundles.

        Args:
            bundle_name: If provided, only get feedback for this bundle
            days: Only include feedback from the last N days

        Returns:
            Dictionary with feedback analysis
        """
        cutoff = datetime.now() - timedelta(days=days)
        feedback_files = list(self._feedback_dir.glob("*.json"))

        summaries: dict[str, dict[str, Any]] = {}
        for fb_file in feedback_files:
            try:
                data = json.loads(fb_file.read_text())
                if datetime.fromisoformat(data["timestamp"]) < cutoff:
                    continue
                if bundle_name and data["bundle"] != bundle_name:
                    continue

                bundle = data["bundle"]
                if bundle not in summaries:
                    summaries[bundle] = {
                        "count": 0,
                        "avg_relevance": 0,
                        "total_tokens": 0,
                        "issues": []
                    }

                summaries[bundle]["count"] += 1
                summaries[bundle]["total_tokens"] += data.get("token_count", 0)
                summaries[bundle]["avg_relevance"] = (
                    (summaries[bundle]["avg_relevance"] * (summaries[bundle]["count"] - 1) + data["relevance_score"])
                    / summaries[bundle]["count"]
                )

                if data["relevance_score"] < 6:
                    summaries[bundle]["issues"].append({
                        "score": data["relevance_score"],
                        "feedback": data["feedback"]
                    })
            except (json.JSONDecodeError, KeyError):
                pass

        return summaries

    def _expand_glob(self, pattern: str) -> list[Path]:
        """Expand glob pattern to actual files."""
        pattern_path = self.project_root / pattern
        try:
            if "*" in pattern:
                return list(self.project_root.glob(pattern))
            else:
                if pattern_path.exists():
                    return [pattern_path]
                return []
        except (OSError, ValueError):
            return []


def print_bundle_metrics(metrics: BundleMetrics) -> None:
    """Pretty-print bundle metrics."""
    print(f"\n{'='*70}")
    print(f"📦 Bundle: {metrics.name}")
    print(f"{'='*70}")
    print(f"Description: {metrics.description}")
    print("\n📊 Size Breakdown:")
    print(f"  Tier 1 (essential):    {metrics.tier1_size_kb:8.1f} KB")
    print(f"  Tier 2 (preferred):    {metrics.tier2_size_kb:8.1f} KB")
    print(f"  Tier 3 (optional):     {metrics.tier3_size_kb:8.1f} KB")
    print("  ─────────────────────────────────")
    print(f"  TOTAL:                 {metrics.total_size_kb:8.1f} KB ({metrics.total_size_mb:.2f} MB)")

    print(f"\n📁 Files: {metrics.file_count} files")
    if metrics.stale_file_count > 0:
        staleness = (metrics.stale_file_count / metrics.file_count) * 100
        print(f"  ⚠️  Stale (>7d old):     {metrics.stale_file_count} files ({staleness:.1f}%)")
    print(f"  ⏱️  Average age:       {metrics.average_file_age_days:.1f} days")

    status = "✅ HEALTHY" if metrics.is_healthy else "⚠️  NEEDS REVIEW"
    print(f"\n🏥 Health: {status}")


def main() -> int:
    """Main entry point for context inspector CLI."""
    parser = argparse.ArgumentParser(
        description="Inspect and analyze context bundles for observability"
    )
    parser.add_argument("--inspect", help="Inspect a specific bundle")
    parser.add_argument("--memory", action="store_true", help="Analyze memory footprint")
    parser.add_argument("--stale", action="store_true", help="Find stale files")
    parser.add_argument("--feedback", action="store_true", help="Show AI feedback summary")
    parser.add_argument("--threshold", type=int, default=7, help="Stale file threshold (days)")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", action="store_true", help="List all bundles")

    args = parser.parse_args()

    inspector = ContextInspector()

    try:
        if args.list:
            print("\n📚 Available Context Bundles:")
            print("=" * 70)
            for name in sorted(inspector.bundles.keys()):
                bundle = inspector.bundles[name]
                desc = bundle.get("description", "")[:50]
                print(f"  • {name:35} {desc}...")
            print()
            return 0

        if args.inspect:
            metrics = inspector.inspect_bundle(args.inspect, verbose=args.verbose)
            if args.output == "json":
                print(json.dumps(asdict(metrics), indent=2))
            else:
                print_bundle_metrics(metrics)
            return 0

        if args.memory:
            footprint = inspector.get_memory_footprint()
            if args.output == "json":
                print(json.dumps(footprint, indent=2))
            else:
                print("\n💾 Context Memory Footprint Analysis")
                print("=" * 70)
                print(f"Bundles loaded: {len(footprint['bundles'])}")
                print("\n📊 Size Summary:")
                print(f"  Tier 1 (essential): {footprint['summary']['tier1_mb']:6.2f} MB")
                print(f"  Tier 2 (optional):  {footprint['summary']['tier2_mb']:6.2f} MB")
                print(f"  Tier 3 (optional):  {footprint['summary']['tier3_mb']:6.2f} MB")
                print("  ────────────────────────────────")
                print(f"  TOTAL:              {footprint['summary']['total_mb']:6.2f} MB")
                print(f"  📝 Token estimate:  ~{footprint['summary']['estimated_tokens']:,} tokens")

                if footprint["health"]["unhealthy_bundles"]:
                    print("\n⚠️  Unhealthy bundles (>10% stale):")
                    for bundle in footprint["health"]["unhealthy_bundles"]:
                        print(f"    • {bundle['name']}: {bundle['stale']}/{bundle['total']} stale")
                print()
            return 0

        if args.stale:
            stale = inspector.get_stale_files(args.threshold)
            if args.output == "json":
                stale_data = {
                    bundle: [asdict(f) for f in files]
                    for bundle, files in stale.items()
                }
                print(json.dumps(stale_data, indent=2))
            else:
                if not stale:
                    print(f"\n✅ No stale files found (threshold: {args.threshold}d)\n")
                else:
                    print(f"\n⚠️  Stale Files (>{ args.threshold}d old):")
                    print("=" * 70)
                    for bundle, files in stale.items():
                        print(f"\n  📦 {bundle}:")
                        for f in files[:5]:  # Show top 5
                            print(f"    • {f.path} ({int(f.modified_days_ago)}d old)")
                        if len(files) > 5:
                            print(f"    ... and {len(files) - 5} more")
                print()
            return 0

        if args.feedback:
            summary = inspector.get_feedback_summary(days=args.threshold)
            if args.output == "json":
                print(json.dumps(summary, indent=2))
            else:
                if not summary:
                    print(f"\n📭 No feedback found in last {args.threshold} days\n")
                else:
                    print(f"\n📊 AI Feedback Summary (last {args.threshold} days):")
                    print("=" * 70)
                    for bundle, stats in summary.items():
                        print(f"\n  📦 {bundle}:")
                        print(f"    Evaluations:        {stats['count']}")
                        print(f"    Avg relevance:      {stats['avg_relevance']:.1f}/10")
                        print(f"    Total tokens used:  {stats['total_tokens']:,}")
                        if stats['issues']:
                            print(f"    ⚠️  Issues reported: {len(stats['issues'])}")
                            for issue in stats['issues'][:2]:
                                print(f"      • Score {issue['score']}: {issue['feedback'][:60]}...")
                print()
            return 0

        # Default: show all metrics
        metrics = inspector.inspect_all_bundles()
        if args.output == "json":
            print(json.dumps({k: asdict(v) for k, v in metrics.items()}, indent=2))
        else:
            print("\n📚 Context Bundle Inspection Report")
            print("=" * 70)
            for metrics_obj in sorted(metrics.values(), key=lambda m: m.total_size_kb, reverse=True):
                print_bundle_metrics(metrics_obj)
            print()
        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
