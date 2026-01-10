#!/usr/bin/env python3
"""
Context Bundling Maintenance and Control System

Provides controls to:
- Enable/disable context-bundling system globally
- Configure bundle behavior for maintenance windows
- Collect AI feedback on context quality
- Manage stale content and refactoring states
- Track context usage patterns

Relationship:
- `context_control.py` manages global state and policies (enable/disable, maintenance windows,
  feedback ingestion). It does not compute memory footprint.
- `context_inspector.py` provides observability (what bundles are loaded, memory footprint,
  stale detection, feedback reporting) and does not change system state.

Usage:
    python context_control.py --disable --reason "extensive refactor in progress"
    python context_control.py --status
    python context_control.py --enable
    python context_control.py --feedback collect bundle-name relevance-score feedback-text
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from AgentQMS.tools.utils.paths import get_project_root
from AgentQMS.tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


class ContextBundleState(str, Enum):
    """Context bundling system states."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


@dataclass
class SystemControl:
    """Control state for context-bundling system."""
    state: str  # enabled, disabled, maintenance, degraded
    timestamp: str
    reason: str | None = None
    disabled_bundles: list[str] | None = None
    maintenance_window_until: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BundleConfig:
    """Configuration for individual bundle behavior."""
    bundle_name: str
    enabled: bool
    tier_level: int  # 1=essential only, 2=essential+preferred, 3=all
    max_memory_mb: float | None = None
    cache_duration_hours: int = 24
    refresh_on_file_change: bool = True
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ContextControl:
    """Manages context-bundling system state and controls."""

    def __init__(self, project_root: Path | None = None):
        """Initialize context control system.

        Args:
            project_root: Project root path (auto-detected if not provided)
        """
        if project_root is None:
            project_root = get_project_root()

        self.project_root = Path(project_root)
        self._control_dir = self.project_root / ".agentqms" / "context_control"
        self._control_dir.mkdir(parents=True, exist_ok=True)

        self._state_file = self._control_dir / "system_state.json"
        self._bundle_configs = self._control_dir / "bundle_configs.json"
        self._usage_log = self._control_dir / "usage_log.jsonl"

        # Initialize state file if it doesn't exist
        if not self._state_file.exists():
            self._write_state(SystemControl(
                state=ContextBundleState.ENABLED.value,
                timestamp=datetime.now().isoformat(),
                reason=None
            ))

    def is_enabled(self) -> bool:
        """Check if context-bundling is currently enabled.

        Returns:
            True if system is enabled, False otherwise
        """
        state = self.get_system_state()
        return state.state == ContextBundleState.ENABLED.value

    def get_system_state(self) -> SystemControl:
        """Get current system control state.

        Returns:
            Current SystemControl state
        """
        if self._state_file.exists():
            data = json.loads(self._state_file.read_text())
            return SystemControl(**data)

        return SystemControl(
            state=ContextBundleState.ENABLED.value,
            timestamp=datetime.now().isoformat()
        )

    def _write_state(self, control: SystemControl) -> None:
        """Write system state to file."""
        self._state_file.write_text(json.dumps(control.to_dict(), indent=2))

    def disable_context_bundling(self, reason: str = "",
                                duration_hours: int | None = None,
                                disabled_bundles: list[str] | None = None) -> SystemControl:
        """Disable context-bundling system.

        Args:
            reason: Reason for disabling (e.g., "extensive refactor in progress")
            duration_hours: Auto-enable after N hours (None = manual re-enable)
            disabled_bundles: Specific bundles to disable (None = all)

        Returns:
            New SystemControl state
        """
        maint_until = None
        if duration_hours:
            maint_until = (datetime.now() + timedelta(hours=duration_hours)).isoformat()

        state = SystemControl(
            state=ContextBundleState.MAINTENANCE.value if duration_hours else ContextBundleState.DISABLED.value,
            timestamp=datetime.now().isoformat(),
            reason=reason,
            disabled_bundles=disabled_bundles,
            maintenance_window_until=maint_until,
            notes=f"Disabled at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self._write_state(state)
        self._log_usage("disable", reason, duration_hours)

        return state

    def enable_context_bundling(self) -> SystemControl:
        """Enable context-bundling system.

        Returns:
            New SystemControl state
        """
        state = SystemControl(
            state=ContextBundleState.ENABLED.value,
            timestamp=datetime.now().isoformat(),
            reason=None,
            notes=f"Re-enabled at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self._write_state(state)
        self._log_usage("enable", "System re-enabled", None)

        return state

    def set_degraded_mode(self, reason: str, affected_bundles: list[str]) -> SystemControl:
        """Set system to degraded mode (some bundles unavailable).

        Args:
            reason: Reason for degraded state
            affected_bundles: List of bundles that are degraded

        Returns:
            New SystemControl state
        """
        state = SystemControl(
            state=ContextBundleState.DEGRADED.value,
            timestamp=datetime.now().isoformat(),
            reason=reason,
            disabled_bundles=affected_bundles,
            notes="System operating in degraded mode"
        )

        self._write_state(state)
        self._log_usage("degraded", reason, None)

        return state

    def configure_bundle(self, bundle_name: str, enabled: bool = True,
                        tier_level: int = 3, max_memory_mb: float | None = None,
                        cache_duration_hours: int = 24) -> BundleConfig:
        """Configure individual bundle behavior.

        Args:
            bundle_name: Name of the bundle
            enabled: Whether bundle is enabled
            tier_level: Max tier to load (1, 2, or 3)
            max_memory_mb: Maximum memory this bundle can use
            cache_duration_hours: How long to cache bundle content

        Returns:
            New BundleConfig
        """
        configs = self._load_bundle_configs()

        config = BundleConfig(
            bundle_name=bundle_name,
            enabled=enabled,
            tier_level=max(1, min(3, tier_level)),
            max_memory_mb=max_memory_mb,
            cache_duration_hours=cache_duration_hours
        )

        configs[bundle_name] = asdict(config)
        self._bundle_configs.write_text(json.dumps(configs, indent=2))

        return config

    def get_bundle_config(self, bundle_name: str) -> BundleConfig | None:
        """Get configuration for a specific bundle.

        Args:
            bundle_name: Name of the bundle

        Returns:
            BundleConfig if found, None otherwise
        """
        configs = self._load_bundle_configs()
        if bundle_name in configs:
            return BundleConfig(**configs[bundle_name])
        return None

    def _load_bundle_configs(self) -> dict[str, dict]:
        """Load all bundle configurations."""
        if self._bundle_configs.exists():
            return json.loads(self._bundle_configs.read_text())
        return {}

    def _log_usage(self, action: str, details: str, duration_hours: int | None) -> None:
        """Log context-bundling usage event.

        Args:
            action: Action taken (disable, enable, degraded, etc.)
            details: Details about the action
            duration_hours: Duration if applicable
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "duration_hours": duration_hours,
        }

        with open(self._usage_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_usage_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get usage history for the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of usage log entries
        """
        if not self._usage_log.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        history = []

        with open(self._usage_log) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if datetime.fromisoformat(entry["timestamp"]) >= cutoff:
                        history.append(entry)
                except (json.JSONDecodeError, ValueError):
                    pass

        return history

    def auto_check_maintenance_window(self) -> bool:
        """Check if maintenance window has expired and auto-enable if so.

        Returns:
            True if system was re-enabled, False otherwise
        """
        state = self.get_system_state()

        if state.state != ContextBundleState.MAINTENANCE.value:
            return False

        if not state.maintenance_window_until:
            return False

        if datetime.fromisoformat(state.maintenance_window_until) <= datetime.now():
            self.enable_context_bundling()
            return True

        return False


class ContextFeedbackCollector:
    """Collects and analyzes AI feedback on context quality."""

    def __init__(self, project_root: Path | None = None):
        """Initialize feedback collector.

        Args:
            project_root: Project root path (auto-detected if not provided)
        """
        if project_root is None:
            project_root = get_project_root()

        self.project_root = Path(project_root)
        self._feedback_dir = self.project_root / ".agentqms" / "context_feedback"
        self._feedback_dir.mkdir(parents=True, exist_ok=True)

    def submit_feedback(self, bundle_name: str, task_description: str,
                       relevance_score: int, token_count: int,
                       feedback_text: str, improvements: list[str] | None = None) -> Path:
        """Submit feedback on bundle relevance and quality.

        Args:
            bundle_name: Name of the bundle being evaluated
            task_description: Description of the task
            relevance_score: 1-10 relevance score (1=irrelevant, 10=perfect)
            token_count: Tokens used from this bundle
            feedback_text: Qualitative feedback
            improvements: Suggested improvements

        Returns:
            Path to saved feedback file
        """
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "bundle": bundle_name,
            "task": task_description,
            "relevance_score": max(1, min(10, relevance_score)),
            "token_count": token_count,
            "feedback": feedback_text,
            "improvements": improvements or []
        }

        filename = f"{bundle_name}_{datetime.now().isoformat().replace(':', '-')}.json"
        feedback_file = self._feedback_dir / filename
        feedback_file.write_text(json.dumps(feedback_data, indent=2))

        return feedback_file

    def get_feedback_analytics(self, bundle_name: str | None = None,
                             days: int = 30) -> dict[str, Any]:
        """Analyze feedback data.

        Args:
            bundle_name: If provided, analyze only this bundle
            days: Look back N days

        Returns:
            Dictionary with feedback analysis
        """
        cutoff = datetime.now() - timedelta(days=days)
        feedback_files = list(self._feedback_dir.glob("*.json"))

        bundle_data: dict[str, dict[str, Any]] = {}
        overall_scores: list[float] = []

        for fb_file in feedback_files:
            try:
                data = json.loads(fb_file.read_text())

                if datetime.fromisoformat(data["timestamp"]) < cutoff:
                    continue

                bundle = data["bundle"]
                if bundle_name and bundle != bundle_name:
                    continue

                if bundle not in bundle_data:
                    bundle_data[bundle] = {
                        "count": 0,
                        "scores": [],
                        "tokens_used": 0,
                        "issues": [],
                        "improvements": []
                    }

                score = data["relevance_score"]
                bundle_data[bundle]["count"] += 1
                bundle_data[bundle]["scores"].append(score)
                bundle_data[bundle]["tokens_used"] += data.get("token_count", 0)
                overall_scores.append(score)

                if score < 6:
                    bundle_data[bundle]["issues"].append({
                        "score": score,
                        "feedback": data["feedback"],
                        "task": data.get("task", "")
                    })

                bundle_data[bundle]["improvements"].extend(data.get("improvements", []))

            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Calculate statistics
        analytics: dict[str, Any] = {
            "period_days": days,
            "bundles_analyzed": len(bundle_data),
            "total_feedback_points": sum(d["count"] for d in bundle_data.values()),
            "overall_avg_relevance": sum(overall_scores) / len(overall_scores) if overall_scores else 0,
            "bundles": {}
        }

        for bundle, data in bundle_data.items():
            scores = data["scores"]
            analytics["bundles"][bundle] = {
                "feedback_count": data["count"],
                "avg_relevance_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "score_distribution": {
                    "low": sum(1 for s in scores if s <= 3),
                    "medium": sum(1 for s in scores if 4 <= s <= 6),
                    "high": sum(1 for s in scores if s >= 7)
                },
                "total_tokens_used": data["tokens_used"],
                "avg_tokens_per_use": data["tokens_used"] / data["count"] if data["count"] > 0 else 0,
                "issues": len(data["issues"]),
                "recent_issues": data["issues"][-3:] if data["issues"] else [],
                "improvement_suggestions": list(set(data["improvements"]))[:5]
            }

        return analytics


def print_system_state(state: SystemControl) -> None:
    """Pretty-print system state."""
    print(f"\n{'='*70}")
    print("🔧 Context Bundling System Status")
    print(f"{'='*70}")

    state_emoji = {
        "enabled": "✅",
        "disabled": "🛑",
        "maintenance": "🔧",
        "degraded": "⚠️"
    }

    emoji = state_emoji.get(state.state, "❓")
    print(f"State:      {emoji} {state.state.upper()}")
    print(f"Timestamp:  {state.timestamp}")

    if state.reason:
        print(f"Reason:     {state.reason}")

    if state.maintenance_window_until:
        until = datetime.fromisoformat(state.maintenance_window_until)
        if until > datetime.now():
            remaining = (until - datetime.now()).total_seconds() / 3600
            print(f"Maintenance until: {state.maintenance_window_until} ({remaining:.1f}h remaining)")

    if state.disabled_bundles:
        print("Disabled bundles:")
        for bundle in state.disabled_bundles:
            print(f"  • {bundle}")

    if state.notes:
        print(f"Notes:      {state.notes}")
    print()


def main() -> int:
    """Main entry point for context control CLI."""
    parser = argparse.ArgumentParser(
        description="Control and manage context-bundling system"
    )

    # System control
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--enable", action="store_true", help="Enable context-bundling")
    parser.add_argument("--disable", action="store_true", help="Disable context-bundling")
    parser.add_argument("--reason", help="Reason for enable/disable")
    parser.add_argument("--duration", type=int, help="Auto-enable after N hours (for --disable)")
    parser.add_argument("--maintenance", action="store_true", help="Enable maintenance mode with duration")

    # Bundle configuration
    parser.add_argument("--configure-bundle", help="Configure specific bundle")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="Max tier level for bundle")
    parser.add_argument("--max-memory", type=float, help="Max memory for bundle (MB)")

    # Feedback
    parser.add_argument("--feedback", nargs=4,
                       metavar=("BUNDLE", "SCORE", "TOKENS", "FEEDBACK"),
                       help="Submit feedback: bundle name, relevance score (1-10), tokens used, feedback text")
    parser.add_argument("--feedback-analytics", help="Analyze feedback (bundle name or 'all')")
    parser.add_argument("--feedback-days", type=int, default=30, help="Days to analyze")

    # History
    parser.add_argument("--history", action="store_true", help="Show usage history")
    parser.add_argument("--history-days", type=int, default=30, help="Days to show")

    args = parser.parse_args()

    try:
        control = ContextControl()

        # Handle auto-check for maintenance window
        control.auto_check_maintenance_window()

        if args.status:
            state = control.get_system_state()
            print_system_state(state)
            return 0

        if args.enable:
            state = control.enable_context_bundling()
            print_system_state(state)
            return 0

        if args.disable or args.maintenance:
            reason = args.reason or "No reason provided"
            duration = args.duration if (args.maintenance or args.duration) else None
            state = control.disable_context_bundling(reason, duration)
            print_system_state(state)
            return 0

        if args.configure_bundle:
            config = control.configure_bundle(
                args.configure_bundle,
                tier_level=args.tier or 3,
                max_memory_mb=args.max_memory
            )
            print(f"\n✅ Configured bundle: {config.bundle_name}")
            print(f"   Enabled:  {config.enabled}")
            print(f"   Tier:     {config.tier_level}/3")
            if config.max_memory_mb:
                print(f"   Max mem:  {config.max_memory_mb} MB")
            print()
            return 0

        if args.feedback:
            bundle, score_str, tokens_str, feedback = args.feedback
            try:
                score = int(score_str)
                tokens = int(tokens_str)
                feedback_file = ContextFeedbackCollector().submit_feedback(
                    bundle, "CLI feedback", score, tokens, feedback
                )
                print(f"✅ Feedback recorded: {feedback_file.name}\n")
            except ValueError as e:
                print(f"❌ Invalid input: {e}\n", file=sys.stderr)
                return 1
            return 0

        if args.feedback_analytics:
            collector = ContextFeedbackCollector()
            bundle_name = None if args.feedback_analytics == "all" else args.feedback_analytics
            analytics = collector.get_feedback_analytics(bundle_name, args.feedback_days)

            print(f"\n📊 Feedback Analytics (last {args.feedback_days} days)")
            print("=" * 70)
            print(f"Total feedback points: {analytics['total_feedback_points']}")
            print(f"Overall avg relevance: {analytics['overall_avg_relevance']:.1f}/10")

            for bundle, stats in analytics["bundles"].items():
                print(f"\n  📦 {bundle}:")
                print(f"     Feedback count:  {stats['feedback_count']}")
                print(f"     Avg relevance:   {stats['avg_relevance_score']:.1f}/10")
                print(f"     Score range:     {stats['min_score']}-{stats['max_score']}")
                print(f"     Low/Med/High:    {stats['score_distribution']['low']}/{stats['score_distribution']['medium']}/{stats['score_distribution']['high']}")
                print(f"     Tokens used:     {stats['total_tokens_used']:,} ({stats['avg_tokens_per_use']:.0f} avg)")
                if stats["recent_issues"]:
                    print(f"     ⚠️  Issues: {len(stats['issues'])} reported")
            print()
            return 0

        if args.history:
            history = control.get_usage_history(args.history_days)
            if not history:
                print(f"\n📭 No usage history in last {args.history_days} days\n")
            else:
                print(f"\n📜 Usage History (last {args.history_days} days)")
                print("=" * 70)
                for entry in reversed(history):
                    ts = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M")
                    print(f"  {ts}: {entry['action']} - {entry['details']}")
                    if entry.get("duration_hours"):
                        print(f"       Duration: {entry['duration_hours']}h")
                print()
            return 0

        # Default: show status
        state = control.get_system_state()
        print_system_state(state)
        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
