#!/usr/bin/env python3
"""
Artifact Monitoring and Compliance System

This script monitors artifact organization compliance and provides alerts
for violations. It can be run as part of CI/CD or as a standalone check.

Usage:
    python monitor_artifacts.py --check
    python monitor_artifacts.py --alert
    python monitor_artifacts.py --report
    python monitor_artifacts.py --fix-suggestions
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


def _load_bootstrap():
    if "scripts._bootstrap" in sys.modules:
        return sys.modules["scripts._bootstrap"]

    current_dir = Path(__file__).resolve().parent
    for directory in (current_dir, *tuple(current_dir.parents)):
        candidate = directory / "_bootstrap.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(
                "scripts._bootstrap", candidate
            )
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


_BOOTSTRAP = _load_bootstrap()
setup_project_paths = _BOOTSTRAP.setup_project_paths

setup_project_paths()

from scripts.agent_tools.compliance.validate_artifacts import ArtifactValidator


class ArtifactMonitor:
    """Monitors artifact organization and compliance."""

    def __init__(self, artifacts_root: str = "docs/artifacts"):
        self.artifacts_root = Path(artifacts_root)
        self.validator = ArtifactValidator(str(artifacts_root))
        self.violations_history_file = Path("artifacts_violations_history.json")

    def check_organization_compliance(self) -> dict:
        """Check overall organization compliance."""
        print("🔍 Checking artifact organization compliance...")

        results = self.validator.validate_all()
        total_files = len(results)
        valid_files = sum(1 for r in results if r["valid"])
        invalid_files = total_files - valid_files

        compliance_rate = (valid_files / total_files * 100) if total_files > 0 else 0

        # Categorize violations
        violation_categories = {"naming": [], "directory": [], "frontmatter": []}

        for result in results:
            if not result["valid"]:
                for error in result["errors"]:
                    if "Naming:" in error:
                        violation_categories["naming"].append(
                            {"file": result["file"], "error": error}
                        )
                    elif "Directory:" in error:
                        violation_categories["directory"].append(
                            {"file": result["file"], "error": error}
                        )
                    elif "Frontmatter:" in error:
                        violation_categories["frontmatter"].append(
                            {"file": result["file"], "error": error}
                        )

        compliance_report = {
            "timestamp": datetime.now().isoformat(),
            "total_files": total_files,
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "compliance_rate": compliance_rate,
            "violation_categories": violation_categories,
            "all_violations": [r for r in results if not r["valid"]],
        }

        return compliance_report

    def generate_compliance_report(self, report: dict) -> str:
        """Generate a human-readable compliance report."""
        lines = []
        lines.append("=" * 60)
        lines.append("ARTIFACT COMPLIANCE REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {report['timestamp']}")
        lines.append(f"Total files: {report['total_files']}")
        lines.append(f"Valid files: {report['valid_files']}")
        lines.append(f"Invalid files: {report['invalid_files']}")
        lines.append(f"Compliance rate: {report['compliance_rate']:.1f}%")
        lines.append("")

        # Overall status
        if report["compliance_rate"] >= 95:
            lines.append("🟢 EXCELLENT: Compliance rate is excellent")
        elif report["compliance_rate"] >= 85:
            lines.append("🟡 GOOD: Compliance rate is good, minor improvements needed")
        elif report["compliance_rate"] >= 70:
            lines.append("🟠 FAIR: Compliance rate is fair, improvements needed")
        else:
            lines.append(
                "🔴 POOR: Compliance rate is poor, significant improvements needed"
            )

        lines.append("")

        # Violation breakdown
        if report["invalid_files"] > 0:
            lines.append("VIOLATION BREAKDOWN:")
            lines.append("-" * 40)

            for category, violations in report["violation_categories"].items():
                if violations:
                    lines.append(
                        f"\n{category.upper()} VIOLATIONS ({len(violations)}):"
                    )
                    for violation in violations[:5]:  # Show first 5
                        lines.append(f"  • {violation['file']}")
                        lines.append(f"    {violation['error']}")

                    if len(violations) > 5:
                        lines.append(f"  ... and {len(violations) - 5} more")

        return "\n".join(lines)

    def check_for_alerts(self, report: dict) -> list[str]:
        """Check for conditions that require alerts."""
        alerts = []

        # Low compliance rate
        if report["compliance_rate"] < 80:
            alerts.append(
                f"🔴 LOW COMPLIANCE: {report['compliance_rate']:.1f}% compliance rate is below 80% threshold"
            )

        # High number of violations
        if report["invalid_files"] > 10:
            alerts.append(
                f"🔴 HIGH VIOLATION COUNT: {report['invalid_files']} files have violations"
            )

        # Specific violation types
        if len(report["violation_categories"]["naming"]) > 5:
            alerts.append(
                f"🟡 NAMING VIOLATIONS: {len(report['violation_categories']['naming'])} files have naming issues"
            )

        if len(report["violation_categories"]["directory"]) > 3:
            alerts.append(
                f"🟡 DIRECTORY VIOLATIONS: {len(report['violation_categories']['directory'])} files are in wrong directories"
            )

        if len(report["violation_categories"]["frontmatter"]) > 5:
            alerts.append(
                f"🟡 FRONTMATTER VIOLATIONS: {len(report['violation_categories']['frontmatter'])} files have frontmatter issues"
            )

        return alerts

    def save_violations_history(self, report: dict) -> None:
        """Save violations history for trend analysis."""
        history = []

        if self.violations_history_file.exists():
            try:
                with open(self.violations_history_file) as f:
                    history = json.load(f)
            except Exception:
                history = []

        # Add current report
        history.append(
            {
                "timestamp": report["timestamp"],
                "compliance_rate": report["compliance_rate"],
                "invalid_files": report["invalid_files"],
                "violation_counts": {
                    "naming": len(report["violation_categories"]["naming"]),
                    "directory": len(report["violation_categories"]["directory"]),
                    "frontmatter": len(report["violation_categories"]["frontmatter"]),
                },
            }
        )

        # Keep only last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        history = [
            entry
            for entry in history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]

        with open(self.violations_history_file, "w") as f:
            json.dump(history, f, indent=2)

    def generate_trend_analysis(self) -> str:
        """Generate trend analysis from violations history."""
        if not self.violations_history_file.exists():
            return "No historical data available for trend analysis."

        try:
            with open(self.violations_history_file) as f:
                history = json.load(f)
        except Exception:
            return "Error reading historical data."

        if len(history) < 2:
            return "Insufficient historical data for trend analysis."

        # Calculate trends
        recent = history[-1]
        previous = history[-2]

        compliance_trend = recent["compliance_rate"] - previous["compliance_rate"]
        violations_trend = recent["invalid_files"] - previous["invalid_files"]

        lines = []
        lines.append("TREND ANALYSIS")
        lines.append("=" * 40)
        lines.append(f"Current compliance: {recent['compliance_rate']:.1f}%")
        lines.append(f"Previous compliance: {previous['compliance_rate']:.1f}%")

        if compliance_trend > 0:
            lines.append(f"📈 Compliance improved by {compliance_trend:.1f}%")
        elif compliance_trend < 0:
            lines.append(f"📉 Compliance declined by {abs(compliance_trend):.1f}%")
        else:
            lines.append("➡️ Compliance unchanged")

        lines.append("")
        lines.append(f"Current violations: {recent['invalid_files']}")
        lines.append(f"Previous violations: {previous['invalid_files']}")

        if violations_trend < 0:
            lines.append(f"📉 Violations decreased by {abs(violations_trend)}")
        elif violations_trend > 0:
            lines.append(f"📈 Violations increased by {violations_trend}")
        else:
            lines.append("➡️ Violations unchanged")

        return "\n".join(lines)

    def generate_fix_suggestions(self, report: dict) -> str:
        """Generate specific fix suggestions for violations."""
        suggestions = []
        suggestions.append("FIX SUGGESTIONS")
        suggestions.append("=" * 40)

        if report["invalid_files"] == 0:
            suggestions.append("✅ No violations found. All artifacts are compliant!")
            return "\n".join(suggestions)

        # Naming violations
        if report["violation_categories"]["naming"]:
            suggestions.append("\n📝 NAMING VIOLATIONS:")
            suggestions.append("Run these commands to fix naming issues:")
            for violation in report["violation_categories"]["naming"][:3]:
                file_path = violation["file"]
                suggestions.append(f"  # Fix: {file_path}")
                suggestions.append(f"  # Check naming convention: {violation['error']}")

        # Directory violations
        if report["violation_categories"]["directory"]:
            suggestions.append("\n📁 DIRECTORY VIOLATIONS:")
            suggestions.append("Move files to correct directories:")
            for violation in report["violation_categories"]["directory"][:3]:
                file_path = violation["file"]
                suggestions.append(f"  # Move: {file_path}")
                suggestions.append(f"  # Issue: {violation['error']}")

        # Frontmatter violations
        if report["violation_categories"]["frontmatter"]:
            suggestions.append("\n📄 FRONTMATTER VIOLATIONS:")
            suggestions.append("Add or fix frontmatter in these files:")
            for violation in report["violation_categories"]["frontmatter"][:3]:
                file_path = violation["file"]
                suggestions.append(f"  # Fix frontmatter: {file_path}")
                suggestions.append(f"  # Issue: {violation['error']}")

        suggestions.append("\n🔧 AUTOMATED FIXES:")
        suggestions.append("  # Update all indexes")
        suggestions.append(
            "  python scripts/agent_tools/update_artifact_indexes.py --all"
        )
        suggestions.append("")
        suggestions.append("  # Validate specific file")
        suggestions.append(
            "  python scripts/agent_tools/validate_artifacts.py --file path/to/file.md"
        )

        return "\n".join(suggestions)

    def run_compliance_check(self) -> bool:
        """Run a complete compliance check and return success status."""
        report = self.check_organization_compliance()
        alerts = self.check_for_alerts(report)

        print(self.generate_compliance_report(report))

        if alerts:
            print("\n🚨 ALERTS:")
            for alert in alerts:
                print(f"  {alert}")

        # Save history
        self.save_violations_history(report)

        # Show trend analysis
        if len(alerts) > 0 or report["compliance_rate"] < 90:
            print(f"\n{self.generate_trend_analysis()}")

        # Return success status
        return report["compliance_rate"] >= 80 and len(alerts) == 0


def main():
    """Main entry point for the artifact monitor."""
    parser = argparse.ArgumentParser(
        description="Monitor artifact organization compliance"
    )
    parser.add_argument("--check", action="store_true", help="Run compliance check")
    parser.add_argument("--alert", action="store_true", help="Check for alerts only")
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed report"
    )
    parser.add_argument(
        "--fix-suggestions", action="store_true", help="Generate fix suggestions"
    )
    parser.add_argument(
        "--artifacts-root",
        default="docs/artifacts",
        help="Root directory for artifacts",
    )
    parser.add_argument("--output", help="Output file for report")

    args = parser.parse_args()

    monitor = ArtifactMonitor(args.artifacts_root)

    if args.check:
        success = monitor.run_compliance_check()
        return 0 if success else 1

    elif args.alert:
        report = monitor.check_organization_compliance()
        alerts = monitor.check_for_alerts(report)

        if alerts:
            print("🚨 ALERTS FOUND:")
            for alert in alerts:
                print(f"  {alert}")
            return 1
        else:
            print("✅ No alerts - compliance is good")
            return 0

    elif args.report:
        report = monitor.check_organization_compliance()
        report_text = monitor.generate_compliance_report(report)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report_text)
            print(f"Report written to {args.output}")
        else:
            print(report_text)

        return 0 if report["compliance_rate"] >= 80 else 1

    elif args.fix_suggestions:
        report = monitor.check_organization_compliance()
        suggestions = monitor.generate_fix_suggestions(report)

        if args.output:
            with open(args.output, "w") as f:
                f.write(suggestions)
            print(f"Fix suggestions written to {args.output}")
        else:
            print(suggestions)

        return 0

    else:
        # Default: run compliance check
        success = monitor.run_compliance_check()
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())
