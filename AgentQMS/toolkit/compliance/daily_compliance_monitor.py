#!/usr/bin/env python3
"""
Daily Compliance Monitoring Script

This script provides comprehensive daily compliance monitoring with:
- Automated daily compliance checks
- Trend analysis and reporting
- Alert system for compliance drops
- Integration with automated fix scripts
- Historical data tracking

Usage:
    python daily_compliance_monitor.py --run-daily-check
    python daily_compliance_monitor.py --generate-report
    python daily_compliance_monitor.py --setup-cron
    python daily_compliance_monitor.py --auto-fix-threshold 0.8
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DailyComplianceReport:
    """Daily compliance report data"""

    date: str
    total_files: int
    compliant_files: int
    total_issues: int
    compliance_rate: float
    issues_by_type: dict[str, int]
    files_by_directory: dict[str, int]
    trend_7day: float
    trend_30day: float
    auto_fixes_applied: int
    manual_fixes_needed: int
    recommendations: list[str]


class DailyComplianceMonitor:
    """Daily compliance monitoring system"""

    def __init__(
        self,
        artifacts_root: str = "docs/artifacts",
        db_path: str = "compliance_monitoring.db",
    ):
        self.artifacts_root = Path(artifacts_root)
        self.db_path = db_path
        self.scripts_dir = Path("scripts/agent_tools")

        # Initialize database
        self._init_database()

        # Compliance thresholds
        self.thresholds = {
            "critical": 0.80,  # Below 80% - critical
            "warning": 0.90,  # Below 90% - warning
            "target": 0.95,  # Target compliance rate
            "excellent": 0.98,  # Above 98% - excellent
        }

        # Auto-fix settings
        self.auto_fix_settings = {
            "enabled": True,
            "threshold": 0.85,  # Auto-fix if compliance below 85%
            "max_files_per_run": 50,
            "backup_enabled": True,
        }

    def _init_database(self):
        """Initialize SQLite database for compliance tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_files INTEGER,
                compliant_files INTEGER,
                total_issues INTEGER,
                compliance_rate REAL,
                issues_by_type TEXT,
                files_by_directory TEXT,
                trend_7day REAL,
                trend_30day REAL,
                auto_fixes_applied INTEGER,
                manual_fixes_needed INTEGER,
                recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                threshold REAL,
                current_value REAL,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auto_fix_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                script_name TEXT NOT NULL,
                files_processed INTEGER,
                fixes_applied INTEGER,
                errors INTEGER,
                execution_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def run_daily_check(self, auto_fix: bool = False) -> DailyComplianceReport:
        """Run comprehensive daily compliance check"""
        print(
            f"🔍 Running daily compliance check for {datetime.now().strftime('%Y-%m-%d')}"
        )

        # Calculate current compliance metrics
        metrics = self._calculate_compliance_metrics()

        # Calculate trends
        trend_7day = self._calculate_trend(7)
        trend_30day = self._calculate_trend(30)

        # Apply auto-fixes if enabled and below threshold
        auto_fixes_applied = 0
        if (
            auto_fix
            and metrics["compliance_rate"] < self.auto_fix_settings["threshold"]
        ):
            auto_fixes_applied = self._apply_auto_fixes()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, trend_7day, trend_30day
        )

        # Create report
        report = DailyComplianceReport(
            date=datetime.now().strftime("%Y-%m-%d"),
            total_files=metrics["total_files"],
            compliant_files=metrics["compliant_files"],
            total_issues=metrics["total_issues"],
            compliance_rate=metrics["compliance_rate"],
            issues_by_type=metrics["issues_by_type"],
            files_by_directory=metrics["files_by_directory"],
            trend_7day=trend_7day,
            trend_30day=trend_30day,
            auto_fixes_applied=auto_fixes_applied,
            manual_fixes_needed=metrics["total_issues"] - auto_fixes_applied,
            recommendations=recommendations,
        )

        # Store report in database
        self._store_report(report)

        # Check for alerts
        self._check_and_store_alerts(report)

        return report

    def _calculate_compliance_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive compliance metrics"""
        metrics = {
            "total_files": 0,
            "compliant_files": 0,
            "total_issues": 0,
            "compliance_rate": 0.0,
            "issues_by_type": {},
            "files_by_directory": {},
        }

        # Run validation script to get detailed metrics
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.scripts_dir / "validate_artifacts.py"),
                    "--all",
                    "--artifacts-root",
                    str(self.artifacts_root),
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                validation_data = json.loads(result.stdout)

                for file_result in validation_data:
                    metrics["total_files"] += 1

                    # Count files by directory
                    file_path = Path(file_result["file"])
                    relative_path = file_path.relative_to(self.artifacts_root)
                    directory = str(relative_path.parent)
                    metrics["files_by_directory"][directory] = (
                        metrics["files_by_directory"].get(directory, 0) + 1
                    )

                    if file_result["valid"]:
                        metrics["compliant_files"] += 1
                    else:
                        metrics["total_issues"] += len(file_result["errors"])

                        # Count issues by type
                        for error in file_result["errors"]:
                            issue_type = error.split(":")[0].strip()
                            metrics["issues_by_type"][issue_type] = (
                                metrics["issues_by_type"].get(issue_type, 0) + 1
                            )

                # Calculate compliance rate
                if metrics["total_files"] > 0:
                    metrics["compliance_rate"] = (
                        metrics["compliant_files"] / metrics["total_files"]
                    )

        except Exception as e:
            print(f"Warning: Could not run validation script: {e}")
            # Fallback to basic file counting
            for file_path in self.artifacts_root.rglob("*.md"):
                if file_path.is_file() and file_path.name != "INDEX.md":
                    metrics["total_files"] += 1

        return metrics

    def _calculate_trend(self, days: int) -> float:
        """Calculate compliance trend over specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get historical data
        cursor.execute(f"""
            SELECT compliance_rate FROM daily_reports
            WHERE date >= date('now', '-{days} days')
            ORDER BY date DESC
        """)

        rates = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(rates) < 2:
            return 0.0

        # Calculate trend (positive = improving, negative = declining)
        trend = rates[0] - rates[-1]
        return trend

    def _apply_auto_fixes(self) -> int:
        """Apply automated fixes and return number of fixes applied"""
        print("🔧 Applying automated fixes...")

        fixes_applied = 0
        scripts = [
            "fix_naming_conventions.py",
            "add_frontmatter.py",
            "fix_categories.py",
            "reorganize_files.py",
        ]

        for script in scripts:
            script_path = self.scripts_dir / script
            if script_path.exists():
                try:
                    print(f"   Running {script}...")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(script_path),
                            "--directory",
                            str(self.artifacts_root),
                            "--auto-fix"
                            if script != "reorganize_files.py"
                            else "--move-to-correct-dirs",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )

                    if result.returncode == 0:
                        # Count fixes from output (basic parsing)
                        output_lines = result.stdout.split("\n")
                        for line in output_lines:
                            if "✅" in line and (
                                "Fixed" in line or "Added" in line or "Moved" in line
                            ):
                                fixes_applied += 1

                    # Log execution
                    self._log_auto_fix_execution(
                        script,
                        result.returncode == 0,
                        len(output_lines) if "output_lines" in locals() else 0,
                    )

                except Exception as e:
                    print(f"   Error running {script}: {e}")
                    self._log_auto_fix_execution(script, False, 0, str(e))

        return fixes_applied

    def _log_auto_fix_execution(
        self,
        script_name: str,
        success: bool,
        files_processed: int,
        error: str | None = None,
    ):
        """Log auto-fix execution to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO auto_fix_log (date, script_name, files_processed, fixes_applied, errors, execution_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().strftime("%Y-%m-%d"),
                script_name,
                files_processed,
                1 if success else 0,
                1 if error else 0,
                0.0,  # Could add timing if needed
            ),
        )

        conn.commit()
        conn.close()

    def _generate_recommendations(
        self, metrics: dict[str, Any], trend_7day: float, trend_30day: float
    ) -> list[str]:
        """Generate actionable recommendations based on metrics"""
        recommendations = []

        # Compliance rate recommendations
        if metrics["compliance_rate"] < self.thresholds["critical"]:
            recommendations.append(
                "🚨 CRITICAL: Compliance rate below 80%. Immediate action required."
            )
        elif metrics["compliance_rate"] < self.thresholds["warning"]:
            recommendations.append(
                "⚠️ WARNING: Compliance rate below 90%. Consider running automated fixes."
            )
        elif metrics["compliance_rate"] < self.thresholds["target"]:
            recommendations.append(
                "📈 IMPROVEMENT: Compliance rate below target. Run automated fixes to reach 95%."
            )
        elif metrics["compliance_rate"] >= self.thresholds["excellent"]:
            recommendations.append(
                "🎉 EXCELLENT: Compliance rate above 98%. Maintain current practices."
            )

        # Trend recommendations
        if trend_7day < -0.05:
            recommendations.append(
                "📉 DECLINING: 7-day trend shows declining compliance. Investigate recent changes."
            )
        elif trend_7day > 0.05:
            recommendations.append(
                "📈 IMPROVING: 7-day trend shows improving compliance. Continue current efforts."
            )

        # Issue type recommendations
        if "Naming" in metrics["issues_by_type"]:
            recommendations.append(
                "📝 NAMING: Run naming convention fixes for better file organization."
            )

        if "Frontmatter" in metrics["issues_by_type"]:
            recommendations.append(
                "📄 FRONTMATTER: Add missing frontmatter to improve metadata compliance."
            )

        if "Directory" in metrics["issues_by_type"]:
            recommendations.append(
                "📁 ORGANIZATION: Reorganize misplaced files to correct directories."
            )

        # Auto-fix recommendations
        if metrics["compliance_rate"] < self.auto_fix_settings["threshold"]:
            recommendations.append(
                "🔧 AUTO-FIX: Enable automated fixes to improve compliance quickly."
            )

        return recommendations

    def _store_report(self, report: DailyComplianceReport):
        """Store daily report in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO daily_reports
            (date, total_files, compliant_files, total_issues, compliance_rate,
             issues_by_type, files_by_directory, trend_7day, trend_30day,
             auto_fixes_applied, manual_fixes_needed, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                report.date,
                report.total_files,
                report.compliant_files,
                report.total_issues,
                report.compliance_rate,
                json.dumps(report.issues_by_type),
                json.dumps(report.files_by_directory),
                report.trend_7day,
                report.trend_30day,
                report.auto_fixes_applied,
                report.manual_fixes_needed,
                json.dumps(report.recommendations),
            ),
        )

        conn.commit()
        conn.close()

    def _check_and_store_alerts(self, report: DailyComplianceReport):
        """Check for alerts and store them in database"""
        alerts = []

        # Compliance rate alerts
        if report.compliance_rate < self.thresholds["critical"]:
            alerts.append(
                {
                    "alert_type": "critical_compliance",
                    "severity": "critical",
                    "message": f"Critical: Compliance rate {report.compliance_rate:.1%} below 80%",
                    "threshold": self.thresholds["critical"],
                    "current_value": report.compliance_rate,
                }
            )
        elif report.compliance_rate < self.thresholds["warning"]:
            alerts.append(
                {
                    "alert_type": "warning_compliance",
                    "severity": "warning",
                    "message": f"Warning: Compliance rate {report.compliance_rate:.1%} below 90%",
                    "threshold": self.thresholds["warning"],
                    "current_value": report.compliance_rate,
                }
            )

        # Trend alerts
        if report.trend_7day < -0.05:
            alerts.append(
                {
                    "alert_type": "declining_trend",
                    "severity": "warning",
                    "message": f"Declining trend: 7-day change {report.trend_7day:.1%}",
                    "threshold": -0.05,
                    "current_value": report.trend_7day,
                }
            )

        # Store alerts
        if alerts:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for alert in alerts:
                cursor.execute(
                    """
                    INSERT INTO compliance_alerts
                    (date, alert_type, severity, message, threshold, current_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        report.date,
                        alert["alert_type"],
                        alert["severity"],
                        alert["message"],
                        alert["threshold"],
                        alert["current_value"],
                    ),
                )

            conn.commit()
            conn.close()

    def generate_daily_report(self, date: str | None = None) -> str:
        """Generate formatted daily compliance report"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM daily_reports WHERE date = ?", (date,))
        row = cursor.fetchone()

        if not row:
            return f"No compliance report found for {date}"

        # Parse data
        report_data = {
            "date": row[1],
            "total_files": row[2],
            "compliant_files": row[3],
            "total_issues": row[4],
            "compliance_rate": row[5],
            "issues_by_type": json.loads(row[6]),
            "files_by_directory": json.loads(row[7]),
            "trend_7day": row[8],
            "trend_30day": row[9],
            "auto_fixes_applied": row[10],
            "manual_fixes_needed": row[11],
            "recommendations": json.loads(row[12]),
        }

        conn.close()

        # Generate formatted report
        report = []
        report.append("=" * 60)
        report.append(f"DAILY COMPLIANCE REPORT - {date}")
        report.append("=" * 60)
        report.append("")

        # Summary metrics
        report.append("📊 SUMMARY METRICS")
        report.append("-" * 30)
        report.append(f"Total Files: {report_data['total_files']}")
        report.append(f"Compliant Files: {report_data['compliant_files']}")
        report.append(f"Total Issues: {report_data['total_issues']}")
        report.append(f"Compliance Rate: {report_data['compliance_rate']:.1%}")
        report.append("")

        # Trends
        report.append("📈 TRENDS")
        report.append("-" * 30)
        report.append(f"7-Day Trend: {report_data['trend_7day']:+.1%}")
        report.append(f"30-Day Trend: {report_data['trend_30day']:+.1%}")
        report.append("")

        # Issues breakdown
        if report_data["issues_by_type"]:
            report.append("🔍 ISSUES BY TYPE")
            report.append("-" * 30)
            for issue_type, count in report_data["issues_by_type"].items():
                report.append(f"{issue_type}: {count}")
            report.append("")

        # Directory breakdown
        if report_data["files_by_directory"]:
            report.append("📁 FILES BY DIRECTORY")
            report.append("-" * 30)
            for directory, count in report_data["files_by_directory"].items():
                report.append(f"{directory}: {count}")
            report.append("")

        # Auto-fixes
        report.append("🔧 AUTO-FIXES")
        report.append("-" * 30)
        report.append(f"Applied: {report_data['auto_fixes_applied']}")
        report.append(f"Manual Fixes Needed: {report_data['manual_fixes_needed']}")
        report.append("")

        # Recommendations
        if report_data["recommendations"]:
            report.append("💡 RECOMMENDATIONS")
            report.append("-" * 30)
            for recommendation in report_data["recommendations"]:
                report.append(f"• {recommendation}")
            report.append("")

        return "\n".join(report)

    def setup_cron_job(self):
        """Set up cron job for daily monitoring"""
        cron_script = Path("scripts/agent_tools/daily_compliance_cron.sh")

        # Create cron script
        cron_content = f"""#!/bin/bash
# Daily Compliance Monitoring Cron Job
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

cd {Path.cwd()}
python scripts/agent_tools/daily_compliance_monitor.py --run-daily-check --auto-fix-threshold 0.85 >> logs/daily_compliance.log 2>&1
"""

        cron_script.write_text(cron_content)
        cron_script.chmod(0o755)

        # Create logs directory
        Path("logs").mkdir(exist_ok=True)

        print(f"📅 Cron script created: {cron_script}")
        print("To set up daily monitoring, add this to your crontab:")
        print("0 9 * * * " + str(cron_script.absolute()))

    def get_compliance_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get compliance history for specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT date, compliance_rate, total_files, total_issues, auto_fixes_applied
            FROM daily_reports
            WHERE date >= date('now', '-{days} days')
            ORDER BY date DESC
        """)

        history = []
        for row in cursor.fetchall():
            history.append(
                {
                    "date": row[0],
                    "compliance_rate": row[1],
                    "total_files": row[2],
                    "total_issues": row[3],
                    "auto_fixes_applied": row[4],
                }
            )

        conn.close()
        return history


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Daily compliance monitoring system")
    parser.add_argument(
        "--run-daily-check", action="store_true", help="Run daily compliance check"
    )
    parser.add_argument(
        "--generate-report", help="Generate report for specific date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--setup-cron", action="store_true", help="Set up cron job for daily monitoring"
    )
    parser.add_argument(
        "--auto-fix-threshold",
        type=float,
        default=0.85,
        help="Auto-fix threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--history", type=int, help="Show compliance history for N days"
    )
    parser.add_argument(
        "--artifacts-root",
        default="docs/artifacts",
        help="Root directory for artifacts",
    )
    parser.add_argument(
        "--db-path", default="compliance_monitoring.db", help="Database file path"
    )

    args = parser.parse_args()

    monitor = DailyComplianceMonitor(args.artifacts_root, args.db_path)

    if args.run_daily_check:
        auto_fix = args.auto_fix_threshold < 1.0
        report = monitor.run_daily_check(auto_fix=auto_fix)

        # Print summary
        print("\n📊 Daily Compliance Check Complete")
        print(f"Date: {report.date}")
        print(f"Compliance Rate: {report.compliance_rate:.1%}")
        print(f"Total Files: {report.total_files}")
        print(f"Total Issues: {report.total_issues}")
        print(f"Auto-fixes Applied: {report.auto_fixes_applied}")

        if report.recommendations:
            print("\n💡 Recommendations:")
            for rec in report.recommendations:
                print(f"  {rec}")

    elif args.generate_report:
        report = monitor.generate_daily_report(args.generate_report)
        print(report)

    elif args.setup_cron:
        monitor.setup_cron_job()

    elif args.history:
        history = monitor.get_compliance_history(args.history)
        print(f"\n📈 Compliance History ({args.history} days)")
        print("-" * 50)
        for entry in history:
            print(
                f"{entry['date']}: {entry['compliance_rate']:.1%} ({entry['total_files']} files, {entry['total_issues']} issues)"
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
