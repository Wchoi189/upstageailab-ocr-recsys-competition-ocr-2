#!/usr/bin/env python3
"""
Enhanced Compliance Alert System

This script provides comprehensive alerting capabilities for compliance monitoring:
- Multi-channel alert delivery (email, Slack, webhook, file)
- Intelligent alert throttling and deduplication
- Escalation policies for critical issues
- Alert history and resolution tracking
- Integration with monitoring systems

Usage:
    python compliance_alert_system.py --check-alerts
    python compliance_alert_system.py --send-test-alert
    python compliance_alert_system.py --setup-notifications
    python compliance_alert_system.py --alert-history
"""

import argparse
import json
import os
import smtplib
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import requests


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    condition: str  # 'compliance_rate', 'total_issues', 'trend_decline'
    operator: str  # 'lt', 'gt', 'eq', 'gte', 'lte'
    threshold: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    channels: list[str]  # ['email', 'slack', 'webhook', 'file']
    enabled: bool = True
    cooldown_minutes: int = 60  # Prevent spam


@dataclass
class Alert:
    """Alert instance"""

    id: str
    rule_name: str
    severity: str
    message: str
    current_value: float
    threshold: float
    timestamp: str
    channels_sent: list[str]
    resolved: bool = False
    resolved_at: str | None = None


class ComplianceAlertSystem:
    """Enhanced compliance alert system"""

    def __init__(
        self,
        config_file: str = "alert_config.json",
        db_path: str = "compliance_monitoring.db",
    ):
        self.config_file = config_file
        self.db_path = db_path
        self.config = self._load_config()
        self.alert_rules = self._load_alert_rules()

        # Initialize database
        self._init_alert_database()

    def _load_config(self) -> dict[str, Any]:
        """Load alert system configuration"""
        default_config = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_email": "",
                "to_emails": [],
            },
            "slack": {
                "enabled": False,
                "webhook_url": "",
                "channel": "#compliance-alerts",
                "username": "Compliance Bot",
            },
            "webhook": {"enabled": False, "url": "", "headers": {}, "timeout": 30},
            "file": {
                "enabled": True,
                "path": "compliance_alerts.json",
                "max_alerts": 1000,
            },
            "throttling": {
                "enabled": True,
                "max_alerts_per_hour": 10,
                "cooldown_minutes": 60,
            },
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file) as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")

        return default_config

    def _load_alert_rules(self) -> list[AlertRule]:
        """Load alert rules from configuration"""
        default_rules = [
            AlertRule(
                name="critical_compliance",
                condition="compliance_rate",
                operator="lt",
                threshold=0.80,
                severity="critical",
                channels=["email", "slack", "webhook", "file"],
                cooldown_minutes=30,
            ),
            AlertRule(
                name="warning_compliance",
                condition="compliance_rate",
                operator="lt",
                threshold=0.90,
                severity="high",
                channels=["slack", "file"],
                cooldown_minutes=60,
            ),
            AlertRule(
                name="too_many_issues",
                condition="total_issues",
                operator="gt",
                threshold=100,
                severity="medium",
                channels=["slack", "file"],
                cooldown_minutes=120,
            ),
            AlertRule(
                name="declining_trend",
                condition="trend_decline",
                operator="lt",
                threshold=-0.05,
                severity="medium",
                channels=["file"],
                cooldown_minutes=180,
            ),
            AlertRule(
                name="excellent_compliance",
                condition="compliance_rate",
                operator="gte",
                threshold=0.98,
                severity="low",
                channels=["file"],
                cooldown_minutes=1440,  # 24 hours
            ),
        ]

        # Load custom rules from config if available
        if "alert_rules" in self.config:
            custom_rules = []
            for rule_data in self.config["alert_rules"]:
                custom_rules.append(AlertRule(**rule_data))
            return custom_rules

        return default_rules

    def _init_alert_database(self):
        """Initialize alert-specific database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create alerts table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_alerts (
                id TEXT PRIMARY KEY,
                rule_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                current_value REAL,
                threshold REAL,
                timestamp TEXT NOT NULL,
                channels_sent TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create alert delivery log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_delivery_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (alert_id) REFERENCES compliance_alerts (id)
            )
        """)

        conn.commit()
        conn.close()

    def check_alerts(self, compliance_data: dict[str, Any]) -> list[Alert]:
        """Check compliance data against alert rules and generate alerts"""
        alerts = []

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # Check if rule should trigger
            should_alert, current_value = self._evaluate_rule(rule, compliance_data)

            if should_alert:
                # Check throttling/cooldown
                if self._is_alert_throttled(rule):
                    continue

                # Create alert
                alert = self._create_alert(rule, current_value, compliance_data)
                alerts.append(alert)

        return alerts

    def _evaluate_rule(
        self, rule: AlertRule, data: dict[str, Any]
    ) -> tuple[bool, float]:
        """Evaluate if an alert rule should trigger"""
        current_value = 0.0

        if rule.condition == "compliance_rate":
            current_value = data.get("compliance_rate", 0.0)
        elif rule.condition == "total_issues":
            current_value = data.get("total_issues", 0)
        elif rule.condition == "trend_decline":
            current_value = data.get("trend_7day", 0.0)
        else:
            return False, 0.0

        # Evaluate condition
        if rule.operator == "lt":
            return current_value < rule.threshold, current_value
        elif rule.operator == "gt":
            return current_value > rule.threshold, current_value
        elif rule.operator == "eq":
            return current_value == rule.threshold, current_value
        elif rule.operator == "gte":
            return current_value >= rule.threshold, current_value
        elif rule.operator == "lte":
            return current_value <= rule.threshold, current_value

        return False, current_value

    def _is_alert_throttled(self, rule: AlertRule) -> bool:
        """Check if alert is throttled due to cooldown"""
        if not self.config["throttling"]["enabled"]:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check for recent alerts of this rule
        cooldown_time = datetime.now() - timedelta(minutes=rule.cooldown_minutes)
        cursor.execute(
            """
            SELECT COUNT(*) FROM compliance_alerts
            WHERE rule_name = ? AND timestamp > ? AND resolved = FALSE
        """,
            (rule.name, cooldown_time.isoformat()),
        )

        recent_count = cursor.fetchone()[0]
        conn.close()

        return recent_count > 0

    def _create_alert(
        self, rule: AlertRule, current_value: float, data: dict[str, Any]
    ) -> Alert:
        """Create alert instance"""
        alert_id = f"{rule.name}_{int(time.time())}"

        # Generate message
        message = self._generate_alert_message(rule, current_value, data)

        return Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=message,
            current_value=current_value,
            threshold=rule.threshold,
            timestamp=datetime.now().isoformat(),
            channels_sent=[],
        )

    def _generate_alert_message(
        self, rule: AlertRule, current_value: float, data: dict[str, Any]
    ) -> str:
        """Generate alert message based on rule and data"""
        severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📊", "low": "ℹ️"}

        emoji = severity_emoji.get(rule.severity, "📢")

        if rule.condition == "compliance_rate":
            return f"{emoji} Compliance Alert: Rate is {current_value:.1%} (threshold: {rule.threshold:.1%})"
        elif rule.condition == "total_issues":
            return f"{emoji} Issue Alert: {current_value} issues found (threshold: {rule.threshold})"
        elif rule.condition == "trend_decline":
            return f"{emoji} Trend Alert: 7-day trend is {current_value:.1%} (threshold: {rule.threshold:.1%})"

        return f"{emoji} Alert: {rule.name} triggered"

    def send_alert(
        self, alert: Alert, channels: list[str] | None = None
    ) -> dict[str, bool]:
        """Send alert through specified channels"""
        if channels is None:
            # Find channels from rule
            rule = next(
                (r for r in self.alert_rules if r.name == alert.rule_name), None
            )
            if rule:
                channels = rule.channels
            else:
                channels = ["file"]  # Default fallback

        results = {}

        for channel in channels:
            try:
                success = self._send_to_channel(alert, channel)
                results[channel] = success

                if success:
                    alert.channels_sent.append(channel)

                # Log delivery attempt
                self._log_delivery(alert.id, channel, success)

            except Exception as e:
                results[channel] = False
                self._log_delivery(alert.id, channel, False, str(e))

        # Store alert in database
        self._store_alert(alert)

        return results

    def _send_to_channel(self, alert: Alert, channel: str) -> bool:
        """Send alert to specific channel"""
        if channel == "email":
            return self._send_email(alert)
        elif channel == "slack":
            return self._send_slack(alert)
        elif channel == "webhook":
            return self._send_webhook(alert)
        elif channel == "file":
            return self._send_to_file(alert)
        else:
            return False

    def _send_email(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not self.config["email"]["enabled"]:
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.config["email"]["from_email"]
            msg["To"] = ", ".join(self.config["email"]["to_emails"])
            msg["Subject"] = (
                f"[{alert.severity.upper()}] Compliance Alert: {alert.rule_name}"
            )

            body = f"""
Compliance Alert Details:
- Rule: {alert.rule_name}
- Severity: {alert.severity}
- Message: {alert.message}
- Current Value: {alert.current_value}
- Threshold: {alert.threshold}
- Timestamp: {alert.timestamp}

Please review the compliance status and take appropriate action.
"""

            msg.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP(
                self.config["email"]["smtp_server"], self.config["email"]["smtp_port"]
            )
            server.starttls()
            server.login(
                self.config["email"]["username"], self.config["email"]["password"]
            )
            server.send_message(msg)
            server.quit()

            return True

        except Exception as e:
            print(f"Email send failed: {e}")
            return False

    def _send_slack(self, alert: Alert) -> bool:
        """Send alert via Slack webhook"""
        if not self.config["slack"]["enabled"]:
            return False

        try:
            severity_colors = {
                "critical": "danger",
                "high": "warning",
                "medium": "good",
                "low": "#36a64f",
            }

            payload = {
                "channel": self.config["slack"]["channel"],
                "username": self.config["slack"]["username"],
                "attachments": [
                    {
                        "color": severity_colors.get(alert.severity, "good"),
                        "title": f"Compliance Alert: {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity,
                                "short": True,
                            },
                            {
                                "title": "Current Value",
                                "value": str(alert.current_value),
                                "short": True,
                            },
                            {
                                "title": "Threshold",
                                "value": str(alert.threshold),
                                "short": True,
                            },
                            {
                                "title": "Timestamp",
                                "value": alert.timestamp,
                                "short": True,
                            },
                        ],
                        "footer": "Compliance Monitoring System",
                        "ts": int(time.time()),
                    }
                ],
            }

            response = requests.post(
                self.config["slack"]["webhook_url"], json=payload, timeout=30
            )

            return response.status_code == 200

        except Exception as e:
            print(f"Slack send failed: {e}")
            return False

    def _send_webhook(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        if not self.config["webhook"]["enabled"]:
            return False

        try:
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity,
                "message": alert.message,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp,
            }

            response = requests.post(
                self.config["webhook"]["url"],
                json=payload,
                headers=self.config["webhook"]["headers"],
                timeout=self.config["webhook"]["timeout"],
            )

            return response.status_code in [200, 201, 202]

        except Exception as e:
            print(f"Webhook send failed: {e}")
            return False

    def _send_to_file(self, alert: Alert) -> bool:
        """Send alert to file"""
        try:
            alert_file = Path(self.config["file"]["path"])

            # Load existing alerts
            alerts_data = []
            if alert_file.exists():
                try:
                    with open(alert_file) as f:
                        alerts_data = json.load(f)
                except Exception:
                    alerts_data = []

            # Add new alert
            alerts_data.append(asdict(alert))

            # Keep only max alerts
            max_alerts = self.config["file"]["max_alerts"]
            if len(alerts_data) > max_alerts:
                alerts_data = alerts_data[-max_alerts:]

            # Write back
            with open(alert_file, "w") as f:
                json.dump(alerts_data, f, indent=2)

            return True

        except Exception as e:
            print(f"File write failed: {e}")
            return False

    def _log_delivery(
        self, alert_id: str, channel: str, success: bool, error: str | None = None
    ):
        """Log alert delivery attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO alert_delivery_log (alert_id, channel, success, error_message)
            VALUES (?, ?, ?, ?)
        """,
            (alert_id, channel, success, error),
        )

        conn.commit()
        conn.close()

    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO compliance_alerts
            (id, rule_name, severity, message, current_value, threshold,
             timestamp, channels_sent, resolved, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                alert.id,
                alert.rule_name,
                alert.severity,
                alert.message,
                alert.current_value,
                alert.threshold,
                alert.timestamp,
                json.dumps(alert.channels_sent),
                alert.resolved,
                alert.resolved_at,
            ),
        )

        conn.commit()
        conn.close()

    def get_alert_history(self, days: int = 7) -> list[dict[str, Any]]:
        """Get alert history for specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT * FROM compliance_alerts
            WHERE timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp DESC
        """)

        alerts = []
        for row in cursor.fetchall():
            alerts.append(
                {
                    "id": row[0],
                    "rule_name": row[1],
                    "severity": row[2],
                    "message": row[3],
                    "current_value": row[4],
                    "threshold": row[5],
                    "timestamp": row[6],
                    "channels_sent": json.loads(row[7]) if row[7] else [],
                    "resolved": bool(row[8]),
                    "resolved_at": row[9],
                }
            )

        conn.close()
        return alerts

    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE compliance_alerts
            SET resolved = TRUE, resolved_at = ?
            WHERE id = ?
        """,
            (datetime.now().isoformat(), alert_id),
        )

        conn.commit()
        conn.close()

    def setup_notifications(self):
        """Interactive setup for notification channels"""
        print("🔔 Compliance Alert System Setup")
        print("=" * 40)

        # Email setup
        if input("Configure email notifications? (y/n): ").lower() == "y":
            self.config["email"]["enabled"] = True
            self.config["email"]["smtp_server"] = (
                input("SMTP Server: ") or "smtp.gmail.com"
            )
            self.config["email"]["smtp_port"] = int(input("SMTP Port: ") or "587")
            self.config["email"]["username"] = input("Username: ")
            self.config["email"]["password"] = input("Password: ")
            self.config["email"]["from_email"] = input("From Email: ")
            self.config["email"]["to_emails"] = input(
                "To Emails (comma-separated): "
            ).split(",")

        # Slack setup
        if input("Configure Slack notifications? (y/n): ").lower() == "y":
            self.config["slack"]["enabled"] = True
            self.config["slack"]["webhook_url"] = input("Webhook URL: ")
            self.config["slack"]["channel"] = input("Channel: ") or "#compliance-alerts"
            self.config["slack"]["username"] = (
                input("Bot Username: ") or "Compliance Bot"
            )

        # Webhook setup
        if input("Configure webhook notifications? (y/n): ").lower() == "y":
            self.config["webhook"]["enabled"] = True
            self.config["webhook"]["url"] = input("Webhook URL: ")
            headers_input = input("Headers (JSON format): ")
            if headers_input:
                self.config["webhook"]["headers"] = json.loads(headers_input)

        # Save configuration
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)

        print(f"✅ Configuration saved to {self.config_file}")

    def send_test_alert(self):
        """Send test alert to verify configuration"""
        test_alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            severity="medium",
            message="This is a test alert to verify notification configuration",
            current_value=0.85,
            threshold=0.90,
            timestamp=datetime.now().isoformat(),
            channels_sent=[],
        )

        print("📤 Sending test alert...")
        results = self.send_alert(test_alert, ["file"])  # Always include file

        # Add other channels if enabled
        channels = ["file"]
        if self.config["email"]["enabled"]:
            channels.append("email")
        if self.config["slack"]["enabled"]:
            channels.append("slack")
        if self.config["webhook"]["enabled"]:
            channels.append("webhook")

        results = self.send_alert(test_alert, channels)

        print("Test alert results:")
        for channel, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {channel}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Compliance alert system")
    parser.add_argument("--check-alerts", action="store_true", help="Check for alerts")
    parser.add_argument(
        "--send-test-alert", action="store_true", help="Send test alert"
    )
    parser.add_argument(
        "--setup-notifications", action="store_true", help="Setup notification channels"
    )
    parser.add_argument(
        "--alert-history", type=int, help="Show alert history for N days"
    )
    parser.add_argument("--resolve-alert", help="Resolve alert by ID")
    parser.add_argument("--config", default="alert_config.json", help="Config file")
    parser.add_argument(
        "--db-path", default="compliance_monitoring.db", help="Database file path"
    )

    args = parser.parse_args()

    alert_system = ComplianceAlertSystem(args.config, args.db_path)

    if args.check_alerts:
        # This would typically be called by the monitoring system
        # with current compliance data
        print("Alert checking would be integrated with monitoring system")

    elif args.send_test_alert:
        alert_system.send_test_alert()

    elif args.setup_notifications:
        alert_system.setup_notifications()

    elif args.alert_history:
        history = alert_system.get_alert_history(args.alert_history)
        print(f"\n📋 Alert History ({args.alert_history} days)")
        print("-" * 50)
        for alert in history:
            status = "✅" if alert["resolved"] else "🔴"
            print(f"{status} {alert['timestamp']}: {alert['message']}")

    elif args.resolve_alert:
        alert_system.resolve_alert(args.resolve_alert)
        print(f"✅ Alert {args.resolve_alert} resolved")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
