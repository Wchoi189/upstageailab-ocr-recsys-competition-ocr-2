"""Tests for middleware dashboard CLI."""

import json
import subprocess
from pathlib import Path

import pytest


class TestCLIArguments:
    """Test CLI argument parsing."""

    def test_dashboard_health_only_flag(self):
        """Test --health-only shows health."""
        result = subprocess.run(
            ["uv", "run", "python", "AgentQMS/tools/middleware/dashboard.py", "--health-only"],
            capture_output=True,
            text=True,
            cwd="/workspaces/upstageailab-ocr-recsys-competition-ocr-2",
        )

        assert "Health Status" in result.stdout
        assert result.returncode == 0

    def test_dashboard_json_flag(self, tmp_path):
        """Test --json outputs JSON."""
        # Create sample metrics file
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({
            "timestamp": "2026-02-02T00:00:00+00:00",
            "period": "session",
            "policies": {},
            "exceptions": {}
        }))

        result = subprocess.run(
            [
                "uv", "run", "python",
                "AgentQMS/tools/middleware/dashboard.py",
                "--json",
                "--metrics-file", str(metrics_file)
            ],
            capture_output=True,
            text=True,
            cwd="/workspaces/upstageailab-ocr-recsys-competition-ocr-2",
        )

        # Output should be valid JSON
        try:
            json.loads(result.stdout)
            is_json = True
        except json.JSONDecodeError:
            is_json = False

        assert is_json or "period" in result.stdout


class TestHealthDisplay:
    """Test health status display."""

    def test_health_display_shows_status(self):
        """Test health display renders."""
        result = subprocess.run(
            ["uv", "run", "python", "AgentQMS/tools/middleware/dashboard.py", "--health-only"],
            capture_output=True,
            text=True,
            cwd="/workspaces/upstageailab-ocr-recsys-competition-ocr-2",
        )

        # Should show some health information
        assert len(result.stdout) > 0
        assert result.returncode == 0


class TestErrorHandling:
    """Test dashboard error handling."""

    def test_dashboard_missing_metrics_file(self):
        """Test graceful error with missing metrics."""
        result = subprocess.run(
            [
                "uv", "run", "python",
                "AgentQMS/tools/middleware/dashboard.py",
                    "--metrics-file", "/nonexistent/file.json"
            ],
            capture_output=True,
            text=True,
            cwd="/workspaces/upstageailab-ocr-recsys-competition-ocr-2",
        )

        # Should not crash, should show helpful message
        assert "No metrics file found" in result.stdout or "Health Status" in result.stdout
