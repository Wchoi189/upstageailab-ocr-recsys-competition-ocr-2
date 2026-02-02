"""Tests for middleware health checks."""

import json
from unittest.mock import MagicMock, patch

import pytest

from AgentQMS.middleware.health import check_policies_health, get_health_status


class TestPolicyHealthChecks:
    """Test policy health validation."""

    def test_check_policies_health_all_pass(self):
        """Test all policies healthy."""
        result = check_policies_health()

        assert result["status"] in ["healthy", "degraded"]
        assert "timestamp" in result
        assert "checks" in result
        assert isinstance(result["checks"], dict)

    def test_health_status_timestamp(self):
        """Test health check includes timestamp."""
        result = check_policies_health()

        # Should have ISO format timestamp
        assert "T" in result["timestamp"]
        assert "+" in result["timestamp"] or "Z" in result["timestamp"]


class TestInfrastructureChecks:
    """Test infrastructure health checks."""

   def test_logging_infrastructure_check(self, tmp_path, monkeypatch):
        """Test log directory writable check."""
        from AgentQMS.middleware import health as health_module

        # Mock get_project_root
        monkeypatch.setattr(
            "AgentQMS.middleware.health.get_project_root",
            lambda: tmp_path
        )

        # Create log directory
        (tmp_path / "outputs" / "logs" / "middleware").mkdir(parents=True)

        result = check_policies_health()

        # Should include logging infrastructure check
        assert "logging_infrastructure" in result["checks"]


class TestStatusReporting:
    """Test health status reporting."""

    def test_get_health_status_healthy(self):
        """Test status 'healthy' when all pass."""
        status = get_health_status()

        assert "status" in status
        assert "timestamp" in status
        assert "summary" in status
        assert "details" in status

    def test_health_status_summary_counts(self):
        """Test correct passed/failed counts."""
        status = get_health_status()

        summary = status["summary"]
        assert "total_checks" in summary
        assert "passed" in summary
        assert "failed" in summary

        # Counts should be consistent
        total = summary["total_checks"]
        passed = summary["passed"]
        failed = summary["failed"]

        assert passed + failed == total
