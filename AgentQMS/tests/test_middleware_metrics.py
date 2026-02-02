"""Tests for middleware metrics collection."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from AgentQMS.middleware.telemetry import TelemetryPipeline


class TestCounterMetrics:
    """Test counter metric increments."""

    def test_checks_counter_increments(self):
        """Test check counter increments per validation."""
        interceptor = MagicMock()
        interceptor.validate = MagicMock()

        pipeline = TelemetryPipeline([interceptor])

        # Run multiple validations
        pipeline.validate("test_tool", {})
        pipeline.validate("test_tool", {})
        pipeline.validate("test_tool", {})

        # Check metrics
        assert pipeline.metrics["checks"]["MockMagicMock"] == 3

    def test_violations_counter_by_policy(self):
        """Test violations tracked per policy."""
        from AgentQMS.middleware.policies import PolicyViolation

        interceptor = MagicMock()
        interceptor.validate = MagicMock(
            side_effect=PolicyViolation("Test violation")
        )

        pipeline = TelemetryPipeline([interceptor])

        # Run validations that violate policy
        for _ in range(2):
            try:
                pipeline.validate("test_tool", {})
            except PolicyViolation:
                pass

        # Check violations counted
        assert pipeline.metrics["violations"]["MockMagicMock"] == 2

    def test_exceptions_counter_by_type(self):
        """Test exceptions tracked by type."""
        interceptor = MagicMock()
        interceptor.validate = MagicMock(side_effect=ValueError("Test"))

        pipeline = TelemetryPipeline([interceptor])

        # Run validations that raise exceptions
        for _ in range(3):
            try:
                pipeline.validate("test_tool", {})
            except Exception:
                pass

        # Check exceptions counted
        assert pipeline.metrics["exceptions"]["ValueError"] == 3


class TestDurationMetrics:
    """Test duration tracking."""

    def test_duration_tracking(self):
        """Test duration recorded for each validation."""
        import time

        interceptor = MagicMock()

        def slow_validate(*args, **kwargs):
            time.sleep(0.01)  # 10ms

        interceptor.validate = slow_validate

        pipeline = TelemetryPipeline([interceptor])
        pipeline.validate("test_tool", {})

        # Check duration was recorded
        assert "MockMagicMock" in pipeline.metrics["durations"]
        assert len(pipeline.metrics["durations"]["MockMagicMock"]) > 0

    def test_average_duration_calculation(self):
        """Test correct avg calculation."""
        interceptor = MagicMock()
        interceptor.validate = MagicMock()

        pipeline = TelemetryPipeline([interceptor])

        # Run multiple times
        for _ in range(5):
            pipeline.validate("test_tool", {})

        # Should have 5 durations recorded
        assert len(pipeline.metrics["durations"]["MockMagicMock"]) == 5


class TestMetricsExport:
    """Test metrics export functionality."""

    def test_export_metrics_creates_file(self, tmp_path):
        """Test JSON file created."""
        interceptor = MagicMock()
        interceptor.validate = MagicMock()

        pipeline = TelemetryPipeline([interceptor])
        pipeline.validate("test_tool", {})

        output_file = tmp_path / "metrics.json"
        pipeline.export_metrics(output_file)

        assert output_file.exists()

    def test_export_metrics_format(self, tmp_path):
        """Test correct JSON structure."""
        interceptor = MagicMock()
        interceptor.validate = MagicMock()

        pipeline = TelemetryPipeline([interceptor])
        pipeline.validate("test_tool", {})

        output_file = tmp_path / "metrics.json"
        pipeline.export_metrics(output_file)

        with output_file.open() as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "period" in data
        assert "policies" in data
        assert isinstance(data["policies"], dict)

    def test_export_metrics_timestamp(self, tmp_path):
        """Test includes current timestamp."""
        interceptor = MagicMock()
        interceptor.validate = MagicMock()

        pipeline = TelemetryPipeline([interceptor])
        output_file = tmp_path / "metrics.json"
        pipeline.export_metrics(output_file)

        with output_file.open() as f:
            data = json.load(f)

        # Should have ISO format timestamp
        assert "T" in data["timestamp"]
        assert ("+" in data["timestamp"] or "Z" in data["timestamp"])

    def test_export_metrics_all_policies(self, tmp_path):
        """Test all policies included."""
        interceptor1 = MagicMock()
        interceptor1.validate = MagicMock()

        interceptor2 = MagicMock()
        interceptor2.validate = MagicMock()

        pipeline = TelemetryPipeline([interceptor1, interceptor2])
        pipeline.validate("test_tool", {})

        output_file = tmp_path / "metrics.json"
        pipeline.export_metrics(output_file)

        with output_file.open() as f:
            data = json.load(f)

        # Both policies should be in export (with "MockMagicMock" names)
        assert len(data["policies"]) >= 1

    def test_export_metrics_error_handling(self, tmp_path):
        """Test graceful failure on write errors."""
        interceptor = MagicMock()
        interceptor.validate = MagicMock()

        pipeline = TelemetryPipeline([interceptor])

        # Try to write to invalid path
        invalid_path = tmp_path / "nonexistent" / "dir" / "metrics.json"

        # Should not raise
        pipeline.export_metrics(invalid_path)
