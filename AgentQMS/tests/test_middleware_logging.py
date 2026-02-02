"""Tests for middleware logging infrastructure."""

import json
import logging
from pathlib import Path

import pytest

from AgentQMS.middleware.logging_config import (
    JsonFormatter,
    ensure_log_directories,
    setup_middleware_logger,
)


class TestJsonFormatter:
    """Test JSON formatter for structured logging."""

    def test_json_formatter_basic(self):
        """Verify basic JSON structure."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["component"] == "test"
        assert data["message"] == "Test message"

    def test_json_formatter_with_extra(self):
        """Verify extra context fields are included."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.extra = {"custom_field": "custom_value", "count": 42}

        output = formatter.format(record)
        data = json.loads(output)

        assert data["custom_field"] == "custom_value"
        assert data["count"] == 42

    def test_json_formatter_timestamp_timezone(self):
        """Verify UTC timezone in timestamp."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        # ISO format with timezone
        assert "+00:00" in data["timestamp"] or "Z" in data["timestamp"]


class TestLoggerSetup:
    """Test logger setup and configuration."""

    def test_setup_middleware_logger(self, tmp_path):
        """Test basic logger creation."""
        logger = setup_middleware_logger("test", output_dir=tmp_path)

        assert logger.name == "agentqms.middleware.test"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_logger_level_configuration(self, tmp_path):
        """Test log level filtering."""
        logger = setup_middleware_logger("test", level=logging.WARNING, output_dir=tmp_path)

        assert logger.level == logging.WARNING

    def test_logger_rotation_enabled(self, tmp_path):
        """Test rotation handler is attached when enabled."""
        from logging.handlers import TimedRotatingFileHandler

        logger = setup_middleware_logger(
            "test", output_dir=tmp_path, enable_rotation=True
        )

        # Check for TimedRotatingFileHandler
        has_rotating_handler = any(
            isinstance(h, TimedRotatingFileHandler) for h in logger.handlers
        )
        assert has_rotating_handler

    def test_logger_rotation_disabled(self, tmp_path):
        """Test no rotation when disabled."""
        from logging.handlers import TimedRotatingFileHandler

        logger = setup_middleware_logger(
            "test", output_dir=tmp_path, enable_rotation=False
        )

        # Should not have TimedRotatingFileHandler
        has_rotating_handler = any(
            isinstance(h, TimedRotatingFileHandler) for h in logger.handlers
        )
        assert not has_rotating_handler


class TestDirectoryManagement:
    """Test log directory creation and management."""

    def test_ensure_log_directories(self, tmp_path, monkeypatch):
        """Test directory creation."""
        monkeypatch.chdir(tmp_path)
        project_root = tmp_path

        # Mock get_project_root
        from AgentQMS.middleware import logging_config

        monkeypatch.setattr(
            logging_config, "get_project_root", lambda: project_root
        )

        ensure_log_directories()

        assert (project_root / "outputs" / "logs" / "middleware").exists()
        assert (project_root / "outputs" / "logs" / "audit").exists()
        assert (project_root / "outputs" / "metrics").exists()

    def test_existing_directories(self, tmp_path, monkeypatch):
        """Test handling of existing directories."""
        monkeypatch.chdir(tmp_path)
        project_root = tmp_path

        # Create directories first
        (project_root / "outputs" / "logs" / "middleware").mkdir(parents=True)

        from AgentQMS.middleware import logging_config

        monkeypatch.setattr(
            logging_config, "get_project_root", lambda: project_root
        )

        # Should not raise
        ensure_log_directories()


class TestLogOutput:
    """Test actual log file output."""

    def test_log_file_creation(self, tmp_path):
        """Test log file is created on write."""
        logger = setup_middleware_logger("test", output_dir=tmp_path)
        logger.info("Test log message")

        log_file = tmp_path / "test.log"
        assert log_file.exists()

    def test_log_format_parsing(self, tmp_path):
        """Test output is valid JSON."""
        logger = setup_middleware_logger("test", output_dir=tmp_path)
        logger.info("Test message", extra={"extra": {"key": "value"}})

        log_file = tmp_path / "test.log"
        with log_file.open() as f:
            line = f.readline()
            data = json.loads(line)  # Should not raise

        assert data["message"] == "Test message"
        assert data["key"] == "value"

    def test_multiple_log_messages(self, tmp_path):
        """Test multiple messages in sequence."""
        logger = setup_middleware_logger("test", output_dir=tmp_path)

        logger.info("Message 1")
        logger.warning("Message 2")
        logger.error("Message 3")

        log_file = tmp_path / "test.log"
        with log_file.open() as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Parse each line
        for line in lines:
            json.loads(line)  # Should not raise
