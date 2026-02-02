"""Logging configuration for AgentQMS middleware.

This module provides centralized logging infrastructure for middleware components
with structured JSON logging, log rotation, and professional observability standards.
"""

import json
import logging
import traceback
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Custom formatter for structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data)


def setup_middleware_logger(
    name: str,
    level: int = logging.INFO,
    output_dir: Path | None = None,
    enable_rotation: bool = True,
) -> logging.Logger:
    """Setup structured logger for middleware component.

    Args:
        name: Logger name (e.g., 'policies', 'telemetry')
        level: Logging level (default: INFO)
        output_dir: Output directory for logs (default: outputs/logs/middleware)
        enable_rotation: Enable log rotation (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"agentqms.middleware.{name}")

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Determine output directory
    if output_dir is None:
        # Default to project_root/outputs/logs/middleware/
        from AgentQMS.tools.utils.paths import get_project_root

        output_dir = get_project_root() / "outputs" / "logs" / "middleware"

    # Create directory structure
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Fallback to console logging if directory creation fails
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
            )
        )
        logger.addHandler(console_handler)
        logger.warning(
            f"Failed to create log directory {output_dir}: {e}. "
            "Falling back to console logging."
        )
        return logger

    # Setup file handler with rotation
    log_file = output_dir / f"{name}.log"

    if enable_rotation:
        # Daily rotation, keep 30 days
        handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8"
        )
    else:
        handler = logging.FileHandler(log_file, encoding="utf-8")

    # Use JSON formatter
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    # Also add console handler for ERROR and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(
        logging.Formatter("[%(levelname)s] [%(name)s] %(message)s")
    )
    logger.addHandler(console_handler)

    logger.setLevel(level)
    logger.propagate = False  # Don't propagate to root logger

    return logger


def log_with_context(
    logger: logging.Logger, level: int, message: str, **context: Any
) -> None:
    """Log message with additional context.

    Args:
        logger: Logger instance
        level: Log level (e.g., logging.INFO)
        message: Log message
        **context: Additional context fields to include in log
    """
    # Create a LogRecord with extra context
    extra = {"extra": context}
    logger.log(level, message, extra=extra)


def ensure_log_directories() -> None:
    """Ensure all required log directories exist.

    Creates:
        - outputs/logs/middleware/
        - outputs/logs/audit/
        - outputs/metrics/
    """
    from AgentQMS.tools.utils.paths import get_project_root

    project_root = get_project_root()

    directories = [
        project_root / "outputs" / "logs" / "middleware",
        project_root / "outputs" / "logs" / "audit",
        project_root / "outputs" / "metrics",
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # Use basic logging as fallback
            logging.warning(f"Failed to create directory {directory}: {e}")


# Initialize directories on module import
ensure_log_directories()
