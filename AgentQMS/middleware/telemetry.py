"""Telemetry Middleware for AgentQMS.

This module provides the infrastructure for "Agent-In-the-Loop" feedback by intercepting
tool calls, validating them against active policies, and potentially rejecting them
with constructive feedback.
"""
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .logging_config import setup_middleware_logger

# Initialize logger for telemetry
logger = setup_middleware_logger("telemetry")


class PolicyViolation(Exception):
    """Exception raised when a tool call violates an active policy.

    Attributes:
        message (str): Internal log message describing the violation.
        feedback_to_ai (str): Message to be returned to the AI agent.
    """

    def __init__(self, message: str, feedback_to_ai: str):
        super().__init__(message)
        self.feedback_to_ai = feedback_to_ai


@runtime_checkable
class Interceptor(Protocol):
    """Protocol for telemetry interceptors."""

    def validate(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Validate the tool call.

        Args:
            tool_name: The name of the tool being called.
            arguments: The arguments passed to the tool.

        Raises:
            PolicyViolation: If the validation fails.
        """
        ...


class TelemetryPipeline:
    """Pipeline for running a sequence of interceptors with metrics collection."""

    def __init__(self, interceptors: list[Interceptor] | None = None) -> None:
        self.interceptors = interceptors or []
        # Metrics tracking
        self.metrics = {
            "checks": defaultdict(int),
            "violations": defaultdict(int),
            "exceptions": defaultdict(int),
            "durations": defaultdict(list),
        }

    def validate(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Run all interceptors against the tool call.

        Args:
            tool_name: The name of the tool being called.
            arguments: The arguments passed to the tool.

        Raises:
            PolicyViolation: If any interceptor fails.
        """
        for interceptor in self.interceptors:
            policy_name = interceptor.__class__.__name__
            self.metrics["checks"][policy_name] += 1

            start_time = time.perf_counter()
            try:
                interceptor.validate(tool_name, arguments)
            except PolicyViolation as e:
                # Track violations
                self.metrics["violations"][policy_name] += 1
                logger.info(
                    "Policy violation detected",
                    extra={
                        "extra": {
                            "policy": policy_name,
                            "tool_name": tool_name,
                            "violation": e.feedback_to_ai,
                        }
                    },
                )
                raise
            except Exception as e:
                # Track unexpected exceptions
                self.metrics["exceptions"][type(e).__name__] += 1
                logger.error(
                    f"Unexpected exception in policy {policy_name}",
                    extra={
                        "extra": {
                            "policy": policy_name,
                            "tool_name": tool_name,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        }
                    },
                )
                raise
            finally:
                # Track duration
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics["durations"][policy_name].append(duration_ms)

    def export_metrics(self, output_path: Path | None = None) -> None:
        """Export collected metrics to JSON file.

        Args:
            output_path: Path to export metrics (default: outputs/metrics/middleware_stats.json)
        """
        if output_path is None:
            from AgentQMS.tools.utils.paths import get_project_root

            output_path = (
                get_project_root() / "outputs" / "metrics" / "middleware_stats.json"
            )

        # Calculate statistics
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "period": "session",
            "policies": {},
            "exceptions": dict(self.metrics["exceptions"]),
        }

        for policy_name in self.metrics["checks"].keys():
            durations = self.metrics["durations"][policy_name]
            avg_duration = sum(durations) / len(durations) if durations else 0

            stats["policies"][policy_name] = {
                "checks": self.metrics["checks"][policy_name],
                "violations": self.metrics["violations"][policy_name],
                "exceptions": sum(
                    1 for exc in self.metrics["exceptions"] if policy_name in str(exc)
                ),
                "avg_duration_ms": round(avg_duration, 2),
            }

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write metrics file
        try:
            with output_path.open("w") as f:
                json.dump(stats, f, indent=2)
            logger.debug(f"Exported metrics to {output_path}")
        except Exception as e:
            logger.error(
                "Failed to export metrics",
                extra={"extra": {"error": str(e), "output_path": str(output_path)}},
            )

