"""Telemetry Middleware for AgentQMS.

This module provides the infrastructure for "Agent-In-the-Loop" feedback by intercepting
tool calls, validating them against active policies, and potentially rejecting them
with constructive feedback.
"""
from typing import Any, Protocol, runtime_checkable

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
    """Pipeline for running a sequence of interceptors."""

    def __init__(self, interceptors: list[Interceptor] | None = None) -> None:
        self.interceptors = interceptors or []

    def validate(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Run all interceptors against the tool call.

        Args:
            tool_name: The name of the tool being called.
            arguments: The arguments passed to the tool.

        Raises:
            PolicyViolation: If any interceptor fails.
        """
        for interceptor in self.interceptors:
            interceptor.validate(tool_name, arguments)
