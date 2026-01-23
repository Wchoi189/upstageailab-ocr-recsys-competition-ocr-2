"""
Project Compass - Router Module

Implements the Router/Meta-Tool Pattern to reduce tool proliferation.
Instead of exposing 9 individual tools, we expose 2 meta-tools:
- compass_meta_pulse: Routes to pulse management tools
- compass_meta_spec: Routes to spec kit tools

This reduces model context burden while maintaining full functionality.
"""

from __future__ import annotations

from typing import Any


# Define available pulse kinds and their mappings
PULSE_KINDS = {
    "init": "pulse_init",
    "sync": "pulse_sync",
    "export": "pulse_export",
    "status": "pulse_status",
    "checkpoint": "pulse_checkpoint",
}

# Define available spec kinds and their mappings
SPEC_KINDS = {
    "constitution": "spec_constitution",
    "specify": "spec_specify",
    "plan": "spec_plan",
    "tasks": "spec_tasks",
}


def get_pulse_kinds() -> list[str]:
    """Get list of available pulse kinds."""
    return list(PULSE_KINDS.keys())


def get_spec_kinds() -> list[str]:
    """Get list of available spec kinds."""
    return list(SPEC_KINDS.keys())


def route_pulse(kind: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Route a pulse kind to its corresponding tool arguments.

    Args:
        kind: Pulse kind (e.g., "init", "sync")
        arguments: Arguments passed to the meta-tool

    Returns:
        Dict with tool_name and arguments
    """
    if kind not in PULSE_KINDS:
        raise ValueError(f"Unknown pulse kind: {kind}. Valid kinds: {list(PULSE_KINDS.keys())}")

    tool_name = PULSE_KINDS[kind]

    # For pulse tools, arguments are passed through directly
    # Each tool has its own validation in the call_tool handler
    return {
        "tool_name": tool_name,
        "arguments": arguments
    }


def route_spec(kind: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Route a spec kind to its corresponding tool arguments.

    Args:
        kind: Spec kind (e.g., "constitution", "specify")
        arguments: Arguments passed to the meta-tool

    Returns:
        Dict with tool_name and arguments
    """
    if kind not in SPEC_KINDS:
        raise ValueError(f"Unknown spec kind: {kind}. Valid kinds: {list(SPEC_KINDS.keys())}")

    tool_name = SPEC_KINDS[kind]

    # For spec tools, arguments are passed through directly
    # Each tool has its own validation in the call_tool handler
    return {
        "tool_name": tool_name,
        "arguments": arguments
    }
