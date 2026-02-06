#!/usr/bin/env python3
"""
State File Utilities for Ultra-Concise State Management

Provides atomic read/write operations for 100-byte .state files.
Format: line-delimited key=value pairs (no YAML parser needed).

Design goals:
- <1ms read performance
- <5ms atomic write performance
- 100-byte target size for core state
- Zero-parse workflow for AI agents

Ref: /home/vscode/.gemini/antigravity/brain/.../state_management_recommendations.md
"""

from datetime import UTC, datetime
from pathlib import Path


def read_state(exp_path: Path) -> dict[str, str]:
    """
    Read experiment state from .state file.

    Performance: O(1) file reads, O(n) parsing where n=5 fields.
    Target: <1ms

    Args:
        exp_path: Path to experiment directory

    Returns:
        dict with state fields (experiment_id, status, current_task, current_phase, last_updated)
        Empty dict if file doesn't exist

    Example:
        >>> state = read_state(Path("experiments/20251219_exp"))
        >>> state['current_task']
        'integrate_pipeline'
    """
    state_file = exp_path / ".state"

    if not state_file.exists():
        return {}

    state = {}
    content = state_file.read_text(encoding="utf-8").strip()

    for line in content.split("\n"):
        line = line.strip()
        if not line or "=" not in line:
            continue

        key, value = line.split("=", 1)
        state[key.strip()] = value.strip()

    return state


def update_state(exp_path: Path, **updates) -> None:
    """
    Atomically update state file fields.

    Performance: <5ms including atomic write

    Args:
        exp_path: Path to experiment directory
        **updates: Key-value pairs to update

    Raises:
        FileNotFoundError: If experiment directory doesn't exist

    Example:
        >>> update_state(exp_path, current_task="new_task", current_phase="testing")
    """
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_path}")

    # Read existing state
    state = read_state(exp_path)

    # Apply updates
    state.update(updates)

    # Always update timestamp
    state["last_updated"] = datetime.now(UTC).isoformat()

    # Atomic write via temp file
    state_file = exp_path / ".state"
    temp_file = exp_path / ".state.tmp"

    # Write to temp file
    lines = [f"{k}={v}" for k, v in state.items()]
    temp_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Atomic rename (POSIX guarantees atomicity)
    temp_file.replace(state_file)


def create_state_file(exp_path: Path, experiment_id: str, checkpoint_path: str | None = None) -> None:
    """
    Create initial .state file for new experiment.

    Args:
        exp_path: Path to experiment directory
        experiment_id: Unique experiment identifier (YYYYMMDD_HHMMSS_name)
        checkpoint_path: Optional path to initial checkpoint

    Example:
        >>> create_state_file(exp_path, "20251219_2100_test_experiment")
    """
    if not exp_path.exists():
        exp_path.mkdir(parents=True, exist_ok=True)

    state_file = exp_path / ".state"

    if state_file.exists():
        raise FileExistsError(f"State file already exists: {state_file}")

    now = datetime.now(UTC).isoformat()

    # Define 5 required fields (target: ~100 bytes)
    initial_state = {
        "experiment_id": experiment_id,
        "status": "active",
        "current_task": "",  # Empty until first task assigned
        "current_phase": "planning",
        "last_updated": now,
    }

    # Optional: add checkpoint if provided
    if checkpoint_path:
        initial_state["checkpoint_path"] = checkpoint_path

    # Write initial state
    lines = [f"{k}={v}" for k, v in initial_state.items()]
    state_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_state_size(exp_path: Path) -> int:
    """
    Get size of .state file in bytes.

    Target: <150 bytes (goal is ~100 bytes)

    Returns:
        File size in bytes, 0 if file doesn't exist
    """
    state_file = exp_path / ".state"

    if not state_file.exists():
        return 0

    return state_file.stat().st_size


def validate_state(state: dict[str, str]) -> bool:
    """
    Validate that state dict has required fields.

    Required fields:
    - experiment_id
    - status (must be: active/completed/failed/paused)
    - current_task (can be empty string)
    - current_phase
    - last_updated (ISO8601 format)

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["experiment_id", "status", "current_task", "current_phase", "last_updated"]

    # Check all required fields exist
    if not all(field in state for field in required_fields):
        return False

    # Validate status enum
    valid_statuses = {"active", "completed", "failed", "paused", "deprecated"}
    if state["status"] not in valid_statuses:
        return False

    # Validate experiment_id format (YYYYMMDD_HHMMSS_name)
    exp_id = state["experiment_id"]
    if not exp_id or len(exp_id) < 17:  # Minimum: YYYYMMDD_HHMMSS_x
        return False

    return True
