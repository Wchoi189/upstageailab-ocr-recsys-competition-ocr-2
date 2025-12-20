#!/usr/bin/env python3
"""
Unit tests for state_file utilities.

Tests:
- Atomic read/write operations
- State file creation
- Field validation
- Performance benchmarks
"""

import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from experiment_tracker.utils.state_file import (
    create_state_file,
    get_state_size,
    read_state,
    update_state,
    validate_state,
)


@pytest.fixture
def temp_exp_dir():
    """Create temporary experiment directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_path = Path(tmpdir) / "20251219_2100_test_exp"
        exp_path.mkdir()
        yield exp_path


def test_create_state_file(temp_exp_dir):
    """Test initial state file creation."""
    experiment_id = "20251219_2100_test_experiment"
    create_state_file(temp_exp_dir, experiment_id)

    # Verify file exists
    state_file = temp_exp_dir / ".state"
    assert state_file.exists()

    # Verify content
    state = read_state(temp_exp_dir)
    assert state["experiment_id"] == experiment_id
    assert state["status"] == "active"
    assert state["current_task"] == ""
    assert state["current_phase"] == "planning"
    assert "last_updated" in state


def test_read_empty_state(temp_exp_dir):
    """Test reading state when file doesn't exist."""
    state = read_state(temp_exp_dir)
    assert state == {}


def test_update_state(temp_exp_dir):
    """Test atomic state updates."""
    experiment_id = "20251219_2100_test"
    create_state_file(temp_exp_dir, experiment_id)

    # Update fields
    update_state(temp_exp_dir, current_task="task_123", current_phase="execution")

    # Verify updates
    state = read_state(temp_exp_dir)
    assert state["current_task"] == "task_123"
    assert state["current_phase"] == "execution"
    assert state["experiment_id"] == experiment_id  # Unchanged


def test_state_file_size(temp_exp_dir):
    """Test that state file meets size target (<150 bytes)."""
    experiment_id = "20251219_2100_minimal_state"
    create_state_file(temp_exp_dir, experiment_id)

    size = get_state_size(temp_exp_dir)

    # Target: ~100 bytes, max acceptable: 150 bytes
    assert size > 0
    assert size < 150, f"State file too large: {size} bytes (target: <150)"


def test_validate_state_valid():
    """Test state validation with valid state."""
    state = {
        "experiment_id": "20251219_2100_test",
        "status": "active",
        "current_task": "task_1",
        "current_phase": "planning",
        "last_updated": datetime.now(UTC).isoformat(),
    }

    assert validate_state(state) is True


def test_validate_state_invalid_status():
    """Test state validation rejects invalid status."""
    state = {
        "experiment_id": "20251219_2100_test",
        "status": "invalid_status",  # Invalid
        "current_task": "task_1",
        "current_phase": "planning",
        "last_updated": datetime.now(UTC).isoformat(),
    }

    assert validate_state(state) is False


def test_validate_state_missing_fields():
    """Test state validation rejects missing fields."""
    state = {
        "experiment_id": "20251219_2100_test",
        "status": "active",
        # Missing current_task, current_phase, last_updated
    }

    assert validate_state(state) is False


def test_atomic_write_performance(temp_exp_dir):
    """Benchmark: state updates should complete in <5ms."""
    experiment_id = "20251219_2100_perf_test"
    create_state_file(temp_exp_dir, experiment_id)

    # Measure update time
    start = time.perf_counter()
    update_state(temp_exp_dir, current_task="benchmark_task")
    duration_ms = (time.perf_counter() - start) * 1000

    # Performance target: <5ms
    assert duration_ms < 5.0, f"Update too slow: {duration_ms:.2f}ms (target: <5ms)"


def test_read_performance(temp_exp_dir):
    """Benchmark: state reads should complete in <1ms."""
    experiment_id = "20251219_2100_read_test"
    create_state_file(temp_exp_dir, experiment_id)

    # Measure read time
    start = time.perf_counter()
    state = read_state(temp_exp_dir)
    duration_ms = (time.perf_counter() - start) * 1000

    # Performance target: <1ms
    assert duration_ms < 1.0, f"Read too slow: {duration_ms:.2f}ms (target: <1ms)"
    assert state["experiment_id"] == experiment_id


def test_create_with_checkpoint(temp_exp_dir):
    """Test state creation with checkpoint path."""
    experiment_id = "20251219_2100_ckpt_test"
    checkpoint = "/path/to/checkpoint.ckpt"

    create_state_file(temp_exp_dir, experiment_id, checkpoint_path=checkpoint)

    state = read_state(temp_exp_dir)
    assert state["checkpoint_path"] == checkpoint


def test_concurrent_updates_safety(temp_exp_dir):
    """Test that atomic writes prevent corruption."""
    experiment_id = "20251219_2100_concurrent_test"
    create_state_file(temp_exp_dir, experiment_id)

    # Simulate rapid updates (atomic write should handle this)
    for i in range(100):
        update_state(temp_exp_dir, current_task=f"task_{i}")

    # Verify final state is valid
    state = read_state(temp_exp_dir)
    assert validate_state(state)
    assert state["current_task"] == "task_99"
