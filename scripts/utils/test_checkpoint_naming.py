#!/usr/bin/env python3
"""
Test checkpoint naming with metric scores.

This script verifies that the UniqueModelCheckpoint callback correctly
formats checkpoint filenames with metric scores and sanitizes special characters.

Usage:
    uv run python scripts/utils/test_checkpoint_naming.py
"""

import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ocr.core.lightning.callbacks.unique_checkpoint import UniqueModelCheckpoint


class MockTrainer:
    """Mock trainer for testing."""

    def __init__(self, epoch=10, global_step=1000):
        self.current_epoch = epoch
        self.global_step = global_step


def test_checkpoint_naming():
    """Test various checkpoint naming scenarios."""
    print("=" * 70)
    print("Checkpoint Naming Tests")
    print("=" * 70)

    # Create callback with auto_insert_metric_name=True
    callback = UniqueModelCheckpoint(
        dirpath="/tmp/test_checkpoints",
        filename="best",
        monitor="val/acc",
        mode="max",
        auto_insert_metric_name=True,
    )

    # Attach mock trainer
    callback.trainer = MockTrainer(epoch=36, global_step=188660)

    print("\nTest 1: Best checkpoint with float metric")
    print("-" * 70)
    metrics = {"val/acc": 0.83006}  # Float type (most common)
    filename = callback.format_checkpoint_name(metrics, "best", ver=None)
    print(f"Monitor: {callback.monitor}")
    print(f"Metrics: {metrics}")
    print(f"Result: {filename}")
    expected = "/tmp/test_checkpoints/best-acc-0.8301.ckpt"
    status = "✓ PASS" if "best-acc-0.8301" in filename else "✗ FAIL"
    print(f"Expected: {expected}")
    print(f"Status: {status}\n")

    print("\nTest 2: Best checkpoint with Tensor metric")
    print("-" * 70)
    metrics = {"val/acc": torch.tensor(0.83725)}  # Tensor type
    filename = callback.format_checkpoint_name(metrics, "best", ver=None)
    print(f"Monitor: {callback.monitor}")
    print(f"Metrics: {metrics}")
    print(f"Result: {filename}")
    expected = "/tmp/test_checkpoints/best-acc-0.8373.ckpt"
    status = "✓ PASS" if "best-acc-0.8373" in filename else "✗ FAIL"
    print(f"Expected: {expected}")
    print(f"Status: {status}\n")

    print("\nTest 3: Best checkpoint with version number")
    print("-" * 70)
    metrics = {"val/acc": 0.83450}
    filename = callback.format_checkpoint_name(metrics, "best", ver=2)
    print(f"Monitor: {callback.monitor}")
    print(f"Metrics: {metrics}")
    print(f"Result: {filename}")
    expected = "/tmp/test_checkpoints/best-acc-0.8345-v2.ckpt"
    status = "✓ PASS" if "best-acc-0.8345" in filename and "v2" in filename else "✗ FAIL"
    print(f"Expected: {expected}")
    print(f"Status: {status}\n")

    print("\nTest 4: Last checkpoint (no metric)")
    print("-" * 70)
    metrics = {"val/acc": 0.82750}
    filename = callback.format_checkpoint_name(metrics, "last", ver=None)
    print(f"Result: {filename}")
    expected = "/tmp/test_checkpoints/last.ckpt"
    status = "✓ PASS" if filename == expected else "✗ FAIL"
    print(f"Expected: {expected}")
    print(f"Status: {status}\n")

    print("\nTest 5: Special characters sanitization")
    print("-" * 70)
    callback_special = UniqueModelCheckpoint(
        dirpath="/tmp/test",
        filename="best",
        monitor="val/hmean",  # Contains "/"
        mode="max",
        auto_insert_metric_name=True,
    )
    callback_special.trainer = MockTrainer(epoch=10, global_step=5000)
    metrics = {"val/hmean": 0.89234}
    filename = callback_special.format_checkpoint_name(metrics, "best", ver=None)
    print(f"Monitor: val/hmean (contains '/')")
    print(f"Result: {filename}")
    expected = "/tmp/test/best-hmean-0.8923.ckpt"
    status = "✓ PASS" if "best-hmean-0.8923" in filename else "✗ FAIL"
    print(f"Expected: {expected}")
    print(f"Status: {status}\n")

    print("\nTest 6: Epoch checkpoint with metric")
    print("-" * 70)
    metrics = {"val/acc": 0.81500}
    callback.trainer.current_epoch = 5
    callback.trainer.global_step = 10000
    filename = callback.format_checkpoint_name(metrics, "epoch", ver=None)
    print(f"Epoch: 5, Step: 10000")
    print(f"Result: {filename}")
    expected_contains = ["epoch-05", "step-010000", "acc-0.8150"]
    status = "✓ PASS" if all(s in filename for s in expected_contains) else "✗ FAIL"
    print(f"Expected to contain: {', '.join(expected_contains)}")
    print(f"Status: {status}\n")

    print("=" * 70)
    print("Tests Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_checkpoint_naming()
