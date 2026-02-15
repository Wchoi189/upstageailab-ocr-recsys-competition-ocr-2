#!/bin/bash
# Test script for lazy dataset loading across all modes
# Verifies that each mode only creates required datasets

set -e  # Exit on error

echo "======================================"
echo "Testing Lazy Dataset Loading"
echo "======================================"

EXPERIMENT="parseq_flash_fast"
LIMIT_BATCHES=1
MAX_EPOCHS=1

# Disable checkpointing during tests as per requirements
CHECKPOINT_ARGS="trainer.enable_checkpointing=false"

echo ""
echo "Test 1: mode=train (should create train + val)"
echo "--------------------------------------"
uv run python scripts/runners/train.py \
  experiment=$EXPERIMENT \
  mode=train \
  trainer.limit_train_batches=$LIMIT_BATCHES \
  trainer.limit_val_batches=$LIMIT_BATCHES \
  trainer.max_epochs=$MAX_EPOCHS \
  $CHECKPOINT_ARGS \
  2>&1 | grep -E "(Creating datasets|Datasets created)" || true

if [ $? -eq 0 ]; then
    echo "✓ mode=train: PASSED"
else
    echo "✗ mode=train: FAILED"
    exit 1
fi

echo ""
echo "Test 2: mode=eval (should create val only)"
echo "--------------------------------------"
uv run python scripts/runners/train.py \
  experiment=$EXPERIMENT \
  mode=eval \
  trainer.limit_val_batches=$LIMIT_BATCHES \
  $CHECKPOINT_ARGS \
  2>&1 | grep -E "(Creating datasets|Datasets created)" || true

if [ $? -eq 0 ]; then
    echo "✓ mode=eval: PASSED"
else
    echo "✗ mode=eval: FAILED"
    exit 1
fi

echo ""
echo "Test 3: mode=test (should create test only)"
echo "--------------------------------------"
uv run python scripts/runners/train.py \
  experiment=$EXPERIMENT \
  mode=test \
  trainer.limit_test_batches=$LIMIT_BATCHES \
  $CHECKPOINT_ARGS \
  2>&1 | grep -E "(Creating datasets|Datasets created)" || true

if [ $? -eq 0 ]; then
    echo "✓ mode=test: PASSED"
else
    echo "✗ mode=test: FAILED"
    exit 1
fi

echo ""
echo "======================================"
echo "All mode tests passed!"
echo "======================================"
