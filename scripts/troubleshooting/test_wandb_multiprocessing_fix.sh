#!/bin/bash
# Test script to verify wandb multiprocessing fix
# This should now work without CUDA errors

set -e

echo "=========================================="
echo "Testing Wandb Multiprocessing Fix"
echo "=========================================="

# Enable wandb and test with ResNet18 (lightweight model)
export CUDNN_BENCHMARK=0

echo ""
echo "Running training with wandb ENABLED..."
echo "This should now work without CUDA errors!"
echo ""

uv run python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  exp_name=test-wandb-spawn-fix \
  model.encoder.model_name=resnet18 \
  dataloaders.train_dataloader.batch_size=2 \
  trainer.max_steps=20 \
  logger.wandb.enabled=true \
  seed=42

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "Wandb multiprocessing fix verified ✅"
echo "=========================================="
