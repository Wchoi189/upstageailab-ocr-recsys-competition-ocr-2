#!/bin/bash
# Test forkserver multiprocessing fix with cuDNN optimizations disabled
# This should be faster than spawn while still being CUDA-safe

set -e

echo "=========================================="
echo "Testing Forkserver Multiprocessing Fix"
echo "=========================================="

# Disable cuDNN benchmark/autotuning to avoid cuDNN instability
export CUDNN_BENCHMARK=0
export TORCH_CUDNN_V8_API_ENABLED=1

echo ""
echo "Running training with:"
echo "  - Multiprocessing: forkserver (CUDA-safe, faster than spawn)"
echo "  - cuDNN benchmark: disabled (stability over speed)"
echo "  - Wandb: enabled"
echo "  - Skip test phase: yes (test training only)"
echo ""

uv run python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  exp_name=test-forkserver-fix \
  model.encoder.model_name=resnet18 \
  dataloaders.train_dataloader.batch_size=4 \
  trainer.max_steps=50 \
  logger.wandb.enabled=true \
  +skip_test=true \
  seed=42

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "Forkserver fix verified ✅"
echo "=========================================="
