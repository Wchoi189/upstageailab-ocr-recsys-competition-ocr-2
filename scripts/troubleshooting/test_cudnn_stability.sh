#!/bin/bash
# Test cuDNN stability without wandb to isolate the issue

set -e

echo "=========================================="
echo "Testing cuDNN Stability (No Wandb)"
echo "=========================================="

# Disable cuDNN benchmark/autotuning
export CUDNN_BENCHMARK=0
export TORCH_CUDNN_V8_API_ENABLED=1

echo ""
echo "Running training with wandb DISABLED to test cuDNN stability..."
echo ""

uv run python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  exp_name=test-cudnn-stability \
  model.encoder.model_name=resnet18 \
  dataloaders.train_dataloader.batch_size=4 \
  trainer.max_steps=50 \
  logger.wandb.enabled=false \
  +skip_test=true \
  seed=42

echo ""
echo "=========================================="
echo "cuDNN stability test passed ✅"
echo "=========================================="
