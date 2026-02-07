#!/bin/bash
# Quick training test with limited batches for fast verification
# Uses GPU and limits train/val batches to avoid 20-hour epochs

set -e

echo "================================================"
echo "Quick Training Test (GPU, Limited Batches)"
echo "================================================"

cd /workspaces

uv run python scripts/runners/train.py \
  experiment=rec_baseline_v1 \
  hardware=rtx3090 \
  trainer.limit_train_batches=50 \
  trainer.max_epochs=200 \
  +trainer.overfit_batches=1 \
  trainer.check_val_every_n_epoch=10 \
  trainer.log_every_n_steps=1 \
  trainer.accelerator=gpu \
  trainer.devices=[0] \
  trainer.precision=32 \
  train.optimizer.lr=1e-3 \
  data.num_workers=4 \
  trainer.num_sanity_val_steps=0 \
  trainer.gradient_clip_val=1.0

echo ""
echo "✅ Training test completed successfully!"
