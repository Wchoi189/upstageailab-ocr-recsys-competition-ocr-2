#!/bin/bash
# Optimization Run Script
# Goal: Run 10 epochs with optimizations enabled (Workers=12, Batch=160, BF16)

# PREVIOUS_CKPT="outputs/checkpoints/last.ckpt"  # Uncomment to resume
PREVIOUS_CKPT="outputs/checkpoints/last-v2.ckpt"

MAX_EPOCHS=10  # Run in small chunks

# Source bashrc to get WANDB_API_KEY if not present
if [ -z "$WANDB_API_KEY" ]; then
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    fi
fi

# Explicit check
if [ -z "$WANDB_API_KEY" ]; then
    echo "❌ Error: WANDB_API_KEY is not set."
    echo "   Please export it or add it to ~/.bashrc"
    exit 1
fi

echo "🔑 Logging into WandB..."
uv run wandb login --relogin "$WANDB_API_KEY"

# Navigate to project root
cd ../../ || exit 1

echo "🚀 Starting Optimized Training (Chunk: 10 Epochs)"
echo "Config: configs/experiment/rec_optimized_rtx3090.yaml"

uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_optimized_rtx3090 \
  run_name="skilled-firebrand-627" \
  trainer.max_epochs=${MAX_EPOCHS} \
  ${PREVIOUS_CKPT:+"+checkpoint_path=$PREVIOUS_CKPT"} \
  +train/logger=wandb

echo "✅ Run Complete. Check WandB/Logs for 'it/s' improvement (>10 it/s target)."
