#!/bin/bash
# Minimal test with batch_size=1 and reduced complexity

echo "=========================================="
echo "Minimal CUDA Test (Batch Size 1)"
echo "=========================================="
echo ""

# Clear CUDA cache first
echo "Clearing CUDA cache..."
uv run python -c "import torch; torch.cuda.empty_cache(); print('✓ CUDA cache cleared')"
echo ""

# Test with minimal configuration
echo "Running training with minimal batch size..."
echo "  - Batch size: 1"
echo "  - Workers: 0 (no multiprocessing)"
echo "  - Epochs: 1"
echo "  - Steps: will stop after 10 successful steps"
echo ""

CUDA_LAUNCH_BLOCKING=1 uv run python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  exp_name=test-minimal-batch \
  logger.wandb.enabled=false \
  model/architectures=dbnet \
  model.encoder.model_name=resnet50 \
  model.component_overrides.decoder.name=pan_decoder \
  dataloaders.train_dataloader.batch_size=1 \
  dataloaders.train_dataloader.num_workers=0 \
  dataloaders.val_dataloader.batch_size=1 \
  dataloaders.val_dataloader.num_workers=0 \
  trainer.max_epochs=1 \
  trainer.max_steps=10 \
  trainer.precision=32 \
  seed=42

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Test PASSED with batch_size=1"
    echo ""
    echo "The issue is likely:"
    echo "  1. Batch size too large for RTX 3060 12GB"
    echo "  2. cuDNN workspace memory exhaustion"
    echo ""
    echo "Solution: Use batch_size=1 or 2, and increase accumulate_grad_batches"
    echo "Example: batch_size=2, accumulate_grad_batches=2 (effective batch=4)"
else
    echo ""
    echo "✗ Test FAILED even with batch_size=1"
    echo ""
    echo "The issue is deeper - likely:"
    echo "  1. cuDNN version incompatibility"
    echo "  2. CUDA driver issue"
    echo "  3. Model architecture issue"
    echo ""
    echo "Next steps:"
    echo "  1. Check CUDA/cuDNN versions match PyTorch build"
    echo "  2. Try different encoder (e.g., resnet18 instead of resnet50)"
    echo "  3. Verify GPU is functioning correctly"
fi
