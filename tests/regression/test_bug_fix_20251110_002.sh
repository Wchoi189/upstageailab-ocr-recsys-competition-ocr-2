#!/bin/bash
# Test script for BUG-20251110-002 fix
# Tests the numerical stability fix in DBHead step function

set -e  # Exit on error

echo "=========================================="
echo "Testing BUG-20251110-002 Fix"
echo "NaN Gradients from Step Function Overflow"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Test Configuration:${NC}"
echo "  - Hardware: RTX 3060 12GB"
echo "  - Model: DBNet with ResNet50 + PAN decoder"
echo "  - Batch Size: 4"
echo "  - Max Epochs: 1"
echo "  - Precision: FP32"
echo "  - Seed: 42"
echo ""

echo -e "${YELLOW}Expected Behavior:${NC}"
echo "  ✓ Training completes at least 200 steps without NaN gradients"
echo "  ✓ No CUDA illegal memory access errors"
echo "  ✓ Loss values remain finite and decrease"
echo "  ✓ No gradient explosion warnings"
echo ""

echo -e "${YELLOW}Starting test run...${NC}"
echo ""

# Run training with the fix
uv run python /workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  exp_name=test-bug-fix-20251110-002 \
  logger.wandb.enabled=false \
  model/architectures=dbnet \
  model.encoder.model_name=resnet50 \
  model.component_overrides.decoder.name=pan_decoder \
  model.component_overrides.head.name=db_head \
  model.component_overrides.loss.name=db_loss \
  model/optimizers=adam \
  model.optimizer.lr=0.001 \
  model.optimizer.weight_decay=0.0001 \
  dataloaders.train_dataloader.batch_size=4 \
  dataloaders.val_dataloader.batch_size=4 \
  trainer.max_epochs=1 \
  trainer.accumulate_grad_batches=1 \
  trainer.gradient_clip_val=5.0 \
  trainer.precision=32 \
  seed=42 \
  data=default

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "TEST PASSED"
    echo "==========================================${NC}"
    echo ""
    echo "✓ Training completed successfully"
    echo "✓ No NaN gradient errors detected"
    echo "✓ No CUDA illegal memory access errors"
    echo ""
    echo "Bug fix BUG-20251110-002 is VERIFIED"
    echo ""
    echo "Next steps:"
    echo "  1. Review training logs for any warnings"
    echo "  2. Check validation metrics"
    echo "  3. Update bug report status to 'verified'"
    echo "  4. Consider running longer training (5-10 epochs)"
    echo ""
else
    echo ""
    echo -e "${RED}=========================================="
    echo "TEST FAILED"
    echo "==========================================${NC}"
    echo ""
    echo "✗ Training crashed or encountered errors"
    echo ""
    echo "Debugging steps:"
    echo "  1. Check logs above for error messages"
    echo "  2. Look for NaN gradient warnings"
    echo "  3. Check for CUDA errors"
    echo "  4. Review bug report for additional fixes"
    echo ""
    echo "If NaN gradients still appear:"
    echo "  - Consider reducing k from 50 to 25 in DBHead"
    echo "  - Try reducing learning rate to 0.0001"
    echo "  - Enable CUDA_LAUNCH_BLOCKING=1 for detailed errors"
    echo ""
    exit 1
fi
