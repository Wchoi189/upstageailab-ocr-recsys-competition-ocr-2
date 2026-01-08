---
category: planning
status: active
type: implementation_plan
date: 2024-10-01 10:00 (KST)
---

# Console Output Polish Plan

## Goal Description
Fix the "huge gap" in the training console output reported by the user. The current configuration uses a custom `MultiLineRichProgressBar` which is essentially a pass-through wrapper. Reverting to the standard `lightning.pytorch.callbacks.RichProgressBar` should eliminate any custom behavior issues and simplify the codebase.

## User Review Required
> [!NOTE]
> Replacing custom `MultiLineRichProgressBar` with standard `RichProgressBar`.

## Proposed Changes

### Configuration
#### [MODIFY] [configs/training/callbacks/rich_progress_bar.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/training/callbacks/rich_progress_bar.yaml)
- Change `_target_` from `ocr.lightning_modules.callbacks.multi_line_progress_bar.MultiLineRichProgressBar` to `lightning.pytorch.callbacks.RichProgressBar`.

### Code Cleanup
#### [DELETE] [ocr/lightning_modules/callbacks/multi_line_progress_bar.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/lightning_modules/callbacks/multi_line_progress_bar.py)
- Remove the unnecessary wrapper class.

## Verification Plan

### Manual Verification
1. **Training Dry Run**: Run the training command and observe the console output for gaps.
   ```bash
   UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
     domain=detection \
     data=canonical \
     batch_size=4 \
     trainer.max_epochs=1 \
     trainer.limit_train_batches=10 \
     logger.wandb.enabled=false
   ```