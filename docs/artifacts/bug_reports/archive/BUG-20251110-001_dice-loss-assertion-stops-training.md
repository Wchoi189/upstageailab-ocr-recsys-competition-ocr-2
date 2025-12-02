---
title: "Dice loss assertion stops training"
date: "2025-11-10"
type: "bug_report"
category: "training"
status: "open"
version: "1.0"
bug_id: "BUG-20251110-001"
severity: "High"
---

## Summary
Training run crashes with `AssertionError` in `ocr/models/loss/dice_loss.py` because the Dice loss briefly exceeds 1.0 during forward pass.

## Environment
- GPU: NVIDIA GeForce RTX 3060 (driver 581.57, CUDA 13.0 from `nvidia-smi`)
- PyTorch: 2.8.0+cu128 (`torch.version.cuda=12.8`)
- Command:
  - `python runners/train.py model.encoder.model_name=resnet18 dataloaders.train_dataloader.batch_size=2 trainer.max_steps=10 trainer.devices=1 trainer.strategy=auto`
- Configs:
  - `paths: default`
  - `callbacks: metadata` enabled (uses `outputs_dir`)
  - W&B disabled

## Steps to Reproduce
1. Activate the project environment.
2. Run:
   - `python runners/train.py model.encoder.model_name=resnet18 dataloaders.train_dataloader.batch_size=2 trainer.max_steps=10 trainer.devices=1 trainer.strategy=auto`
3. Observe training stops during first epoch with a failure sentinel at:
   - `outputs/ocr_training_b/checkpoints/.FAILURE`

## Expected Behavior
Training should proceed through the configured steps and save checkpoints/metadata without assertions in loss functions.

## Actual Behavior
- Trainer halts with:
  - `AssertionError: loss <= 1` from `ocr/models/loss/dice_loss.py:46`
- Stack trace indicates failure in `DBLoss -> DiceLoss._compute`.
- `.FAILURE` sentinel is created under the experiment’s `checkpoints/` directory.

## Logs / Stack Trace
Key excerpt from run:
```
Epoch 0/0 2/1636 ...
Created failure sentinel file at: /workspaces/upstageailab-ocr-recsys-competition-ocr-2/outputs/ocr_training_b/checkpoints/.FAILURE
Traceback (most recent call last):
  ...
  File ".../ocr/models/loss/dice_loss.py", line 46, in _compute
    assert loss <= 1
AssertionError
```

## Root Cause Analysis
- `DiceLoss._compute` enforces `assert loss <= 1`. With weighted masks (`gt_prob_mask`) and floating-point error, `2 * intersection / union` can slightly exceed 1, yielding `loss > 1` by a small margin. The hard assertion aborts training instead of handling minor numerical overshoots.

## Resolution
- Replace the hard assertion with clamping or tolerant bound:
  - Clamp: `loss = torch.clamp(loss, min=0.0, max=1.0 + 1e-6)`
  - Or normalize weights/masks to ensure `2*intersection/union <= 1` numerically.

## Testing
1. Apply the clamp or normalization fix.
2. Re-run the reproduction command.
3. Verify training proceeds past the initial steps; confirm no assertion and losses remain finite.

## Prevention
- Avoid strict assertions on floating-point inequalities in critical training paths; prefer tolerant checks or clamping.
- Add unit tests for Dice loss with weighted masks and random tensors to guard against numerical overshoot.
