---
ads_version: 1.0
type: bug_report
category: troubleshooting
status: active
version: 1.0
severity: high
description: CharErrorRate metric never reset between epochs causing 0.0 checkpoint scores and accuracy drop on resume
tags: checkpoint,metrics,recognition,training
title: Checkpoint Metric Accumulation Bug
date: 2026-02-18 02:37 (KST)
branch: 003-ocr-data-quality-remediation
---

# Bug Report - Checkpoint Metric Accumulation Bug
Bug ID: BUG-20260218-001

## Summary

The `CharErrorRate` metric in the recognition module is never reset between validation epochs, causing:
1. Checkpoints saved with incorrect 0.0 scores
2. Significant accuracy drop when resuming from checkpoints (e.g., 83% → 77%)
3. Metric accumulation across all epochs instead of per-epoch computation

## Environment

- **OS**: Linux
- **Python**: 3.11
- **Framework**: PyTorch Lightning
- **Module**: `ocr.domains.recognition.module.RecognitionPLModule`
- **Metric**: `torchmetrics.text.CharErrorRate`

## Reproduction

### Issue 1: Checkpoint with 0.0 Score

1. Start training from scratch
2. Checkpoint saved at epoch 0, step 1
3. Inspect checkpoint:
```bash
uv run python -c "
import torch
ckpt = torch.load('outputs/checkpoints/best-v3.ckpt', map_location='cpu')
print(f'Epoch: {ckpt[\"epoch\"]}, Score: {ckpt[\"callbacks\"][...][\"best_model_score\"]}')
"
# Output: Epoch: 0, Score: 0.0
```

### Issue 2: Accuracy Drop on Resume

1. Train model to epoch 10, val/acc = 83%
2. Save checkpoint
3. Resume training from checkpoint
4. First validation shows val/acc = 77% (expected: ~83%)

## Comparison

| Scenario | Expected | Actual (Buggy) |
|----------|----------|----------------|
| Epoch 0 checkpoint score | Actual epoch 0 accuracy (e.g., 0.75) | 0.0 |
| Resume accuracy | Matches last epoch (83%) | Drops significantly (77%) |
| Metric scope | Current epoch only | Accumulates all epochs |
| `rec_cer` state | Reset each epoch | Never reset |

## Root Cause Analysis

### Bug #1: Metric Never Reset (CRITICAL)

**Location**: `ocr/domains/recognition/module.py`

```python
# BEFORE (buggy)
def on_validation_epoch_start(self):
    if self._high_loss_audit_enabled():
        self._clear_high_loss_epoch_buffer()
    # BUG: Missing self.rec_cer.reset()

def on_validation_epoch_end(self):
    self._log_high_loss_audit()
    # BUG: Missing self.rec_cer.reset()
```

**Impact**: Metric state accumulates across ALL epochs, causing:
- Stale metric state on checkpoint resume
- Diluted accuracy values (old + new data averaged)
- Incorrect checkpoint scores

### Bug #2: Missing `on_epoch=True` Flag

```python
# BEFORE (buggy)
self.log("val/acc", batch_acc, batch_size=len(pred_texts), prog_bar=True)

# AFTER (fixed)
self.log("val/acc", batch_acc, batch_size=len(pred_texts), prog_bar=True, on_epoch=True, sync_dist=True)
```

**Impact**: Lightning doesn't properly aggregate batch metrics into epoch metrics.

### Bug #3: Missing `_checkpoint_metrics`

Recognition module doesn't set `_checkpoint_metrics` like detection module does.

**Impact**: CheckpointHandler has no explicit metrics to save.

## Logs

### Checkpoint Analysis (Buggy)
```
=== best-v3.ckpt ===
Epoch: 0
Global step: 1
best_model_score: 0.0
current_score: 0.0
best_k_models:
  best-v3.ckpt: 0.0
```

### Resume Behavior (Buggy)
```
Epoch 10: val/acc = 0.8300 (checkpoint saved)
--- Resume from checkpoint ---
Epoch 11: val/acc = 0.7700 (WRONG - should be ~0.83)
```

## Impact

### Severity: HIGH

1. **Training Quality**: Checkpoint selection based on wrong scores
2. **Resource Waste**: Training resumes from suboptimal checkpoints
3. **Debugging Difficulty**: Inconsistent metrics confuse developers
4. **Reproducibility**: Results vary based on resume history

### Affected Components

| Component | Status |
|-----------|--------|
| Recognition training | ✅ Fixed |
| Detection training | ✅ Already correct |
| Checkpoint saving | ✅ Fixed |
| Checkpoint resume | ✅ Fixed |

## Fix Applied

**File**: `ocr/domains/recognition/module.py`

### Change 1: Reset metrics at epoch start/end
```python
def on_validation_epoch_start(self):
    """Reset metrics at start of validation epoch."""
    self.rec_cer.reset()  # ← ADDED
    if self._high_loss_audit_enabled():
        self._clear_high_loss_epoch_buffer()

def on_validation_epoch_end(self):
    self._log_high_loss_audit()
    # Set checkpoint metrics for consistent checkpoint saving
    if hasattr(self, "trainer") and self.trainer:
        self._checkpoint_metrics = {
            "val/acc": float(self.trainer.callback_metrics.get("val/acc", 0.0)),
            "val/cer": float(self.trainer.callback_metrics.get("val/cer", 0.0)),
        }
    self.rec_cer.reset()  # ← ADDED
```

### Change 2: Add proper logging flags
```python
def _compute_metrics(self, pred_texts, gt_texts):
    self.rec_cer(pred_texts, gt_texts)
    matches = sum([1 for p, g in zip(pred_texts, gt_texts, strict=True) if p == g])
    batch_acc = matches / len(pred_texts) if len(pred_texts) > 0 else 0.0

    # Added on_epoch=True, sync_dist=True
    self.log("val/acc", batch_acc, batch_size=len(pred_texts), prog_bar=True, on_epoch=True, sync_dist=True)
    self.log("val/cer", self.rec_cer, batch_size=len(pred_texts), prog_bar=True, on_epoch=True, sync_dist=True)
```

## Verification

```bash
# 1. Verify metric reset is working
uv run python scripts/debug/checkpoint_debug.py

# 2. Start new training and verify epoch 0 has valid score
python train.py ...

# 3. Resume and verify accuracy matches
python train.py trainer.checkpoint_path=outputs/.../best-acc-*.ckpt
# Expected: Resume accuracy ≈ Last epoch accuracy
```

## Related Artifacts

- `/workspaces/docs/checkpoint_score_bug_fix.md` - Detailed analysis
- `/workspaces/docs/checkpoint_management.md` - Checkpoint management guide
- `/workspaces/docs/checkpoint_fix_summary.md` - Summary of all fixes

## References

- Detection module (correct implementation): `ocr/domains/detection/module.py:198`
- TorchMetrics documentation: https://torchmetrics.readthedocs.io/
- PyTorch Lightning metric logging: https://lightning.ai/docs/pytorch/
