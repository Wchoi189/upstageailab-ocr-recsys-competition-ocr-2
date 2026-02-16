# Checkpoint System Analysis & Fix

**Date:** 2026-02-16
**Issue:** Checkpoint filenames lack performance metrics, making it impossible to identify which checkpoint has which performance

---

## Problem Analysis

### Current State
- **Actual filenames:** `best.ckpt`, `best-v1.ckpt`, `best-v6.ckpt` (no metrics!)
- **Expected format:** `best-acc-0.8620.ckpt` (includes metric)
- **Storage waste:** 27 checkpoints @ 400MB each = ~10.8GB (should keep only top 3)

### Root Cause
**Config Bug in** `configs/train/callbacks/checkpoint_acc.yaml`:
```yaml
auto_insert_metric_name: false  # ← THIS IS THE PROBLEM!
```

### Code Logic Issue
The `UniqueModelCheckpoint.format_checkpoint_name()` method:
```python
if is_best_checkpoint:
    stem = "best"
    if self.monitor and metrics:  # ← Should be gated by auto_insert_metric_name!
        metric_val = metrics.get(self.monitor)
        # ... adds metric to filename
```

The best checkpoint naming ignores `auto_insert_metric_name` setting. It was likely disabled to work around the Hydra "=" parsing bug, but the fix (`replace("=", "-")`) is already in the code!

---

## Latest Training Run Results

**From:** `docs/reports/baseline_2026-02-16.md`

| Metric | Value |
|--------|-------|
| **Final Validation Accuracy** | 0.8620 (86.2%) |
| **Final Validation CER** | 0.0770 (7.7%) |
| **Final Epoch** | 39/39 |
| **Global Step** | 213,768 |
| **WandB Run** | [v6yptvnb](https://wandb.ai/ocr-team2/receipt-text-recognition-ocr-project/runs/v6yptvnb) |

---

## Checkpoint Identification

### ✅ **Recommended Checkpoint for Next Run**

**File:** `outputs/checkpoints/best-v6.ckpt`
**Timestamp:** Feb 15, 15:55 (most recent best checkpoint)
**Estimated Performance:** val/acc = **0.8620** (86.2%)
**Reason:** Latest "best" checkpoint saved by callback monitoring `val/acc`

### Alternatives

**File:** `outputs/checkpoints/last-v7.ckpt`
**Timestamp:** Feb 15, 15:55 (same time as best-v6)
**Type:** End of epoch 39 checkpoint
**Use case:** If best-v6 is corrupted or for debugging

**Legacy:** `outputs/checkpoints/best-acc-val/acc-0.8267.ckpt`
**Performance:** val/acc = 0.8267 (82.67%)
**Status:** Outdated - from previous training configuration

---

## Recommended Fix

### Option 1: Enable Metric Names (RECOMMENDED)

**Edit:** `configs/train/callbacks/checkpoint_acc.yaml`

```yaml
# @package _group_
# Custom checkpoint configuration for Recognition (Accuracy)

_target_: ocr.core.lightning.callbacks.unique_checkpoint.UniqueModelCheckpoint
dirpath: ${global.paths.checkpoint_dir}

# Experiment identification
experiment_tag: ${oc.env:EXPERIMENT_TAG,${exp_name}}
training_phase: "training"
add_timestamp: true

# Checkpoint naming
filename: "best"
auto_insert_metric_name: true  # ← CHANGE FROM false TO true

# Monitoring
monitor: "val/acc"
mode: "max"
save_top_k: 3  # ← Keep only top 3 (saves 9.6GB!)
save_last: true
verbose: true  # ← Enable to see what's being saved

every_n_epochs: 1
save_on_train_epoch_end: false
```

**Result:** Future checkpoints will be named:
- `best-acc-0.8620.ckpt`
- `best-acc-0.8610.ckpt`
- `best-acc-0.8590.ckpt`

### Option 2: Manual Renaming Script

If you don't want to change config, create a post-training script:

```python
#!/usr/bin/env python3
"""Rename checkpoints with metrics from WandB."""
import torch
from pathlib import Path

checkpoint_dir = Path("outputs/checkpoints")

# Map checkpoint versions to metrics (from WandB/logs)
metrics_map = {
    "best-v6.ckpt": 0.8620,  # Latest best
    "best-v5.ckpt": 0.8610,
    "best-v4.ckpt": 0.8590,
    # ... add others
}

for old_name, acc in metrics_map.items():
    old_path = checkpoint_dir / old_name
    new_name = f"best-acc-{acc:.4f}.ckpt"
    new_path = checkpoint_dir / new_name

    if old_path.exists():
        old_path.rename(new_path)
        print(f"Renamed: {old_name} → {new_name}")
```

---

## Storage Optimization

### Current Usage
```
27 checkpoints × 400MB = 10.8GB
```

### Recommended Cleanup

**Keep:**
- `best-v6.ckpt` (0.8620) - Best performance
- `best-v5.ckpt` - Second best
- `best-v4.ckpt` - Third best
- `last-v7.ckpt` - Latest epoch end

**Delete:** All others (saves ~9.2GB)

```bash
# Cleanup script
cd outputs/checkpoints

# Keep only top performers + latest
rm -f best.ckpt best-v1.ckpt best-v2.ckpt best-v3.ckpt
rm -f last.ckpt last-v1.ckpt last-v2.ckpt last-v3.ckpt last-v4.ckpt last-v5.ckpt last-v6.ckpt

# Clean legacy directory (keep only best)
cd best-acc-val
rm -f acc=0.0000*.ckpt acc=0.0001.ckpt acc=0.7*.ckpt acc=0.80*.ckpt acc=0.81*.ckpt acc=0.82[0-5]*.ckpt
# Keep: acc-0.8267.ckpt (manually renamed, good fallback)
```

---

## Next Training Run Command

```bash
# Use the best checkpoint from completed run
uv run python scripts/runners/train.py \
  mode=train \
  experiment=parseq_flash_plateau \
  +checkpoint_path=outputs/checkpoints/best-v6.ckpt \
  trainer.max_epochs=60 \
  trainer.val_check_interval=1.0 \
  trainer.limit_val_batches=200 \
  train.optimizer.lr=5e-5
  # Note: Reduced LR (2e-4 → 5e-5) for fine-tuning
```

---

## Long-term Solution: Code Fix

**File:** `ocr/core/lightning/callbacks/unique_checkpoint.py`

The `format_checkpoint_name` method should respect `auto_insert_metric_name` for ALL checkpoint types:

```python
elif is_best_checkpoint:
    stem = "best"
    # Add metric ONLY if enabled
    if self.auto_insert_metric_name and self.monitor and metrics:  # ← Add check!
        metric_val = metrics.get(self.monitor)
        if isinstance(metric_val, torch.Tensor):
            metric_name_clean = self.monitor.split("/")[-1]
            stem = f"best-{metric_name_clean}-{metric_val.item():.4f}"
```

This makes the naming behavior consistent and predictable.

---

## Summary

| Question | Answer |
|----------|--------|
| **Best checkpoint to use?** | `best-v6.ckpt` (val/acc ≈ 0.8620) |
| **Why no metrics in names?** | `auto_insert_metric_name: false` in config |
| **How to fix?** | Set `auto_insert_metric_name: true` |
| **How many to keep?** | Top 3 best + 1 last (saves 9.2GB) |
| **Next run LR?** | 5e-5 (reduced for fine-tuning) |
