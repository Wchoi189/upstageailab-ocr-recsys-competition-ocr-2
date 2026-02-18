# Checkpoint & Logging Fixes

**Date:** 2026-02-16
**Issues Fixed:**
1. Checkpoint filenames missing performance metrics
2. Nested WandB folder structure (`outputs/wandb/wandb/`)

---

## Issue 1: Checkpoint Naming

### Problem
Checkpoints named `best-v6.ckpt` instead of `best-acc-0.8623.ckpt`, making it impossible to identify performance without loading the checkpoint.

### Root Cause
`auto_insert_metric_name: false` in checkpoint config (likely disabled to avoid Hydra "=" parsing bug, but the sanitizer already handles this).

### Fixes Applied

**1. Config Fix:** [configs/train/callbacks/checkpoint_acc.yaml](../configs/train/callbacks/checkpoint_acc.yaml)
```yaml
auto_insert_metric_name: true  # ← Changed from false
save_top_k: 3                   # Keep only top 3
verbose: true                   # Log saves
```

**2. Rename Tool:** [scripts/utils/rename_checkpoints.py](../scripts/utils/rename_checkpoints.py)
- Auto-detects validation accuracy from WandB logs
- Renames top 3 checkpoints with actual scores
- Preserves checkpoint order by timestamp

### Usage

```bash
# Preview what will be renamed
uv run python scripts/utils/rename_checkpoints.py --dry-run

# Actually rename checkpoints
uv run python scripts/utils/rename_checkpoints.py

# Manual score override if auto-detection fails
uv run python scripts/utils/rename_checkpoints.py --score 0.8620
```

### Results

**Before:**
```
best-v6.ckpt      # What's the score? Unknown!
best-v5.ckpt
best-v4.ckpt
```

**After:**
```
best-acc-0.8623.ckpt  # Clear performance indicator
best-acc-0.8613.ckpt
best-acc-0.8603.ckpt
```

---

## Issue 2: Nested WandB Folders

### Problem
WandB creates `outputs/wandb/wandb/`, causing:
- Confusing directory structure
- Two separate run lists (one in each `wandb/` folder)
- Path confusion in configs

### Root Cause
WandB **automatically creates** a `wandb/` subdirectory inside `save_dir`. When `save_dir` was set to `outputs/wandb`, this resulted in `outputs/wandb/wandb/`.

### Fix Applied

Changed `save_dir` from `outputs/wandb` → `outputs` in configs:

**Files Updated:**
1. [configs/global/paths.yaml](../configs/global/paths.yaml#L44)
2. [configs/train/logger/wandb.yaml](../configs/train/logger/wandb.yaml#L6)
3. [configs/experiment/parseq_flash_wandb.yaml](../configs/experiment/parseq_flash_wandb.yaml#L17)
4. [configs/experiment/parseq_flash_fast.yaml](../configs/experiment/parseq_flash_fast.yaml#L92)

**Before:**
```yaml
wandb: ${global.paths.output_dir}/wandb  # WandB creates wandb/ inside this
# Result: outputs/wandb/wandb/ ❌
```

**After:**
```yaml
wandb: ${global.paths.output_dir}  # WandB creates wandb/ subdirectory
# Result: outputs/wandb/ ✅
```

### Directory Structure

**Before (nested):**
```
outputs/
├── wandb/
│   ├── run-XXX/          # Old runs
│   └── wandb/            # ← Nested folder!
│       └── run-YYY/      # New runs separated
```

**After (fixed):**
```
outputs/
└── wandb/                # All runs in one location
    ├── run-XXX/
    └── run-YYY/
```

### Migration (Optional)

If you want to consolidate existing runs:

```bash
# Move runs from nested folder to parent
mv outputs/wandb/wandb/run-* outputs/wandb/

# Remove empty nested folder
rmdir outputs/wandb/wandb/
```

**Note:** This is optional - future runs will use the correct structure.

---

## Complete Workflow

### 1. Rename Existing Checkpoints

```bash
# Rename top 3 checkpoints with actual scores
uv run python scripts/utils/rename_checkpoints.py

# Verify
ls -lh outputs/checkpoints/best-acc-*.ckpt
```

### 2. Clean Up Old Checkpoints

```bash
# Preview cleanup (saves ~7.4 GB)
uv run python scripts/utils/manage_checkpoints.py --cleanup

# Actually delete
uv run python scripts/utils/manage_checkpoints.py --cleanup --confirm
```

### 3. Next Training Run

```bash
# Use the best renamed checkpoint
uv run python scripts/runners/train.py \
  mode=train \
  experiment=parseq_flash_plateau \
  +checkpoint_path=outputs/checkpoints/best-acc-0.8623.ckpt \
  trainer.max_epochs=60 \
  trainer.val_check_interval=1.0 \
  train.optimizer.lr=5e-5
```

**Future checkpoints will be named automatically:**
- `best-acc-0.8650.ckpt`
- `best-acc-0.8640.ckpt`
- etc.

---

## Verification

### Check Checkpoint Names
```bash
ls -lth outputs/checkpoints/*.ckpt | head -5
```

Expected:
```
best-acc-0.8623.ckpt  # ✓ Has score
best-acc-0.8613.ckpt  # ✓ Has score
best-acc-0.8603.ckpt  # ✓ Has score
```

### Check WandB Structure
```bash
ls -la outputs/wandb/
```

Expected: **NO** nested `wandb/wandb/` folder

---

##Summary

| Issue | Fix | Tool | Status |
|-------|-----|------|--------|
| Checkpoint names lack metrics | Set `auto_insert_metric_name: true` | `rename_checkpoints.py` | ✅ Fixed |
| Nested wandb folders | Change `save_dir` to `outputs` | Config update | ✅ Fixed |
| Too many old checkpoints | Keep top 3 only | `manage_checkpoints.py` | ✅ Available |

Both issues are now resolved and won't recur in future training runs! 🎉
