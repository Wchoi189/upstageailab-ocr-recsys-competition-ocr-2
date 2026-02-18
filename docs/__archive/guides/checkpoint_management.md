# Checkpoint Management Guide

## Problem: Chaotic Checkpoint Output

### Symptoms
- Multiple `last.ckpt` files (`last.ckpt`, `last-v1.ckpt`, `last-v2.ckpt`, etc.)
- Versioned best checkpoints (`best-v1.ckpt`, `best-v2.ckpt`, `best-v3.ckpt`)
- Backup files cluttering the directory
- Checkpoints with misleading names (missing score in filename)
- Checkpoints from failed training runs (score=0.0)

### Root Causes

1. **Version Counter Enabled**: PyTorch Lightning's `enable_version_counter=True` (default) adds `_v1`, `_v2`, `_v3` suffixes when files already exist.

2. **Metric Name Not Inserted**: When `auto_insert_metric_name=False` or when the metric value is 0.0, filenames don't include the score.

3. **No Score Threshold**: Checkpoints are saved even with 0.0 accuracy (failed training runs).

4. **Multiple Training Resumptions**: Each time training is resumed, new versioned checkpoints are created.

## Solution

### 1. Updated Checkpoint Configuration

Use the new configuration with proper settings:

```yaml
# configs/train/callbacks/checkpoint_acc_fixed.yaml
_target_: ocr.core.lightning.callbacks.unique_checkpoint.UniqueModelCheckpoint
dirpath: ${global.paths.checkpoint_dir}

# Checkpoint naming
filename: "best"
auto_insert_metric_name: true  # ALWAYS include metric in filename

# Monitoring
monitor: "val/acc"
mode: "max"
save_top_k: 3

# CRITICAL settings
enable_version_counter: false    # Prevent v1, v2, v3 suffixes
min_score_threshold: 0.01        # Skip failed training runs
save_last: "link"                # Use symlink instead of copy
add_timestamp: false             # Use index-based structure
```

### 2. Enhanced UniqueModelCheckpoint

The callback now includes:

- **`min_score_threshold`**: Skip checkpoints below threshold
- **NaN/Inf filtering**: Skip invalid metric values
- **Better logging**: Inform when checkpoints are skipped

### 3. Cleanup Existing Checkpoints

Run the cleanup script:

```bash
# Dry run (see what would be deleted)
uv run python scripts/utils/cleanup_checkpoints.py

# Apply cleanup
uv run python scripts/utils/cleanup_checkpoints.py --apply
```

## Best Practices

### Directory Structure

With index-based organization (managed by Hydra):

```
outputs/
└── <index>/
    └── checkpoints/
        ├── best-acc-0.8656.ckpt       # Best checkpoint
        ├── best-acc-0.8653.ckpt       # 2nd best
        ├── best-acc-0.8624.ckpt       # 3rd best
        ├── last.ckpt -> best-acc-0.8656.ckpt  # Symlink to best
        ├── best-acc-0.8656.metadata.json
        └── best-acc-0.8656.config.json
```

### Training Configuration

```bash
# Start new training with proper checkpoint config
python train.py \
  callbacks=checkpoint_acc_fixed \
  global.paths.checkpoint_dir=outputs/<index>/checkpoints
```

### Resuming Training

When resuming, the checkpoint callback will:
1. Load the previous state
2. Continue saving with metric in filename
3. Not create versioned suffixes (version counter disabled)
4. Skip saving if score is below threshold

```bash
# Resume from best checkpoint
python train.py \
  callbacks=checkpoint_acc_fixed \
  trainer.checkpoint_path=outputs/<index>/checkpoints/best-acc-0.8656.ckpt
```

## Cleanup Script Options

```bash
# Basic cleanup (dry run)
uv run python scripts/utils/cleanup_checkpoints.py

# Remove backups and zero-score checkpoints
uv run python scripts/utils/cleanup_checkpoints.py \
  --remove-backups \
  --remove-zero-scores

# Keep only top 2 checkpoints
uv run python scripts/utils/cleanup_checkpoints.py \
  --keep-top-k 2 \
  --apply

# Also remove versioned checkpoints (use with caution!)
uv run python scripts/utils/cleanup_checkpoints.py \
  --remove-versioned \
  --apply
```

## Migration Guide

### From Old to New Configuration

1. **Backup existing checkpoints**:
   ```bash
   cp -r outputs/checkpoints outputs/checkpoints_backup_$(date +%Y%m%d)
   ```

2. **Run cleanup**:
   ```bash
   uv run python scripts/utils/cleanup_checkpoints.py --apply
   ```

3. **Update training config**:
   ```yaml
   # In your training config
   defaults:
     - callbacks: checkpoint_acc_fixed  # Changed from checkpoint_acc
   ```

4. **Verify new checkpoints**:
   After next training run, verify checkpoints have:
   - Metric in filename: `best-acc-0.8656.ckpt`
   - No version suffix: No `_v1`, `_v2`
   - Metadata files: `.metadata.json`, `.config.json`

## Troubleshooting

### Checkpoints Still Getting Versioned

**Problem**: Files like `best-acc-0.8656_v1.ckpt` are created.

**Solution**: Ensure `enable_version_counter: false` in checkpoint config.

### Metric Not in Filename

**Problem**: Checkpoints named `best.ckpt` instead of `best-acc-0.8656.ckpt`.

**Causes**:
1. `auto_insert_metric_name: false` - Set to `true`
2. Metric value is 0.0 or NaN - Check training logs
3. Metric not logged properly - Verify `self.log("val/acc", ...)` in validation

### Too Many Checkpoints Saved

**Problem**: More than `save_top_k` checkpoints exist.

**Solution**: This is normal for:
- `last.ckpt` (always saved separately)
- Epoch checkpoints (if configured)
- Checkpoints from previous training runs

Run cleanup script to remove old checkpoints.

### Checkpoint from Failed Training

**Problem**: Checkpoint with 0.0 accuracy saved.

**Solution**: Set `min_score_threshold: 0.01` to skip these.

## Files Modified

| File | Purpose |
|------|---------|
| `ocr/core/lightning/callbacks/unique_checkpoint.py` | Added `min_score_threshold`, `check_monitor_top_k` override |
| `configs/train/callbacks/checkpoint_acc_fixed.yaml` | New config with proper settings |
| `scripts/utils/cleanup_checkpoints.py` | Cleanup tool for existing checkpoints |
| `scripts/utils/fix_checkpoint_names.py` | Fixed to skip backups and 0.0 scores |
