# Checkpoint Management Fix - Implementation Summary

## Changes Made

### 1. Fixed Checkpoint Renaming Script
**File**: `scripts/utils/fix_checkpoint_names.py`

**Changes**:
- Added skip logic for backup files (containing "backup" in filename)
- Added skip logic for checkpoints with 0.0 scores (failed training runs)

**Before**:
```
Found 7 checkpoint(s) to rename:
Processing: best-v1_backup_20260217_022400.ckpt  # ❌ Wrong!
Processing: best_backup_20260217_022401.ckpt      # ❌ Wrong!
```

**After**:
```
⏭️  Skipping (backup): best-v1_backup_20260217_022400.ckpt
⏭️  Skipping (backup): best_backup_20260217_022401.ckpt
⚠️  Invalid score 0.0 (failed training) - skipping: best-v3.ckpt
```

### 2. Enhanced UniqueModelCheckpoint
**File**: `ocr/core/lightning/callbacks/unique_checkpoint.py`

**Changes**:
- Added `min_score_threshold` parameter
- Added `check_monitor_top_k()` override to filter invalid checkpoints
- Added import for `rank_zero_info`

**New Features**:
```python
# Skip checkpoints below threshold
min_score_threshold: 0.01  # Skip if val/acc < 0.01

# Skip NaN/Inf values
if torch.isnan(current) or torch.isinf(current):
    return False
```

### 3. New Checkpoint Configuration
**File**: `configs/train/callbacks/checkpoint_acc_fixed.yaml`

**Key Settings**:
```yaml
enable_version_counter: false    # Prevent v1, v2, v3 suffixes
min_score_threshold: 0.01        # Skip failed training runs
save_last: "link"                # Symlink instead of copy
auto_insert_metric_name: true    # Always include score in filename
add_timestamp: false             # Use index-based structure
```

### 4. New Cleanup Script
**File**: `scripts/utils/cleanup_checkpoints.py`

**Features**:
- Remove backup files
- Remove checkpoints with 0.0 scores
- Keep only top-k best checkpoints
- Dry-run mode by default

**Usage**:
```bash
# See what would be cleaned
uv run python scripts/utils/cleanup_checkpoints.py

# Apply cleanup
uv run python scripts/utils/cleanup_checkpoints.py --apply
```

### 5. Documentation
**File**: `docs/checkpoint_management.md`

Comprehensive guide covering:
- Root cause analysis
- Configuration best practices
- Cleanup procedures
- Migration guide
- Troubleshooting

## Current State Analysis

### Checkpoint Directory Before Fix
```
outputs/checkpoints/
├── best-acc-0.8301_v1.ckpt      # Already renamed
├── best-acc-0.8372.ckpt          # Already renamed
├── best-acc-0.8372_v2.ckpt       # Already renamed
├── best_backup_20260217_022401.ckpt  # ❌ Backup clutter
├── best.ckpt                      # ⚠️ Orphaned from different run
├── best-v1_backup_20260217_022400.ckpt  # ❌ Backup clutter
├── best-v1.ckpt                   # ✓ Valid (score=0.8653)
├── best-v2_backup_20260217_022400.ckpt  # ❌ Backup clutter
├── best-v2.ckpt                   # ✓ Valid (score=0.8656) BEST
├── best-v3.ckpt                   # ❌ Failed training (score=0.0)
├── last.ckpt                      # ✓ Current
├── last-v1.ckpt                   # ⚠️ Old version
├── last-v2.ckpt                   # ⚠️ Old version
├── last-v3.ckpt                   # ⚠️ Old version
└── last-v4.ckpt                   # ⚠️ Old version
```

### Recommended Cleanup Result
```
outputs/checkpoints/
├── best-acc-0.8656.ckpt          # ✓ Best (renamed from best-v2.ckpt)
├── best-acc-0.8653.ckpt          # ✓ 2nd best (renamed from best-v1.ckpt)
├── best-acc-0.8653_v1.ckpt       # ✓ 3rd best (renamed from best.ckpt)
├── last.ckpt                      # ✓ Current
├── best-acc-0.8656.metadata.json
└── best-acc-0.8656.config.json
```

## Action Items

### Immediate (Run These Commands)

1. **Test the fix** (dry run):
   ```bash
   uv run python scripts/utils/fix_checkpoint_names.py
   ```

2. **Apply checkpoint renaming**:
   ```bash
   uv run python scripts/utils/fix_checkpoint_names.py --apply
   ```

3. **Clean up old checkpoints**:
   ```bash
   uv run python scripts/utils/cleanup_checkpoints.py --apply
   ```

### For Future Training

1. **Use new checkpoint config**:
   ```yaml
   # In your training config
   defaults:
     - callbacks: checkpoint_acc_fixed
   ```

2. **Or specify via command line**:
   ```bash
   python train.py callbacks=checkpoint_acc_fixed
   ```

## Expected Behavior After Fix

### Checkpoint Naming
- ✅ Score ALWAYS in filename: `best-acc-0.8656.ckpt`
- ✅ No version suffixes: No `_v1`, `_v2`, `_v3`
- ✅ No backup clutter: Backups only created by rename script
- ✅ Clear identification: Best checkpoint obvious from name

### Checkpoint Saving
- ✅ Skip failed training: Score < 0.01 not saved
- ✅ Skip invalid metrics: NaN/Inf not saved
- ✅ Top-k only: Only best 3 checkpoints kept
- ✅ Symlink for last: `last.ckpt` is symlink (saves space)

### Directory Cleanup
- ✅ No manual cleanup needed
- ✅ Automatic filtering of bad checkpoints
- ✅ Clear metadata for each checkpoint

## Testing

Run these commands to verify the fix:

```bash
# 1. Verify script skips backups and 0.0 scores
uv run python scripts/utils/fix_checkpoint_names.py

# 2. Verify cleanup script identifies files to remove
uv run python scripts/utils/cleanup_checkpoints.py

# 3. (Optional) Apply changes
uv run python scripts/utils/fix_checkpoint_names.py --apply
uv run python scripts/utils/cleanup_checkpoints.py --apply
```

## Files Changed

| File | Status | Purpose |
|------|--------|---------|
| `scripts/utils/fix_checkpoint_names.py` | ✅ Modified | Skip backups and 0.0 scores |
| `ocr/core/lightning/callbacks/unique_checkpoint.py` | ✅ Modified | Add min_score_threshold |
| `configs/train/callbacks/checkpoint_acc_fixed.yaml` | ✅ Created | New recommended config |
| `scripts/utils/cleanup_checkpoints.py` | ✅ Created | Cleanup tool |
| `docs/checkpoint_management.md` | ✅ Created | Documentation |
| `scripts/debug/checkpoint_bug_analysis.md` | ✅ Created | Bug analysis |
| `scripts/debug/checkpoint_debug.py` | ✅ Created | Debug tool |
