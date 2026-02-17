# Checkpoint Scoring Bug Analysis

## Problem

The checkpoint renaming script was incorrectly processing checkpoints that should have been skipped:
1. Backup checkpoints (containing "backup" in filename)
2. Checkpoints from failed training runs (with 0.0 accuracy score)

## Root Cause Analysis

### Issue 1: Backup Checkpoints Being Renamed

**Symptom:** Files like `best-v1_backup_20260217_022400.ckpt` were being processed for renaming.

**Cause:** The script only checked for:
- Files already containing the metric name (e.g., `best-acc-0.8301.ckpt`)
- Files starting with "last"

It did not check for backup files created during previous rename operations.

**Fix:** Added a check to skip any checkpoint containing "backup" in the filename.

### Issue 2: Checkpoints with 0.0 Score

**Symptom:** `best-v3.ckpt` had a `best_model_score` of 0.0 despite being saved as a "best" checkpoint.

**Cause:** Investigation revealed:
- `best-v3.ckpt` was saved at **epoch 0, global step 1**
- The `val/acc` metric was 0.0 at save time
- This indicates the checkpoint was saved either:
  - Before validation properly ran, OR
  - When validation produced 0% accuracy (all predictions wrong)

**How PyTorch Lightning Sets `best_model_score`:**

```python
# In ModelCheckpoint._save_monitor_checkpoint()
current = monitor_candidates.get(self.monitor)  # Gets val/acc from trainer.callback_metrics
if self.check_monitor_top_k(trainer, current):
    self._update_best_and_save(current, trainer, monitor_candidates)

# In ModelCheckpoint._update_best_and_save()
self.current_score = current
self.best_k_models[filepath] = current
self.best_model_score = self.best_k_models[self.best_model_path]
```

The `best_model_score` is set to whatever value is in `trainer.callback_metrics` for the monitored metric. If validation fails or produces 0.0 accuracy, that value is persisted.

**Fix:** Added a check to skip checkpoints with 0.0 metric scores, as these indicate failed training runs.

## Checkpoint Analysis Results

| Checkpoint | Epoch | Global Step | Score | Status |
|------------|-------|-------------|-------|--------|
| best-v3.ckpt | 0 | 1 | 0.0 | ❌ Failed training |
| best.ckpt | 47 | 280,730 | 0.8653 | ⚠️ Orphaned from different run |
| best-v1.ckpt | 46 | 272,360 | 0.8653 | ✓ Valid |
| best-v2.ckpt | 49 | 297,470 | 0.8656 | ✓ Valid (best) |
| best-acc-0.8301_v1.ckpt | - | - | 0.8301 | ✓ Already renamed |
| best-acc-0.8372.ckpt | - | - | 0.8372 | ✓ Already renamed |
| best-acc-0.8372_v2.ckpt | - | - | 0.8372 | ✓ Already renamed |

## Recommendations

### Immediate Actions
1. ✅ **Fixed:** Script now skips backup files
2. ✅ **Fixed:** Script now skips checkpoints with 0.0 scores
3. ⚠️ **Manual:** Delete `best-v3.ckpt` (failed training artifact)
4. ⚠️ **Manual:** Review `best.ckpt` - appears orphaned from a different training run

### Long-term Improvements

1. **Add validation in training code:**
   - Skip saving "best" checkpoints if metric is 0.0 or NaN
   - Add minimum threshold for saving best checkpoints

2. **Improve checkpoint callback:**
   ```python
   # In UniqueModelCheckpoint or training config
   def check_monitor_top_k(self, trainer, current):
       if current is None or (isinstance(current, Tensor) and torch.isnan(current)):
           return False
       if current == 0.0 and self.monitor == "val/acc":
           # Don't save checkpoints with 0% accuracy
           return False
       return super().check_monitor_top_k(trainer, current)
   ```

3. **Add checkpoint validation script:**
   - Run after training completes
   - Flag checkpoints with suspicious scores (0.0, NaN, inf)
   - Generate report of checkpoint health

## Files Modified

- `/workspaces/scripts/utils/fix_checkpoint_names.py`
  - Added backup file skip check
  - Added 0.0 score validation check

## Testing

Run the fixed script:
```bash
uv run python scripts/utils/fix_checkpoint_names.py
```

Expected output:
- Backup files are skipped with message "⏭️ Skipping (backup)"
- 0.0 score checkpoints are skipped with message "⚠️ Invalid score 0.0 (failed training) - skipping"
