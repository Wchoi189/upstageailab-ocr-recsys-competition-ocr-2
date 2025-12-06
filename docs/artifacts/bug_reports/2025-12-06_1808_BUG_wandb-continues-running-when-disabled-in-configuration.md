---
title: "Bug 20251112 015 Wandb Continues Running When Disabled In Configuration"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---





# Bug Report: WandB Continues Running When Disabled in Configuration

## Bug ID
BUG-20251112-015

## Summary
WandB continues to initialize and run even when explicitly disabled in configuration files. This causes WandB-related messages to appear in logs and may contribute to CUDA errors during training.

## Environment
- **OS**: Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)
- **Python Version**: 3.10.12
- **PyTorch**: 2.8.0+cu128
- **CUDA**: 12.8
- **Configuration**:
  - `configs/train.yaml`: `wandb: false`
  - `configs/logger/wandb.yaml`: `wandb.enabled: False`

## Steps to Reproduce
1. Set `wandb: false` in `configs/train.yaml`
2. Set `wandb.enabled: False` in `configs/logger/wandb.yaml`
3. Run training: `python runners/train.py ...`
4. Observe WandB messages in output:
   ```
   wandb: 🚀 View run wchoi189_resnet18-unet-dbhead-dbloss-bs16-lr1e-3_SCORE_PLACEHOLDER at: https://wandb.ai/...
   ```

## Expected Behavior
When WandB is disabled in configuration:
- No WandB logger should be initialized
- No WandB callbacks should be active
- No WandB-related messages should appear in logs
- `wandb.finish()` should not be called if WandB was never initialized

## Actual Behavior
WandB continues to run despite being disabled:
- WandB logger is initialized (line 164 in `runners/train.py` incorrectly checks `config.logger.wandb` as truthy)
- WandB callbacks (`WandbCompletionCallback`, `WandbImageLoggingCallback`) are always instantiated regardless of WandB status
- `wandb.finish()` is called unconditionally (line 108-109), potentially initializing WandB
- WandB messages appear in logs even when disabled

## Error Messages
```
wandb: 🚀 View run wchoi189_resnet18-unet-dbhead-dbloss-bs16-lr1e-3_SCORE_PLACEHOLDER at: https://wandb.ai/ocr-team2/receipt-text-recognition-ocr-project/runs/wo9oj6gg
```

## Root Cause Analysis

### Issue 1: Incorrect WandB Enable Check (Primary)
**Location**: `runners/train.py:164`
**Problem**: Code checks `if config.logger.wandb:` which evaluates to `True` because `config.logger.wandb` is a dict (from `configs/logger/wandb.yaml`), not a boolean. A non-empty dict is always truthy in Python, even if it contains `enabled: False`.

**Current Code**:
```python
if config.logger.wandb:  # ❌ Always True if wandb.yaml is included
    # Initialize WandB logger
```

**Expected Check**:
```python
if config.logger.wandb.get("enabled", False):  # ✅ Check the enabled flag
    # Initialize WandB logger
```

### Issue 2: Unconditional wandb.finish() Call
**Location**: `runners/train.py:108-109`
**Problem**: `wandb.finish()` is called unconditionally at the start of training, which may initialize WandB even when it should be disabled.

**Current Code**:
```python
# Clean up any lingering W&B session to prevent warnings (lazy import)
import wandb
wandb.finish()  # ❌ Always called, may initialize WandB
```

**Expected Behavior**: Only call `wandb.finish()` if WandB was actually initialized.

### Issue 3: WandB Callbacks Always Instantiated
**Location**: `configs/callbacks/default.yaml:12-16`
**Problem**: `WandbCompletionCallback` and `WandbImageLoggingCallback` are always instantiated regardless of WandB status. These callbacks import and use WandB, potentially causing initialization.

**Current Config**:
```yaml
wandb_completion:
  _target_: ocr.lightning_modules.callbacks.wandb_completion.WandbCompletionCallback

wandb_image_logging:
  _target_: ocr.lightning_modules.callbacks.wandb_image_logging.WandbImageLoggingCallback
```

**Expected Behavior**: These callbacks should only be instantiated when WandB is enabled.

## Proposed Solution

### Fix 1: Correct WandB Enable Check
Update `runners/train.py:164` to check the `enabled` flag:
```python
if config.logger.wandb.get("enabled", False):
    # Initialize WandB logger
```

### Fix 2: Conditional wandb.finish() Call
Only call `wandb.finish()` if WandB is enabled:
```python
# Clean up any lingering W&B session to prevent warnings (lazy import)
if config.logger.wandb.get("enabled", False):
    import wandb
    wandb.finish()
```

### Fix 3: Conditional Callback Instantiation
Update `configs/callbacks/default.yaml` to conditionally include WandB callbacks, or add logic in `runners/train.py` to filter out WandB callbacks when WandB is disabled.

**Option A**: Conditional config inclusion (requires Hydra config composition)
**Option B**: Filter callbacks after instantiation (simpler, recommended):
```python
# After instantiating callbacks
if not config.logger.wandb.get("enabled", False):
    from ocr.lightning_modules.callbacks.wandb_completion import WandbCompletionCallback
    from ocr.lightning_modules.callbacks.wandb_image_logging import WandbImageLoggingCallback
    callbacks = [cb for cb in callbacks
                 if not isinstance(cb, (WandbCompletionCallback, WandbImageLoggingCallback))]
```

## Testing Plan
1. Set `wandb: false` in `configs/train.yaml` and `wandb.enabled: False` in `configs/logger/wandb.yaml`
2. Run training and verify:
   - No WandB logger is created
   - No WandB messages appear in logs
   - No WandB callbacks are active
   - Training completes without WandB-related errors
3. Set `wandb: true` and `wandb.enabled: True` and verify WandB works correctly
4. Test edge cases (missing config keys, partial configs)

## Impact
- **Severity**: High
- **Affected Users**: All users attempting to disable WandB
- **Workaround**: None - WandB cannot be fully disabled currently
- **Related Issues**: May contribute to CUDA errors if WandB initialization causes GPU state issues

## Related Issues
- BUG-20251112-014: CUDA illegal instruction error (WandB may be contributing factor)
- PLAN-003: WandB import optimization (lazy imports implemented, but enable check is broken)

## Status
- [ ] Confirmed
- [x] Investigating
- [ ] Fix in progress
- [ ] Fixed
- [ ] Verified

---

*This bug report follows the project's standardized format for issue tracking.*
