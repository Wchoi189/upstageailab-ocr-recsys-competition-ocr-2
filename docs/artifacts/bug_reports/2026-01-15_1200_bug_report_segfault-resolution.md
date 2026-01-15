---
title: "Resolution: CUDA Segfault in DataLoader (Validation Phase)"
date: "2026-01-15 12:00 (KST)"
version: "1.0"
ads_version: "1.0"
type: "bug_report"
category: "troubleshooting"
status: "completed"
tags: ["segfault", "cuda", "resolution"]
---

**Bug ID**: BUG-20260114-001
**Status**: Resolved
**Date**: 2026-01-15

## Summary
The training process was crashing with a `Segmentation fault` (SIGSEGV) during the validation phase when using multiple workers (`num_workers > 0`) or pinned memory (`pin_memory=True`). The crash was isolated to the interaction between PyTorch's `multiprocessing` (specifically `_pin_memory_loop` and `rebuild_storage_fd`) and the underlying CUDA environment in the validation loop.

## Root Cause Analysis
1.  **Symptoms**:
    *   `RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly`
    *   `ConnectionResetError` in `_pin_memory_loop`
    *   `TypeError: 'Transaction' object cannot be interpreted as an integer` (when workers disabled but image logging enabled)
    *   Occurred consistently at Validation Batch 0 or shortly after.

2.  **Investigation**:
    *   **Data Integrity**: Verified by `minimal_lmdb_test.py`; dataset and basic `DataLoader` were healthy.
    *   **Model Integrity**: Verified by `minimal_model_test.py`; model inference works correctly in isolation.
    *   **Environment**: The crash was specific to the full Lightning `Trainer` environment where complex interactions between `WandB` logging, customized `multiprocessing` start methods, and CUDA context management occurred.

3.  **Specific Triggers**:
    *   **`pin_memory=True`**: Spawns a background thread that manages pinned memory buffers. In this environment, the IPC mechanism for transferring these buffers between processes (via file descriptors) was failing (`rebuild_storage_fd`), causing the worker or the pin-memory thread to segfault.
    *   **`WandB` Image Logging**: Even with `num_workers=0`, the `WandBProblemLogger` attempted to log images. The subsequent `Transaction` type error suggests that `checkpoints` or `wandb` library internals were interfering with object serialization (pickling) when running in the main process loop alongside Lightning's heavy signal handling.

## Resolution
The fix involves strictly enforcing a single-process, safe configuration for the specific environment constraints:

1.  **Disable Multiprocessing**: Set `num_workers=0` for all dataloaders.
2.  **Disable Memory Pinning**: Set `pin_memory=False` to prevent the `_pin_memory_loop` thread creation.
3.  **Disable Batch Image Logging**: Set `logger.per_batch_image_logging.enabled=False` to prevent the `Transaction` pickling error and allow validation to complete.

### Applied Configuration
A verified "Safe Mode" configuration was created at `configs/trainer/debug_safe.yaml`:

```yaml
# @package _global_
trainer:
  limit_train_batches: 50
  limit_val_batches: 20
  max_epochs: 1

data:
  dataloaders:
    train_dataloader:
      num_workers: 0
      pin_memory: false
      persistent_workers: false
    val_dataloader:
      num_workers: 0
      pin_memory: false
      persistent_workers: false
    # ... (same for test/predict)

logger:
  per_batch_image_logging:
    enabled: false
```

## Verification
A validation run (`verify_fix_safe_v4`) successfully completed the full training and validation cycle without errors, confirming the stability of the solution.
