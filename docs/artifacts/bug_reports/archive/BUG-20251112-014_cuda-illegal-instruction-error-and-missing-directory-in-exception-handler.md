---
title: "Bug 20251112 014 Cuda Illegal Instruction Error And Missing Directory In Exception Handler"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---





# Bug Report: CUDA Illegal Instruction Error and Missing Directory in Exception Handler

## Bug ID
BUG-20251112-014

## Summary
Brief description of the bug.

## Environment
- **OS**: Not specified
- **Python Version**: Not specified
- **Dependencies**: Not specified
- **Browser**: Not specified

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Error Messages
```
Error message here
```

## Screenshots/Logs
If applicable, include screenshots or relevant log entries.

## Impact
- **Severity**: High
- **Affected Users**: Who is affected
- **Workaround**: Any temporary workarounds

## Investigation

### Root Cause Analysis
- **Cause**: What is causing the issue
- **Location**: Where in the code
- **Trigger**: What triggers the issue

### Related Issues
Related issue 1
Related issue 2

## Proposed Solution

### Fix Strategy
How to fix the issue.

### Implementation Plan
1. Step 1
2. Step 2

### Testing Plan
How to test the fix.

## Status
- [ ] Confirmed
- [ ] Investigating
- [ ] Fix in progress
- [ ] Fixed
- [ ] Verified

## Assignee
Who is working on this bug.

## Priority
High/Medium/Low

---

*This bug report follows the project's standardized format for issue tracking.*

## Summary
Training fails with CUDA illegal instruction error during validation sanity check, followed by a secondary FileNotFoundError when attempting to write failure sentinel file to a non-existent checkpoint directory.

## Environment
- **Pipeline Version:** Training pipeline with PyTorch Lightning
- **Components:**
  - `UniqueModelCheckpoint` callback (creates dynamic checkpoint directories)
  - `WandbCompletionCallback.on_exception()` handler
  - FPN decoder with convolution operations
- **Configuration:**
  - Model: resnet18 encoder with UNet-DBHead architecture
  - CUDA enabled, single device
  - Validation sanity check enabled

## Steps to Reproduce
1. Run training with CUDA enabled:
   WARNING  ocr.datasets.base - ⚠️ Maps caching enabled but load_maps=false. Cached
         maps won't be used during __getitem__.
[2025-11-12 12:09:01,006][ocr.datasets.base][WARNING] - ⚠️ Maps caching enabled but load_maps=false. Cached maps won't be used during __getitem__.
INFO     ocr.datasets.base - Cache initialized with version: fa3c8cf4
[2025-11-12 12:09:12,485][ocr.datasets.base][INFO] - Cache initialized with version: fa3c8cf4
INFO     ocr.datasets.base - Cache config: tensor=[3mFalse[0m, images=[3mTrue[0m, maps=[3mTrue[0m,
         load_maps=[3mFalse[0m
[2025-11-12 12:09:12,486][ocr.datasets.base][INFO] - Cache config: tensor=False, images=True, maps=True, load_maps=False

🚀 Performance Preset: none
   No optimizations (baseline)

┏━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃   ┃ Name   ┃ Type         ┃ Params ┃ Mode  ┃
┡━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ 0 │ model  │ OCRModel     │ 16.5 M │ train │
│ 1 │ metric │ CLEvalMetric │      0 │ train │
└───┴────────┴──────────────┴────────┴───────┘
Trainable params: 16.5 M
Non-trainable params: 0
Total params: 16.5 M
Total estimated model params size (MB): 65
Modules in train mode: 155
Modules in eval mode: 0








INFO     ocr.utils.polygon_utils - Filtered [1m26[0m degenerate polygons
         [1m([0mtoo_few_points=[1m0[0m, too_small=[1m26[0m, zero_span=[1m0[0m, empty=[1m0[0m, none=[1m0[0m[1m)[0m
[2025-11-12 12:09:26,256][ocr.utils.polygon_utils][INFO] - Filtered 26 degenerate polygons (too_few_points=0, too_small=26, zero_span=0, empty=0, none=0)
INFO     ocr.utils.polygon_utils - Filtered [1m6[0m degenerate polygons
         [1m([0mtoo_few_points=[1m0[0m, too_small=[1m6[0m, zero_span=[1m0[0m, empty=[1m0[0m, none=[1m0[0m[1m)[0m
[2025-11-12 12:09:27,240][ocr.utils.polygon_utils][INFO] - Filtered 6 degenerate polygons (too_few_points=0, too_small=6, zero_span=0, empty=0, none=0)
Evaluation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 16/16 • 0:00:00









INFO     ocr.utils.polygon_utils - Filtered [1m1[0m degenerate polygons
         [1m([0mtoo_few_points=[1m0[0m, too_small=[1m1[0m, zero_span=[1m0[0m, empty=[1m0[0m, none=[1m0[0m[1m)[0m
[2025-11-12 12:09:45,412][ocr.utils.polygon_utils][INFO] - Filtered 1 degenerate polygons (too_few_points=0, too_small=1, zero_span=0, empty=0, none=0)
INFO     ocr.utils.polygon_utils - Filtered [1m31[0m degenerate polygons
         [1m([0mtoo_few_points=[1m0[0m, too_small=[1m31[0m, zero_span=[1m0[0m, empty=[1m0[0m, none=[1m0[0m[1m)[0m
[2025-11-12 12:12:32,746][ocr.utils.polygon_utils][INFO] - Filtered 31 degenerate polygons (too_few_points=0, too_small=31, zero_span=0, empty=0, none=0)
INFO     ocr.utils.polygon_utils - Filtered [1m3[0m degenerate polygons
         [1m([0mtoo_few_points=[1m0[0m, too_small=[1m3[0m, zero_span=[1m0[0m, empty=[1m0[0m, none=[1m0[0m[1m)[0m
[2025-11-12 12:13:48,749][ocr.utils.polygon_utils][INFO] - Filtered 3 degenerate polygons (too_few_points=0, too_small=3, zero_span=0, empty=0, none=0)
Epoch 0/199 ━                      12/205 0:04:25 • -:--:-- 0.00it/s v_num: r2gv

[1;34mwandb[0m:
[1;34mwandb[0m: 🚀 View run [33mwchoi189_resnet18-unet-dbhead-dbloss-bs16-lr1e-3_SCORE_PLACEHOLDER[0m at: [34mhttps://wandb.ai/ocr-team2/receipt-text-recognition-ocr-project/runs/7vibr2gv[0m
2. Training starts and reaches validation sanity check phase
3. CUDA error occurs during forward pass in FPN decoder
4. Exception handler attempts to write .FAILURE file
5. FileNotFoundError occurs because checkpoint directory doesn't exist

## Expected Behavior
1. Training should complete successfully without CUDA errors
2. If an exception occurs, the failure sentinel file should be written successfully to the checkpoint directory

## Actual Behavior

### Primary Error: CUDA Illegal Instruction
```
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

**Stack Trace:**
- Error occurs in `ocr/models/decoder/fpn_decoder.py:68` during `self.fusion(fused)` call
- Triggered during validation step sanity check
- Happens in convolution operation: `F.conv2d()` in PyTorch

### Secondary Error: FileNotFoundError
```
FileNotFoundError: [Errno 2] No such file or directory:
'/workspaces/.../outputs/ocr_training-unknown_training_20251112_120342/checkpoints/.FAILURE'
```

**Stack Trace:**
- Error occurs in `ocr/lightning_modules/callbacks/wandb_completion.py:59`
- `on_exception()` handler tries to write failure file without ensuring directory exists
- Checkpoint directory is created dynamically by `UniqueModelCheckpoint.setup()` but may not exist if training fails early

## Root Cause Analysis

### Primary Issue: CUDA Illegal Instruction
**Possible Causes:**
1. **CUDA/PyTorch version mismatch**: The PyTorch build may be incompatible with the CUDA driver/runtime
2. **GPU compute capability mismatch**: The compiled CUDA kernels may target a different compute capability than the available GPU
3. **Corrupted CUDA state**: Previous operations may have left CUDA in an invalid state
4. **Memory corruption**: Invalid tensor operations or memory access patterns

**Code Path:**
```
validation_step (ocr_pl.py:141)
├── self.model(**batch) (architecture.py:28)
│   └── self.decoder(encoded_features) (fpn_decoder.py:68)
│       └── self.fusion(fused) (conv.py:548)
│           └── F.conv2d() → CUDA illegal instruction
```

### Secondary Issue: Missing Directory Creation
**Root Cause:** The `on_exception()` method in `WandbCompletionCallback` assumes the checkpoint directory exists, but `UniqueModelCheckpoint` creates it dynamically in `setup()`. If training fails before the first checkpoint save or before `setup()` completes, the directory may not exist.

**Code Path:**
```
on_exception() (wandb_completion.py:45)
├── checkpoint_callback.dirpath retrieved (line 57)
├── Path(checkpoint_callback.dirpath) / ".FAILURE" (line 58)
└── open(failure_file_path, "w") → FileNotFoundError (line 59)
    ❌ Directory not created before file write
```

**Comparison with `on_train_end()`:**
- `on_train_end()` correctly creates directory: `output_dir.mkdir(parents=True, exist_ok=True)` (line 36)
- `on_exception()` lacks this directory creation step

## Resolution

### Fix 1: Ensure Directory Exists in Exception Handler
Update `ocr/lightning_modules/callbacks/wandb_completion.py` to create the directory before writing the failure file:

```python
def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
    import wandb

    current_run = getattr(wandb, "run", None)
    if current_run:
        current_run.tags = current_run.tags + ("status:failed",)
        current_run.summary["final_status"] = "failed"

    checkpoint_callback = next(
        (cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)),
        None,
    )
    if checkpoint_callback and checkpoint_callback.dirpath:
        failure_file_path = Path(checkpoint_callback.dirpath) / ".FAILURE"
        # Ensure directory exists before writing
        failure_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_file_path, "w", encoding="utf-8") as handle:
            handle.write(str(exception))
        print(f"Created failure sentinel file at: {failure_file_path}")
```

### Fix 2: Investigate CUDA Error
1. **Verify CUDA compatibility:**
   - Check PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
   - Check CUDA driver version: `nvidia-smi`
   - Verify compatibility matrix

2. **Enable CUDA DSA for debugging:**
   ```bash
   TORCH_USE_CUDA_DSA=1 python runners/train.py ...
   ```

3. **Test with CPU fallback:**
   - Run with `trainer.accelerator=cpu` to verify the model code is correct
   - If CPU works, the issue is CUDA-specific

4. **Check GPU compute capability:**
   - Verify GPU supports the operations being used
   - May need to recompile PyTorch for specific compute capability

## Testing
- [ ] Reproduction confirmed with provided command
- [ ] Directory creation fix prevents FileNotFoundError
- [ ] CUDA error root cause identified
- [ ] Fix validated on affected environment
- [ ] Regression tests added for exception handler

## Prevention
- Add defensive directory creation in all file write operations
- Add integration tests for exception handling paths
- Document CUDA compatibility requirements
- Add pre-flight checks for CUDA/PyTorch version compatibility
- Consider adding try-except around CUDA operations with fallback to CPU
