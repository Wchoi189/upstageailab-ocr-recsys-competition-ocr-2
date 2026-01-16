# Debug Session: CUDA Segfault in DataLoader

**Date**: 2026-01-14 (Updated: 2026-01-15)
**Status**: ✅ Resolved
**Bug ID**: BUG-20260114-001 (Segfault)
## Quick Summary
Training crashes with "Unexpected segmentation fault in worker" when `num_workers > 0`.

## Environment
| Component    | Value                      |
| ------------ | -------------------------- |
| PyTorch      | 2.6.0+cu124                |
| CUDA Runtime | 12.4                       |
| Driver       | 581.80 (CUDA 13.0 capable) |
| GPU          | RTX 3090 24GB              |
| /dev/shm     | 8GB                        |

## Hypothesis Table

| ID  | Hypothesis                            | Status            | Priority | Evidence                                                                 |
| --- | ------------------------------------- | ----------------- | -------- | ------------------------------------------------------------------------ |
| H1  | CUDA driver/runtime mismatch          | ❌ Ruled out       | -        | Driver 581.80 is forward-compatible with CUDA 12.4 runtime               |
| H2  | Mask type mismatch (attn_mask)        | ✅ ROOT CAUSE      | High     | Cast to float fixed segfault, verified with num_workers=2                |
| H3  | LMDB handle fork unsafety             | ⚠️ Partially ruled | Medium   | `__getstate__` removes env, but re-init may race                         |
| H4  | Shared memory exhaustion              | ❌ Ruled out       | Medium   | Passed 100+ batches after H2 fix                                         |
| H5  | Dataset path update caused corruption | ❌ Ruled out       | Low      | Path was fine, mask mismatch was the issue                               |
| H6  | LMDB environment fork unsafety        | ✅ RESOLVED        | Critical | Fork safety fix applied to LMDBRecognitionDataset. Verified with script. |

## Documents
- [01_initial_analysis.md](./01_initial_analysis.md) - Problem statement
- [02_investigation.md](./02_investigation.md) - Test steps
- [03_findings.md](./03_findings.md) - Results

## Root Cause Analysis (2026-01-15)

### The Symptom
Validation Segfault in `DataLoader` when `num_workers > 0`. Works with `num_workers=0`.

### The Cause: Unsafe LMDB Forking
The `LMDBRecognitionDataset` initializes the LMDB environment (`self.env`) in `__init__` and leaves it open.
```python
# ocr/features/recognition/data/lmdb_dataset.py
def _init_env(self) -> None:
    if self.env is None:
        self.env = lmdb.open(...) # Opens env
```
When `DataLoader` starts with `num_workers > 0` on Linux (default start method: `fork`), the worker processes inherit the parent's memory, including the **open LMDB environment handle**. Accessing an inherited LMDB environment from a child process is not thread/process-safe and leads to Segmentation Faults or corruption (sometimes manifesting as "Transaction object" errors).

### Domain Leakage Note
While `WandBProblemLogger` is detection-specific, it is correctly guarded in `OCRPLModule` by:
```python
if "prob_maps" in pred or "thresh_maps" in pred:
```
The Recognition architecture (`PARSeq`) does not produce these keys, so the logger is skipped. The crash is purely due to the data loading layer.

### Action Plan
1. **Fix `LMDBRecognitionDataset`**: Ensure `self.env` is closed immediately after collecting metadata (`_num_samples`) in the main process.
2. **Lazy Initialization**: Ensure `self.env` is only kept open inside `__getitem__` (worker process), not inherited from parent.

### Verification
- Apply fix to `lmdb_dataset.py`.
- Run validation with `num_workers=4` and `pin_memory=True`.

## Resolution (2026-01-15)
The detailed investigation confirmed that `LMDBRecognitionDataset` was opening the LMDB environment in `__init__` and leaving it open. When `DataLoader` forked worker processes, this open handle was inherited, leading to race conditions and "Transaction object" segmentation faults.

**Fix Applied**:
Modified `ocr/features/recognition/data/lmdb_dataset.py` to:
1. Initialize the environment in `__init__` to read `num_samples`.
2. **Immediately close** the environment in the main process.
3. Allow `__getitem__` (in worker processes) to re-open the environment lazily and safely.

**Verification**:
Ran `scripts/minimal_lmdb_test.py` with:
- `num_workers=2`, `pin_memory=True`: **PASSED** (Previously Failed)
- `num_workers=2`, `pin_memory=False`: **PASSED** (Previously Failed)



**Final Verification (2026-01-16)**:
- Ran short training with `repro_segfault_short_v3_logging` (1 epoch, ~2000 samples).
- **Result**: Success (Exit Code 0).
- **Logs**: `wandb: Synced 5 W&B file(s), 32 media file(s)`.
- **Visuals**: Recognition images with GT/Prediction overlays are now visible in WandB.

The validation pipeline is now robust for multi-worker data loading AND provides visual feedback for recognition tasks.

## New Issues (Handover)
While the segfault is resolved, Visual Inspection of `repro_segfault_short_v3_logging` revealed:
1. **Mojibake / Encoding Failure**: Ground Truth (GT) text appears as `??????????` (e.g., in `image_9531cf.png`), suggesting `cv2.putText` or the font used in `wandb_utils.py` does not support Korean characters.
2. **Empty Predictions**: `Pr:` text is empty, indicating the model is predicting nothing (all EOS?) or the tokenizer is decoding to empty strings.

**Recommendation**: Start a new debugging session focused on "Recognition Model Inference and Logging" to address these specific issues.
