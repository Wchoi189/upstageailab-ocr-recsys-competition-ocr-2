---
title: "Session Handover: CUDA Training Error Debug Session"
date: "2025-11-10"
type: "session_handover"
category: "docs.sessions"
status: "in_progress"
version: "1.0"
session_id: "2025-11-10-cuda-debug"
next_agent: "continuation"
tags: ["cuda", "debugging", "handover"]
---

# Session Handover: CUDA Training Error Debugging

## Executive Summary

**Problem:** Training crashes with CUDA errors when migrating from RTX 3090 (24GB) to RTX 3060 (12GB) in WSL2 environment.

**Progress:** Fixed numerical stability issue (BUG-20251110-002), identified potential memory constraint (BUG-20251110-003), but **new evidence suggests memory is NOT the primary issue**.

**Critical Finding:** ResNet18 test used only ~5GB GPU memory but still crashed → Memory hypothesis needs revision.

**Status:** Multiple hypotheses tested, root cause still not definitively identified. Ready for next investigation phase.

---

## Hypothesis Testing Results

| # | Hypothesis | Test Performed | Result | Conclusion |
|---|------------|---------------|--------|------------|
| 1 | **NaN gradients from step function overflow** | Changed `reciprocal(exp(-50*z))` to `sigmoid(50*z)` | ✅ FIXED | Numerical stability improved, but training still crashes |
| 2 | **Out-of-bounds polygons in training data** | Data validation script | ✅ VALID | 3272 images with proper annotations, data is fine |
| 3 | **cuDNN workspace memory exhaustion (batch=4)** | Reduced to batch=2, accumulate=2 | ❌ FAILED | Still crashes with batch=2 |
| 4 | **Model too large (ResNet50)** | Switched to ResNet18 | ❌ FAILED | Crashes with only 5GB GPU usage |
| 5 | **cuDNN/CUDA driver compatibility** | Not yet tested | ⏳ PENDING | CUDA 12.8, cuDNN 91002, PyTorch 2.8.0+cu128 |
| 6 | **WSL2-specific GPU passthrough issue** | Not yet tested | ⏳ PENDING | Need to test on native Linux |
| 7 | **Import/initialization ordering issue** | Observed 47s wandb import time | ⚠️ SUSPICIOUS | Abnormally long import times |

---

## Key Findings

### 1. Numerical Stability Fix (COMPLETED)
**Issue:** Original step function caused exp overflow
```python
# Before: torch.reciprocal(1 + torch.exp(-50 * (x - y)))
# After:  torch.sigmoid(50 * (x - y))
```
**Impact:** Fixed NaN gradients at step ~122, but revealed underlying crash

### 2. Memory Hypothesis INVALIDATED
**Expected:** RTX 3060 12GB insufficient for batch=4 with ResNet50
**Actual:** ResNet18 with batch=4 uses only ~5GB but still crashes
**Conclusion:** Memory constraint is NOT the primary root cause

### 3. cuDNN Engine Selection Failure
**Error Pattern:**
- Run 1: `CUDA error: illegal memory access` at `torch.isnan(param.grad).any()`
- Run 2: `RuntimeError: FIND was unable to find an engine` during backward pass
- Run 3 (ResNet18): Crash with <5GB GPU usage

**Implication:** cuDNN cannot find/execute convolution algorithm, not memory related

### 4. Abnormal Import Behavior
**Observation:**
- wandb import: 47+ seconds (should be ~2-5s)
- Large gaps (12+ blank lines) in logging
- Import time logging shows extreme delays

**Possible Causes:**
- WSL2 I/O performance issues
- Swap thrashing (system memory pressure)
- File system corruption
- Python environment issues

### 5. Data Validation (COMPLETED)
**Status:** ✅ Training data is valid
- 3272 images with annotations
- Polygons within bounds
- JSON structure correct: `{"images": {"filename": {"words": {...}}}}`

---

## Current Code State

### Files Modified

1. **ocr/models/head/db_head.py** (BUG-20251110-002)
   - Lines 158-204: Replaced step function with sigmoid
   - Lines 196-222: Added validation for thresh/prob maps
   - Status: ✅ Tested, numerically stable

2. **ocr/models/loss/dice_loss.py** (BUG-20251110-002)
   - Lines 43-96: Enhanced input/output validation
   - Status: ✅ Tested, catches NaN/Inf early

3. **ocr/models/loss/bce_loss.py** (BUG-20251109-002)
   - Previous session fix: CPU fallback for tensor operations
   - Status: ⚠️ May need revision

### Documentation Created

1. **Bug Reports:**
   - `docs/bug_reports/bug-20251110-002-nan-gradients-from-step-function-overflow.md`
   - `docs/bug_reports/bug-20251110-002-code-changes.md`
   - `docs/bug_reports/bug-20251110-003-cudnn-workspace-memory-exhaustion.md`

2. **Scripts:**
   - `scripts/test_bug_fix_20251110_002.sh` - Test with original config
   - `scripts/test_minimal_batch.sh` - Test with batch=1
   - `scripts/check_training_data.py` - Validate training data
   - `scripts/diagnose_cuda_issue.py` - Comprehensive diagnostics

---

## Evidence: ResNet18 Test Log Analysis

### Critical Observations:

**1. GPU Memory Usage: ~5GB**
- ResNet18 is 75% smaller than ResNet50
- Batch size 4 with ResNet18 should use ~3-4GB total
- Actual usage: <5GB before crash
- **12GB RTX 3060 has 7GB+ free → Not a memory issue**

**2. Import Performance Degradation:**
```
wandb import:      47,696,912 µs (47.7 seconds!)
timm import:       47,810,200 µs (47.8 seconds!)
```
Normal import times should be 2-5 seconds total. This is 10x slower.

**3. Crash Pattern:**
- Completes sanity check validation
- Starts training epoch
- Crashes early (step 0-26)
- Variable crash point suggests data-dependent trigger

**4. Error Variability:**
Test 1 (batch=4, ResNet50):
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
at: torch.isnan(param.grad).any()
```

Test 2 (batch=4, ResNet50):
```
RuntimeError: FIND was unable to find an engine to execute this computation
at: loss.backward()
```

Test 3 (batch=4, ResNet18):
```
<5GB GPU usage, crash details truncated>
```

---

## Revised Hypothesis: cuDNN/CUDA Environment Issue

### New Primary Hypothesis:
**cuDNN cannot execute convolution operations due to environment/driver issue, NOT memory constraint.**

### Evidence Supporting This:
1. ✅ Crashes with abundant free memory (7GB+ free)
2. ✅ "FIND was unable to find an engine" = algorithm selection failure
3. ✅ Works on RTX 3090 (same CUDA/cuDNN versions)
4. ✅ WSL2 GPU passthrough may have driver issues
5. ✅ Import slowdown suggests system instability

### Possible Root Causes:

#### A. WSL2 GPU Driver Issue
- **Symptom:** cuDNN can't access GPU properly through WSL2
- **Test:** Run same config on native Linux
- **Fix:** Update WSL2 GPU drivers, or use native Linux

#### B. cuDNN Workspace Configuration
- **Symptom:** cuDNN can't allocate workspace (not due to size, but permissions/fragmentation)
- **Test:** Disable cuDNN benchmarking
- **Fix:** Set environment variables:
  ```bash
  export CUDNN_BENCHMARK=0
  export TORCH_CUDNN_V8_API_ENABLED=0
  ```

#### C. Python Environment Corruption
- **Symptom:** Slow imports, unstable CUDA operations
- **Test:** Fresh venv with minimal dependencies
- **Fix:** Rebuild environment

#### D. System Memory Pressure
- **Symptom:** Slow imports, swap thrashing
- **Test:** Check `free -h` and swap usage
- **Fix:** Increase system RAM or disable swap

---

## Next Steps (Priority Order)

### Immediate Actions (Next Agent Should Do First):

1. **Check System Resources**
   ```bash
   # Check system memory and swap
   free -h
   htop  # Look for swap usage

   # Check GPU driver
   nvidia-smi
   cat /proc/driver/nvidia/version
   ```

2. **Test cuDNN Benchmark Disable**
   ```bash
   export CUDNN_BENCHMARK=0
   export TORCH_CUDNN_V8_API_ENABLED=0

   uv run python runners/train.py \
     +hardware=rtx3060_12gb_i5_16core \
     exp_name=test-cudnn-nobench \
     model.encoder.model_name=resnet18 \
     dataloaders.train_dataloader.batch_size=2 \
     trainer.max_steps=50 \
     seed=42
   ```

3. **Test with CUDA_LAUNCH_BLOCKING=1**
   ```bash
   CUDA_LAUNCH_BLOCKING=1 uv run python runners/train.py \
     +hardware=rtx3060_12gb_i5_16core \
     exp_name=test-blocking \
     model.encoder.model_name=resnet18 \
     dataloaders.train_dataloader.batch_size=2 \
     trainer.max_steps=10 \
     seed=42
   ```
   This will show the EXACT operation causing the error.

4. **Verify cuDNN Algorithms Available**
   ```python
   import torch
   print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
   print(f"cuDNN version: {torch.backends.cudnn.version()}")
   print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

   # Test basic convolution
   x = torch.randn(2, 3, 224, 224, device='cuda')
   conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
   y = conv(x)
   loss = y.sum()
   loss.backward()
   print("✓ Basic convolution + backward works")
   ```

### Secondary Investigation:

5. **Profile System During Training**
   ```bash
   # Terminal 1: Start training
   uv run python runners/train.py ...

   # Terminal 2: Monitor GPU
   watch -n 1 nvidia-smi

   # Terminal 3: Monitor system
   htop
   ```

6. **Test on Native Linux (if available)**
   - Boot into native Linux (non-WSL2)
   - Run same configuration
   - If it works → WSL2 driver issue confirmed

7. **Test Different PyTorch Build**
   ```bash
   # Try CPU-only build to verify logic
   pip install torch==2.8.0+cpu
   # If that works, reinstall CUDA build
   pip install torch==2.8.0+cu128
   ```

---

## Quick Reference: Error Patterns

### Pattern 1: Gradient Check Error
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
at: torch.isnan(param.grad).any() in on_before_optimizer_step
```
**Meaning:** CUDA memory corrupted BEFORE gradient check
**Next Step:** Move check earlier or disable gradient validation temporarily

### Pattern 2: cuDNN FIND Error
```
RuntimeError: FIND was unable to find an engine to execute this computation
at: loss.backward()
```
**Meaning:** cuDNN can't select/execute convolution algorithm
**Next Step:** Disable cuDNN benchmark, try different algorithms

### Pattern 3: Early Training Crash (Variable Step)
**Meaning:** Data-dependent, specific batches trigger error
**Next Step:** Use CUDA_LAUNCH_BLOCKING=1 to find exact operation

---

## Configuration That Worked (RTX 3090)

```bash
uv run python runners/train.py \
  exp_name=ocr_training-dbnet-pan_decoder-resnet50 \
  model/architectures=dbnet \
  model.encoder.model_name=resnet50 \
  model.component_overrides.decoder.name=pan_decoder \
  dataloaders.train_dataloader.batch_size=4 \
  trainer.max_epochs=50 \
  trainer.precision=32 \
  seed=42
```

**Environment:**
- GPU: RTX 3090 24GB
- CUDA: 12.8
- cuDNN: 91002
- PyTorch: 2.8.0+cu128
- OS: Linux (non-WSL2)

---

## Environment Comparison

| Aspect | RTX 3090 (Working) | RTX 3060 (Failing) |
|--------|-------------------|-------------------|
| GPU Memory | 24GB | 12GB |
| OS | Linux | WSL2 Linux |
| CUDA | 12.8 | 12.8 |
| cuDNN | 91002 | 91002 |
| PyTorch | 2.8.0+cu128 | 2.8.0+cu128 |
| Batch Size | 4 works | 4 fails, 2 fails, 1 unknown |
| Model | ResNet50 works | ResNet50 fails, ResNet18 fails |

**Key Difference:** WSL2 vs Native Linux

---

## Continuation Prompt for Next Agent

```
Continue debugging CUDA training errors on RTX 3060 12GB WSL2 environment.

CURRENT STATUS:
- Fixed numerical stability (sigmoid step function) ✅
- Validated training data (3272 images, all valid) ✅
- Memory hypothesis INVALIDATED (ResNet18 uses <5GB, still crashes) ❌

NEW HYPOTHESIS: cuDNN/WSL2 driver issue, not memory constraint

IMMEDIATE ACTIONS NEEDED:
1. Run with CUDA_LAUNCH_BLOCKING=1 to identify exact failing operation
2. Test with cuDNN benchmark disabled (export CUDNN_BENCHMARK=0)
3. Check system memory/swap (free -h, htop)
4. Profile GPU during training (nvidia-smi watch)

KEY EVIDENCE:
- ResNet18 test: <5GB GPU used before crash (7GB+ free)
- Import times: 47 seconds (10x normal) suggests system instability
- cuDNN "FIND unable to find engine" = algorithm selection failure
- Works on RTX 3090 native Linux, fails on RTX 3060 WSL2

FILES MODIFIED (keep these):
- ocr/models/head/db_head.py (BUG-20251110-002: sigmoid fix)
- ocr/models/loss/dice_loss.py (BUG-20251110-002: validation)

DOCUMENTATION:
- docs/bug_reports/bug-20251110-002-*.md (numerical stability)
- docs/bug_reports/bug-20251110-003-*.md (memory hypothesis - needs revision)
- docs/sessions/2025-11-10-cuda-debugging-session-handover.md (this file)

START HERE:
Test with CUDA_LAUNCH_BLOCKING=1 and cuDNN benchmark disabled.
```

---

## Contact/Handoff Information

**Session Duration:** ~2 hours
**Tokens Used:** ~95,000 / 200,000
**Files Modified:** 3 core files (db_head.py, dice_loss.py, bce_loss.py)
**Documentation Created:** 8 files (bug reports, scripts, session notes)
**Tests Created:** 4 diagnostic scripts

**Ready for handoff:** ✅
**Next agent can continue immediately with continuation prompt above.**

---

*End of Session Handover Document*
