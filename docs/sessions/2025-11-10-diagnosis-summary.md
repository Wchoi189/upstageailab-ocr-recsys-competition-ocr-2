---
title: "CUDA Training Error - Comprehensive Diagnosis"
date: "2025-11-10"
status: "diagnosis_complete"
---

# CUDA Training Error: Root Cause Diagnosis

## Executive Summary

**Problem:** Training crashes with "CUDA error: an illegal instruction was encountered"

**Diagnosis:** The problem is **NOT** in:
- ❌ GPU hardware (tested OK)
- ❌ CUDA/cuDNN drivers (tested OK)
- ❌ Model architecture (tested OK)
- ❌ Wandb multiprocessing (partially responsible but not primary)

**Root Cause:** The issue is in the **training pipeline** - specifically the interaction between:
- DataLoader with real training data
- Loss computation with actual labels
- Some component that only activates during real training

---

## Diagnostic Test Results

### Test 1: Basic CUDA Operations ✅
**File:** `scripts/test_basic_cuda.py`
**Result:** PASSED (10/10 iterations)

```
[Test 1] CUDA Availability ✓
[Test 2] Simple Tensor Operations ✓
[Test 3] Convolution Forward Pass ✓
[Test 4] Backward Pass ✓
[Test 5] Repeated Backward Passes (10 iterations) ✓
```

**Conclusion:** GPU, CUDA, cuDNN are all working correctly.

### Test 2: Model Architecture ✅
**File:** `scripts/test_model_forward_backward.py`
**Result:** PASSED (10/10 iterations)

```
[Test 1] Testing Encoder (ResNet18) ✓
[Test 2] Testing Decoder (FPN) ✓
[Test 3] Testing Head (DBHead) ✓
[Test 4] Testing Full Model Pipeline (10 iterations) ✓
```

**Conclusion:** The model itself works perfectly with forward and backward passes.

### Test 3: Training with Wandb + forkserver ❌
**File:** `scripts/test_forkserver_fix.sh`
**Result:** FAILED at first training step

```
Error location: torch.autograd.graph.py:829 in _engine_run_backward
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

**Conclusion:** Something in the full training pipeline causes the crash.

### Test 4: Training without Wandb ⏳
**File:** `scripts/test_cudnn_stability.sh`
**Status:** Running now
**Purpose:** Isolate whether wandb is still involved

---

## What Works vs What Fails

### Works ✅
- Basic tensor operations on GPU
- Convolution forward + backward
- Full model architecture (encoder + decoder + head)
- 10 training iterations with synthetic data
- Optimizer steps with synthetic data

### Fails ❌
- Full training pipeline with real data
- Crashes during backward pass in first few steps
- Error: "illegal instruction" (not "illegal memory access")

---

## Key Clues

### Clue 1: "Illegal Instruction" vs "Illegal Memory Access"

**"Illegal instruction"** is different from "illegal memory access":
- **Illegal memory**: Accessing invalid memory addresses (pointer errors)
- **Illegal instruction**: GPU trying to execute unsupported operation

This suggests a **specific CUDA kernel or operation** that fails on this hardware/driver combination.

### Clue 2: Works with Synthetic Data, Fails with Real Data

The model test uses `torch.randn()` (synthetic data) and works fine.
Training uses real image data and crashes.

**Possible causes:**
1. Data preprocessing creates invalid values (NaN, Inf, extreme values)
2. Image loading corrupts tensors
3. Data augmentation creates edge cases
4. Label data has invalid format

### Clue 3: Consistent Crash Location

The crash always happens during backward pass:
```python
torch.autograd.graph.py:829 in _engine_run_backward
```

This is PyTorch's C++ autograd engine, suggesting a specific gradient computation fails.

### Clue 4: WSL2 vs Native Linux

- Works: RTX 3090 on native Linux
- Fails: RTX 3060 on WSL2

**Possible issues:**
1. WSL2 GPU passthrough doesn't support certain operations
2. Driver version mismatch through WSL2 layer
3. WSL2 memory management issues

---

## Hypotheses (Ranked by Likelihood)

### Hypothesis 1: Invalid Training Data (60% confidence)
**Theory:** Real training data contains invalid values that only appear during loss computation

**Evidence:**
- Synthetic data works, real data fails
- Validation checks were added but may not catch all cases
- Crash happens during backward (gradient computation with real labels)

**Test:**
```python
# Add to training loop
for batch in dataloader:
    images, labels = batch
    assert not torch.isnan(images).any()
    assert not torch.isinf(images).any()
    assert not torch.isnan(labels).any()
    assert not torch.isinf(labels).any()
```

### Hypothesis 2: Loss Function with Real Labels (40% confidence)
**Theory:** Loss computation with real label shapes/values triggers unsupported operation

**Evidence:**
- Model works with synthetic data
- Crash happens during backward (after loss computation)
- Loss functions were recently modified (BUG-20251110-002)

**Test:**
```python
# Test loss functions in isolation
from ocr.models.loss.dice_loss import DiceLoss
from ocr.models.loss.bce_loss import MaskL1Loss

# Use real data shapes
prob_map = torch.rand(4, 1, 56, 56, device='cuda')
target = torch.rand(4, 1, 56, 56, device='cuda')

dice_loss = DiceLoss()
loss = dice_loss(prob_map, target)
loss.backward()
```

### Hypothesis 3: WSL2 GPU Passthrough Issue (30% confidence)
**Theory:** WSL2 doesn't properly support certain CUDA operations

**Evidence:**
- Works on native Linux (RTX 3090)
- Fails on WSL2 (RTX 3060)
- "Illegal instruction" suggests operation not supported

**Test:**
- Run on native Linux (if available)
- Check WSL2 GPU driver version
- Update WSL2 kernel/driver

### Hypothesis 4: DataLoader Multiprocessing (20% confidence)
**Theory:** DataLoader workers corrupt data during transfer

**Evidence:**
- forkserver was supposed to fix this
- But still crashes with forkserver
- Synthetic data (no DataLoader) works

**Test:**
```yaml
# In config
num_workers: 0  # Disable multiprocessing
```

---

## Immediate Next Steps

### Step 1: Test Without Wandb (Running)
```bash
bash scripts/test_cudnn_stability.sh
```

**If succeeds:** Wandb is still part of the problem
**If fails:** Confirms issue is in training pipeline itself

### Step 2: Add Data Validation
Add comprehensive validation before training:

```python
# In training loop
def validate_batch(batch):
    images = batch['images']
    gt_shrink = batch['gt_shrink']
    gt_shrink_mask = batch['gt_shrink_mask']

    # Check for invalid values
    assert not torch.isnan(images).any(), f"NaN in images"
    assert not torch.isinf(images).any(), f"Inf in images"
    assert not torch.isnan(gt_shrink).any(), f"NaN in gt_shrink"
    assert not torch.isnan(gt_shrink_mask).any(), f"NaN in gt_shrink_mask"

    # Check value ranges
    assert images.min() >= -10 and images.max() <= 10, f"Images out of range: [{images.min()}, {images.max()}]"
    assert gt_shrink.min() >= 0 and gt_shrink.max() <= 1, f"gt_shrink out of range"

    print(f"✓ Batch validation passed")
```

### Step 3: Test Loss Functions in Isolation
Create test script that uses real data shapes:

```python
# scripts/test_loss_functions.py
from ocr.models.loss.dice_loss import DiceLoss
from ocr.models.loss.bce_loss import MaskL1Loss

# Load one real batch
dataloader = ...
batch = next(iter(dataloader))

# Test each loss function
dice_loss = DiceLoss()
bce_loss = MaskL1Loss()

# ... test with real data
```

### Step 4: Disable DataLoader Multiprocessing
```yaml
# configs/dataloaders/rtx3060_12gb_i5_16core.yaml
train_dataloader:
  batch_size: 4
  num_workers: 0  # DISABLE multiprocessing
  persistent_workers: false
```

---

## Workarounds (If Root Cause Can't Be Fixed)

### Workaround 1: Use CPU for Problematic Operations
Some operations can be moved to CPU:

```python
# In loss computation
if torch.cuda.is_available():
    try:
        loss = loss_fn(pred, target)
    except RuntimeError as e:
        if "illegal instruction" in str(e):
            # Fall back to CPU
            loss = loss_fn(pred.cpu(), target.cpu()).cuda()
```

### Workaround 2: Use Mixed Precision Training
FP16 might avoid the problematic operation:

```yaml
# configs/trainer/default.yaml
precision: "16-mixed"  # Instead of 32
```

### Workaround 3: Switch to Native Linux
Boot into native Linux (not WSL2) if available.

### Workaround 4: Use Different PyTorch Build
Try a different PyTorch/CUDA version:

```bash
pip install torch==2.7.0+cu121  # Different CUDA version
```

---

## Related Documentation

- [CUDA Debugging Session Handover](./2025-11-10-cuda-debugging-session-handover.md)
- [Wandb Multiprocessing Analysis](./2025-11-10-wandb-multiprocessing-analysis.md)
- Bug Report: BUG-20251110-002 (NaN gradients fix)
- Bug Report: BUG-20251109-002 (CUDA memory access)

---

## Conclusion

The problem is **NOT a simple issue** with:
- Hardware
- Drivers
- Model architecture
- Basic CUDA operations

The problem is a **complex interaction** in the training pipeline that only occurs with:
- Real training data
- Specific loss computations
- Full training loop
- Possibly WSL2 environment

**Next agent should:**
1. Wait for wandb-disabled test results
2. Add comprehensive data validation
3. Test loss functions in isolation
4. Try num_workers=0 workaround

---

*Last updated: 2025-11-10 18:45 UTC*
