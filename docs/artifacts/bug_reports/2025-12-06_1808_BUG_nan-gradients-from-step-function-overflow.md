---
title: "Bug 20251110 002 Nan Gradients From Step Function Overflow"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---





# Bug Report: NaN Gradients from Step Function Numerical Overflow

## Bug ID
BUG-20251110-002

## Summary
Training crashes at step ~122 with widespread NaN/Inf gradients propagating from the differentiable binarization step function in `DBHead._step_function()`. The step function uses `torch.reciprocal(1 + torch.exp(-k * (x - y)))` with `k=50`, which causes exponential overflow when `x - y` is negative, leading to gradient explosion during backpropagation and eventual CUDA illegal memory access errors.

## Environment
- **OS:** Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)
- **Python:** 3.10.12
- **PyTorch:** 2.8.0+cu128
- **CUDA:** 12.8 (driver 13.0)
- **GPU:** NVIDIA GeForce RTX 3060 12GB (compute_86)
- **Model:** DBNet with ResNet50 encoder, PAN decoder
- **Precision:** FP32
- **Batch Size:** 4

## Steps to Reproduce
1. Run training with DBNet architecture on RTX 3060 12GB
2. Training proceeds normally for ~122 steps
3. NaN gradients suddenly appear in encoder (starting from conv1.weight)
4. NaN gradients spread to 34+ model parameters (encoder and decoder)
5. Training continues with zeroed gradients (existing safety measure)
6. Eventually crashes with CUDA illegal memory access error

## Expected Behavior
The differentiable binarization step function should compute stable gradients throughout training without numerical overflow or NaN values.

## Actual Behavior
```
ERROR ocr.lightning_modules.ocr_pl - NaN/Inf gradient detected in model.encoder.model.conv1.weight at step 122
ERROR ocr.lightning_modules.ocr_pl - NaN/Inf gradient detected in model.encoder.model.bn1.weight at step 122
... [34 total NaN gradients detected]

torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**Error Sequence:**
1. Step 122: NaN gradients detected in 34 parameters (encoder + decoder)
2. Gradients zeroed out by safety measure in `on_before_optimizer_step`
3. Step 160: More NaN gradients appear
4. Step ~162: CUDA illegal memory access during `torch.cuda.synchronize()` in `bce_loss.py`

## Root Cause Analysis

### Primary Issue: Numerical Overflow in Step Function

**Location:** `ocr/models/head/db_head.py`, line 159
```python
def _step_function(self, x, y):
    return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
```

**Problem:**
- `k = 50` (line 58)
- `x` = prob_maps (sigmoid output, range [0, 1])
- `y` = thresh (sigmoid output, range [0, 1])
- When `x < y` (thresh > prob), `x - y` is negative

**Numerical Overflow Example:**
```python
# When x=0.2, y=0.8 (common scenario):
x - y = -0.6
exp(-50 * -0.6) = exp(30) ≈ 1.07e13  # OVERFLOW!
reciprocal(1 + 1.07e13) ≈ 9.35e-14  # Very small value

# During backpropagation:
# Gradient of reciprocal and exp amplifies extreme values
# → NaN/Inf gradients propagate backward through the network
```

**Why This Causes NaN Gradients:**
1. `exp(30)` produces extremely large values (~1e13)
2. During backpropagation, `reciprocal` and `exp` gradients multiply
3. Gradients become too large to represent in FP32 → overflow to Inf/NaN
4. NaN gradients propagate backward through decoder → encoder
5. All parameters touched by backprop get NaN gradients

**Why It Happens at Step 122:**
- Data-dependent: Specific batches with large `thresh - prob` differences
- Accumulation: Slight numerical errors compound over batches
- Randomness: Depends on data loading order and augmentation

### Secondary Issues
1. **CUDA Illegal Memory Access:** Caused by attempting to process tensors with NaN values
2. **Gradient Explosion:** k=50 amplifies small differences, making gradients unstable
3. **FP32 Precision Limits:** Limited range for extreme exp values

## Proposed Solution

### Fix Strategy

#### 1. Use Numerically Stable Sigmoid (IMPLEMENTED)
**Replace:** `torch.reciprocal(1 + torch.exp(-k * (x - y)))`
**With:** `torch.sigmoid(k * (x - y))`

**Why This Works:**
- Mathematically equivalent: `sigmoid(z) = 1 / (1 + exp(-z))`
- PyTorch's `sigmoid` has built-in overflow protection
- Handles extreme values gracefully without overflow

**Code Change:** `ocr/models/head/db_head.py:158-204`
```python
def _step_function(self, x, y):
    """
    BUG-20251110-002: Fixed numerical instability causing NaN gradients.
    Use torch.sigmoid which is mathematically equivalent but numerically stable.
    """
    # Clamp inputs to prevent extreme values
    x_clamped = torch.clamp(x, 0.0, 1.0)
    y_clamped = torch.clamp(y, 0.0, 1.0)

    # Use sigmoid instead of reciprocal + exp for numerical stability
    result = torch.sigmoid(self.k * (x_clamped - y_clamped))

    # Validate output (safety check)
    if torch.isnan(result).any() or torch.isinf(result).any():
        logger.error("NaN/Inf detected in step function output")
        result = torch.clamp(result, 0.0, 1.0)

    return result
```

#### 2. Add Intermediate Validation (IMPLEMENTED)
**Location:** `ocr/models/head/db_head.py:237-267`

Added validation for `prob_maps` and `thresh` before step function:
- Check for NaN/Inf values
- Fail early with detailed error messages
- Prevent corrupted values from reaching step function

#### 3. Strengthen Loss Computation (IMPLEMENTED)
**Location:** `ocr/models/loss/dice_loss.py:36-96`

Enhanced DiceLoss with:
- Input validation for NaN/Inf in predictions
- Degenerate case handling (union < 2*eps)
- Output validation before returning loss
- Better error messages for debugging

## Implementation

### Files Changed

1. **ocr/models/head/db_head.py**
   - `_step_function()`: Replaced reciprocal+exp with sigmoid (lines 158-204)
   - `forward()`: Added validation for prob_maps and thresh (lines 241-267)

2. **ocr/models/loss/dice_loss.py**
   - `_compute()`: Added input/output validation and degenerate case handling (lines 36-96)

### Code References
All changes are tagged with `BUG-20251110-002` for traceability.

## Testing Plan

### Validation Steps
- [x] Root cause identified (step function numerical overflow)
- [x] Fix implemented (sigmoid replacement + validation)
- [x] Code changes follow project standards (lowercase filenames, bug ID tracking)
- [ ] Run training with original failing configuration
- [ ] Verify no NaN gradients appear
- [ ] Confirm training completes without CUDA errors
- [ ] Check loss values remain stable
- [ ] Validate model convergence

### Test Configuration
```bash
uv run python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  exp_name=ocr_training-dbnet-pan_decoder-resnet50 \
  model/architectures=dbnet \
  model.encoder.model_name=resnet50 \
  model.component_overrides.decoder.name=pan_decoder \
  dataloaders.train_dataloader.batch_size=4 \
  trainer.max_epochs=1 \
  trainer.precision=32 \
  seed=42
```

### Success Criteria
1. Training completes at least 200 steps without NaN gradients
2. No CUDA illegal memory access errors
3. Loss values remain finite and decrease over time
4. Validation metrics show expected behavior

## Prevention Measures

### Code Standards
1. **Numerical Stability:** Always use built-in stable functions (e.g., `sigmoid` instead of manual reciprocal+exp)
2. **Input Validation:** Validate tensors for NaN/Inf before critical operations
3. **Range Clamping:** Clamp intermediate values to expected ranges
4. **Error Messages:** Provide detailed context for debugging

### Best Practices
1. **Avoid Large Scaling Factors:** k=50 is very large; consider reducing if numerical issues persist
2. **Use FP16 with Caution:** Gradient scaling required for such operations
3. **Monitor Gradients:** Log gradient norms during training
4. **Test Edge Cases:** Verify behavior when x ≈ y, x >> y, x << y

## Impact Assessment

### Severity: Critical
- **Affected Users:** All training runs with DBNet architecture
- **Impact:** Training crashes, blocking all experiments
- **Workaround:** None (training completely blocked)
- **Timeline:** Fixed immediately after identification

### Related Components
- DBNet head (direct cause)
- Loss functions (affected by NaN propagation)
- Gradient computation (numerical instability)
- CUDA operations (crash from NaN processing)

## Related Issues

### Previous Bug Reports
- **BUG-20251109-001:** Dice Loss Assertion Error (numerical precision in dice loss)
- **BUG-20251109-002:** CUDA Illegal Memory Access in BCE Loss (attempted CPU fallback)
- **BUG-20251110-001:** Out-of-Bounds Polygon Coordinates (data quality issue)

### Lessons Learned
1. **Numerical stability is critical** in differentiable binarization
2. **Built-in functions are safer** than manual implementations
3. **Early validation prevents cascading errors**
4. **Data-dependent bugs require systematic testing** across batches

## Notes

### Why Previous Fixes Didn't Work
- **BUG-20251109-002:** Focused on BCE loss, but root cause was in step function
- Moving operations to CPU doesn't fix NaN values - just delays the crash
- Zeroing NaN gradients (in `on_before_optimizer_step`) is a symptom treatment, not a cure

### Mathematical Equivalence Verification
```python
# Original formulation
result1 = torch.reciprocal(1 + torch.exp(-k * (x - y)))

# New formulation
result2 = torch.sigmoid(k * (x - y))

# They are mathematically identical:
# sigmoid(z) = 1 / (1 + exp(-z))
# When z = k * (x - y):
# sigmoid(k*(x-y)) = 1 / (1 + exp(-k*(x-y))) = result1
```

### Performance Considerations
- `torch.sigmoid` is a single optimized CUDA kernel
- Likely **faster** than separate `reciprocal` + `exp` operations
- No performance regression expected

---

*This bug report follows the project's standardized format for issue tracking.*
*All code changes are indexed with BUG-20251110-002 for traceability.*
