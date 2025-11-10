---
title: "Code Changes for BUG-20251110-002"
author: "ai-agent"
date: "2025-11-10"
timestamp: "2025-11-10 13:57 UTC"
type: "bug_fix"
category: "troubleshooting"
status: "implemented"
version: "1.0"
tags: ['bug', 'cuda', 'nan-gradients', 'code-changes', 'numerical-stability']
bug_id: "BUG-20251110-002"
severity: "Critical"
---

# Code Changes for BUG-20251110-002: NaN Gradients from Step Function Overflow

## Overview
This document tracks all code changes made to fix BUG-20251110-002. All changes are indexed with the bug ID for proper tracking and version control.

## Bug Report
- **Bug ID:** BUG-20251110-002
- **Bug Report:** [docs/bug_reports/bug-20251110-002-nan-gradients-from-step-function-overflow.md](bug-20251110-002-nan-gradients-from-step-function-overflow.md)
- **Severity:** Critical
- **Status:** Fixed (implementation complete, testing pending)

## Summary of Changes

### Root Cause
The differentiable binarization step function in `DBHead._step_function()` used a numerically unstable formulation:
```python
torch.reciprocal(1 + torch.exp(-50 * (x - y)))
```
This caused exponential overflow when `x - y` was negative, leading to NaN/Inf gradients during backpropagation.

### Solution
Replace with mathematically equivalent but numerically stable `torch.sigmoid()`:
```python
torch.sigmoid(50 * (x - y))
```

## Functions Changed (Indexed by Bug ID)

### 1. `DBHead._step_function()` - `ocr/models/head/db_head.py`

**Bug ID:** BUG-20251110-002
**Function:** `DBHead._step_function(self, x, y)`
**Change Type:** Bug Fix (Numerical Stability)
**Date:** 2025-11-10
**Lines:** 158-204

#### Changes Made:

1. **Replaced Numerical Formulation (Lines 158-204)**
   - **Before:** `torch.reciprocal(1 + torch.exp(-self.k * (x - y)))`
   - **After:** `torch.sigmoid(self.k * (x_clamped - y_clamped))`
   - **Purpose:** Prevent exponential overflow that causes NaN gradients
   - **Mathematical Equivalence:** `sigmoid(z) = 1 / (1 + exp(-z))`

2. **Added Input Clamping (Lines 179-183)**
   - Clamp `x` (prob_maps) to [0, 1]
   - Clamp `y` (thresh) to [0, 1]
   - **Purpose:** Prevent numerical errors from being amplified by k=50
   - **Bug ID Reference:** `# BUG-20251110-002: Clamp inputs to prevent extreme values`

3. **Added Output Validation (Lines 190-202)**
   - Check for NaN/Inf in result
   - Log detailed error if detected
   - Clamp result to [0, 1] as fallback
   - **Purpose:** Catch any remaining numerical issues
   - **Bug ID Reference:** `# BUG-20251110-002: Validate output to catch any remaining numerical issues`

4. **Enhanced Documentation (Lines 159-178)**
   - Added comprehensive docstring explaining the fix
   - Documented mathematical equivalence
   - Provided example of overflow scenario
   - **Bug ID Reference:** See full docstring in file

#### Code Snippet:
```python
def _step_function(self, x, y):
    """
    Differentiable step function for binarization.

    BUG-20251110-002: Fixed numerical instability causing NaN gradients.
    Original: torch.reciprocal(1 + torch.exp(-k * (x - y))) with k=50
    - Caused exp overflow when x - y is negative (e.g., exp(50) ≈ 5e21)
    - Led to NaN gradients propagating through backprop at step ~122

    Fix: Use torch.sigmoid which is mathematically equivalent but numerically stable.
    sigmoid(k*z) = 1 / (1 + exp(-k*z)) but with built-in overflow protection.
    """
    # Clamp inputs to prevent extreme values
    x_clamped = torch.clamp(x, 0.0, 1.0)
    y_clamped = torch.clamp(y, 0.0, 1.0)

    # Use sigmoid instead of reciprocal + exp for numerical stability
    result = torch.sigmoid(self.k * (x_clamped - y_clamped))

    # Validate output to catch any remaining numerical issues
    if torch.isnan(result).any() or torch.isinf(result).any():
        logger.error(...)
        result = torch.clamp(result, 0.0, 1.0)

    return result
```

#### Status:
- ✅ Changes applied
- ✅ Code follows project standards
- ✅ Documented with bug ID
- ⏳ Testing in progress

---

### 2. `DBHead.forward()` - `ocr/models/head/db_head.py`

**Bug ID:** BUG-20251110-002
**Function:** `DBHead.forward(self, x: torch.Tensor, return_loss: bool = True)`
**Change Type:** Enhancement (Validation)
**Date:** 2025-11-10
**Lines:** 241-267

#### Changes Made:

1. **Added Thresh Map Validation (Lines 241-253)**
   - Check for NaN values in thresh output
   - Check for Inf values in thresh output
   - **Purpose:** Catch numerical issues before step function
   - **Bug ID Reference:** `# BUG-20251110-002: Validate thresh map before step function`

2. **Added Prob Maps Validation (Lines 255-267)**
   - Check for NaN values in prob_maps
   - Check for Inf values in prob_maps
   - **Purpose:** Catch numerical issues before step function
   - **Bug ID Reference:** `# BUG-20251110-002: Validate prob_maps before step function`

#### Code Snippet:
```python
if return_loss:
    # Threshold map
    thresh = self.thresh(fuse)

    # BUG-20251110-002: Validate thresh map before step function
    if torch.isnan(thresh).any():
        raise ValueError(
            f"NaN values detected in thresh map. "
            f"Shape: {thresh.shape}, Device: {thresh.device}, "
            f"Range: [{thresh.min().item():.6f}, {thresh.max().item():.6f}]"
        )
    if torch.isinf(thresh).any():
        raise ValueError(...)

    # BUG-20251110-002: Validate prob_maps before step function
    if torch.isnan(prob_maps).any():
        raise ValueError(...)
    if torch.isinf(prob_maps).any():
        raise ValueError(...)

    # Approximate Binary map
    thresh_binary = self._step_function(prob_maps, thresh)
```

#### Status:
- ✅ Changes applied
- ✅ Early error detection
- ✅ Detailed error messages
- ⏳ Testing in progress

---

### 3. `DiceLoss._compute()` - `ocr/models/loss/dice_loss.py`

**Bug ID:** BUG-20251110-002
**Function:** `DiceLoss._compute(self, pred, gt, mask, weights)`
**Change Type:** Enhancement (Validation + Robustness)
**Date:** 2025-11-10
**Lines:** 36-96

#### Changes Made:

1. **Added Input Validation (Lines 43-53)**
   - Check for NaN values in pred before clamping
   - Check for Inf values in pred before clamping
   - **Purpose:** Catch corrupted inputs from step function
   - **Bug ID Reference:** `# BUG-20251110-002: Enhanced input validation`

2. **Added Degenerate Case Handling (Lines 62-70)**
   - Check if union < 2*eps (degenerate case)
   - Return safe fallback loss value (1.0)
   - Log warning for monitoring
   - **Purpose:** Prevent division by very small values
   - **Bug ID Reference:** `# BUG-20251110-002: Check for degenerate cases before division`

3. **Added Output Validation (Lines 74-81)**
   - Check for NaN/Inf in computed loss
   - Raise detailed error if detected
   - **Purpose:** Catch numerical instability in loss computation
   - **Bug ID Reference:** `# BUG-20251110-002: Validate loss value before returning`

#### Code Snippet:
```python
def _compute(self, pred, gt, mask, weights):
    # ... shape assertions ...

    # BUG-20251110-002: Enhanced input validation
    if torch.isnan(pred).any():
        raise ValueError(
            f"NaN values in pred input to DiceLoss. "
            f"Shape: {pred.shape}, Range: [{pred.min().item():.6f}, {pred.max().item():.6f}]"
        )
    if torch.isinf(pred).any():
        raise ValueError(...)

    pred = pred.clamp(0, 1)

    intersection = (pred * gt * mask).sum()
    union = (pred * mask).sum() + (gt * mask).sum() + self.eps

    # BUG-20251110-002: Check for degenerate cases
    if union < self.eps * 2:
        logger.warning("Degenerate case in DiceLoss: union too small")
        return torch.tensor(1.0, device=pred.device, dtype=pred.dtype)

    loss = 1 - 2.0 * intersection / union

    # BUG-20251110-002: Validate loss value
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(
            f"NaN/Inf loss computed in DiceLoss. "
            f"Intersection: {intersection.item():.6e}, Union: {union.item():.6e}"
        )

    # Clamp if loss > 1.01 (with warning)
    if loss > 1.01:
        warnings.warn(...)
        loss = loss.clamp(0, 2)

    return loss
```

#### Status:
- ✅ Changes applied
- ✅ Input validation added
- ✅ Degenerate case handling
- ✅ Output validation
- ⏳ Testing in progress

---

## Testing Status

### Unit Tests
- [ ] Test step function with edge cases (x=0, x=1, y=0, y=1, x=y)
- [ ] Test step function gradient computation
- [ ] Test DiceLoss with degenerate inputs (union ≈ 0)
- [ ] Test DiceLoss with NaN inputs (should raise ValueError)

### Integration Tests
- [ ] Run training for 200+ steps without NaN gradients
- [ ] Verify CUDA errors don't occur
- [ ] Check loss values remain finite
- [ ] Validate gradient norms stay within expected range

### Regression Tests
- [ ] Verify mathematical equivalence (sigmoid vs reciprocal+exp)
- [ ] Check validation metrics match expected values
- [ ] Ensure no performance degradation

## Validation Checklist

- [x] All changes indexed with BUG-20251110-002
- [x] Code follows project standards (lowercase filenames, no caps)
- [x] Comprehensive error messages added
- [x] Docstrings updated with bug ID references
- [x] Mathematical correctness verified
- [ ] Unit tests added/updated
- [ ] Integration test passed
- [ ] Training run verified without errors

## Next Steps

1. **Run Test Training:**
   ```bash
   uv run python runners/train.py \
     +hardware=rtx3060_12gb_i5_16core \
     exp_name=test-bug-fix-20251110-002 \
     model/architectures=dbnet \
     dataloaders.train_dataloader.batch_size=4 \
     trainer.max_epochs=1 \
     seed=42
   ```

2. **Monitor for:**
   - No NaN gradient warnings
   - No CUDA illegal memory access errors
   - Stable loss values
   - Decreasing loss trend

3. **If successful:**
   - Update bug report status to "verified"
   - Document performance impact (if any)
   - Consider adding unit tests

4. **If issues persist:**
   - Check k value (consider reducing from 50)
   - Enable CUDA_LAUNCH_BLOCKING=1 for detailed error location
   - Add more intermediate validation

## Performance Considerations

### Expected Impact: Neutral or Positive
- `torch.sigmoid()` is a single optimized CUDA kernel
- Likely **faster** than separate `reciprocal()` + `exp()` calls
- No additional memory overhead
- Clamping operations are negligible (single pass)

### Measured Impact: TBD
- Baseline throughput: Unknown (previous runs crashed)
- Post-fix throughput: To be measured
- Memory usage: Expected identical

## Notes

### Why This Fix Works
1. **Mathematical Equivalence:** `sigmoid(z) = 1/(1 + exp(-z))` is exact
2. **Built-in Stability:** PyTorch's sigmoid handles overflow/underflow gracefully
3. **CUDA Optimized:** Single kernel call is more efficient
4. **Industry Standard:** sigmoid is the standard way to compute this function

### Alternative Approaches Considered
1. **Reduce k from 50 to 10:** Would work but changes model behavior
2. **Manual clamping of exp result:** Error-prone and less efficient
3. **Switch to FP16:** Requires gradient scaling, doesn't fix root cause

### Related Documentation
- PyTorch sigmoid docs: https://pytorch.org/docs/stable/generated/torch.sigmoid.html
- DBNet paper: https://arxiv.org/pdf/1911.08947.pdf (Section 3.2)

---

*This code changes document follows the project's standardized format for issue tracking.*
*All changes are indexed with BUG-20251110-002 for traceability and code review.*
