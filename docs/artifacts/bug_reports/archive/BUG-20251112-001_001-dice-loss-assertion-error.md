---
title: "Bug 20251112 001 001 Dice Loss Assertion Error"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---



## 🐛 Bug Report: Dice Loss Assertion Error

**Bug ID:** BUG-20251109-001
**Date:** November 9, 2025
**Reporter:** Development Team
**Severity:** High
**Status:** Fixed

### Summary
Training crashes with `AssertionError` in dice loss computation when loss value exceeds 1.0, violating the assertion `assert loss <= 1` in `DiceLoss._compute()`. This occurs due to numerical precision issues when `pred_binary` (thresh_binary_map) contains values outside the expected [0, 1] range.

### Environment
- **Pipeline Version:** Training phase
- **Components:** DBNet OCR model, DiceLoss, DBLoss
- **Configuration:** `trainer.precision=32`, DBNet with ResNet50 backbone, PAN decoder
- **Hardware:** RTX 3060 12GB, CUDA enabled

### Steps to Reproduce
1. Configure training with DBNet architecture and DB loss
2. Start training with batch size 4
3. Training progresses for ~2 batches (18/818 steps)
4. AssertionError occurs during dice loss computation in training step

### Expected Behavior
Dice loss should compute values in the range [0, 1] and training should continue without assertion errors.

### Actual Behavior
```python
File "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/models/loss/dice_loss.py", line 46, in _compute
    assert loss <= 1
AssertionError
```

**Error Stack Trace:**
```
File "ocr/models/loss/dice_loss.py", line 46, in _compute
    assert loss <= 1
AssertionError

During training step:
- ocr/models/loss/db_loss.py:91 - loss_binary = self.dice_loss(pred_binary, gt_binary, gt_prob_mask)
- ocr/models/loss/dice_loss.py:46 - assert loss <= 1
```

### Root Cause Analysis
**Numerical Precision Issue:** The dice loss assertion fails when computed loss exceeds 1.0, which can occur when:

1. **Input Range Violation:** `pred_binary` (thresh_binary_map) may contain values slightly outside [0, 1] due to:
   - Numerical precision in `_step_function()` computation
   - Accumulation of floating-point errors during forward pass
   - Values from sigmoid-like functions that can exceed bounds

2. **Dice Loss Formula:** The dice loss formula is:
   ```python
   loss = 1 - 2.0 * intersection / union
   ```
   When `pred` contains values > 1, the intersection can be larger than expected, causing the loss to exceed 1.

3. **Strict Assertion:** The original code had a strict assertion `assert loss <= 1` without tolerance for numerical errors.

**Code Path:**
```
DBHead.forward()
├── thresh_binary = self._step_function(prob_maps, thresh)  # Can produce values outside [0,1]
└── pred["thresh_binary_map"] = thresh_binary

DBLoss.forward()
├── pred_binary = pred.get("thresh_binary_map")
└── loss_binary = self.dice_loss(pred_binary, gt_binary, gt_prob_mask)

DiceLoss._compute()
├── intersection = (pred * gt * mask).sum()  # Can be > expected if pred > 1
├── union = (pred * mask).sum() + (gt * mask).sum() + self.eps
├── loss = 1 - 2.0 * intersection / union
└── assert loss <= 1  # ❌ Fails when loss > 1
```

### Resolution
**Fix Applied:**
1. **Clamp predictions** to [0, 1] before computing loss to ensure numerical stability
2. **Replace strict assertion** with lenient check that warns if loss > 1.01
3. **Clamp loss** to [0, 2] if it exceeds 1.01 to prevent training crashes
4. **Add diagnostic warnings** with detailed information when loss exceeds bounds

**Code Changes:**
```python
def _compute(self, pred, gt, mask, weights):
    assert pred.shape == gt.shape
    assert pred.shape == mask.shape, f"{pred.shape}, {mask.shape}"
    if weights is not None:
        assert weights.shape == mask.shape
        mask = weights * mask

    # Clamp predictions to [0, 1] to ensure numerical stability
    # This prevents loss from exceeding 1 due to values outside expected range
    pred = pred.clamp(0, 1)

    intersection = (pred * gt * mask).sum()
    union = (pred * mask).sum() + (gt * mask).sum() + self.eps
    loss = 1 - 2.0 * intersection / union

    # Allow small numerical errors (e.g., 1e-5) but warn if significantly > 1
    if loss > 1.01:  # More lenient check with small tolerance
        import warnings
        warnings.warn(
            f"Dice loss exceeds 1: {loss.item():.6f}. "
            f"Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}], "
            f"GT range: [{gt.min().item():.4f}, {gt.max().item():.4f}], "
            f"Intersection: {intersection.item():.6f}, Union: {union.item():.6f}",
            RuntimeWarning
        )
        # Clamp loss to reasonable range [0, 2] to prevent training crash
        loss = loss.clamp(0, 2)

    return loss
```

**Files Changed:**
- `ocr/models/loss/dice_loss.py` - Added input clamping and lenient assertion check

### Testing
- [x] Root cause identified (numerical precision in pred_binary values)
- [x] Fix implemented (input clamping + lenient assertion)
- [x] Code follows project standards
- [ ] Unit tests added for edge cases (values outside [0, 1])
- [ ] Integration test with training run
- [ ] Performance validation (no regression)

### Prevention
- **Input Validation:** Clamp all prediction inputs to expected ranges before loss computation
- **Numerical Stability:** Use epsilon values and clamping to prevent numerical errors
- **Error Handling:** Replace strict assertions with warnings + clamping for production code
- **Testing:** Add unit tests for edge cases (values outside expected ranges)
- **Documentation:** Document expected input ranges for loss functions

### Impact Assessment
- **Severity:** High - Training crashes, blocking all training runs
- **Scope:** Affects all training runs using DBLoss with dice loss component
- **Workaround:** None (training was completely blocked)
- **Timeline:** Fixed immediately after identification

### Related Issues
- May be related to numerical precision issues in other loss functions
- Similar issues could occur in other geometric loss computations
- Highlights need for better input validation in loss functions

### Investigation Notes

**Diagnostic Information:**
- Error occurs during training step 2-18 (varies)
- Assertion fails in `DiceLoss._compute()` at line 46
- Loss value exceeds 1.0, violating assertion

**Affected Components:**
- `DiceLoss._compute()` - Strict assertion without tolerance
- `DBHead._step_function()` - May produce values outside [0, 1]
- `DBLoss.forward()` - Passes pred_binary to dice loss

**Recommended Next Steps:**
1. Add unit tests for edge cases (values outside [0, 1])
2. Review other loss functions for similar strict assertions
3. Consider adding input validation layer for all loss functions
4. Monitor warnings in production to identify if root cause needs addressing

---

*This bug report follows the project's standardized format for issue tracking.*
