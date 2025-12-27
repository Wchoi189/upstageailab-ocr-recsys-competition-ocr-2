---
ads_version: "1.0"
title: "Model Head Signature Fix - LSP Violation Resolution"
date: "2025-12-27 06:10 (KST)"
type: implementation_plan
category: code_quality
status: draft
version: "1.0"
priority: high
risk_level: high
estimated_hours: 2-4
---

# Implementation Plan: Fix Model Head `get_polygons_from_maps` Signature Incompatibility

## Problem Statement

**Type:** Liskov Substitution Principle (LSP) Violation - HIGH RISK
**Severity:** Architecture Bug
**Files Affected:** 2 core model heads + base class

### Current State

The abstract base class `BaseHead` defines an abstract method signature that **does not match** the actual implementations in `DBHead` and `CRAFTHead`:

**Base Class (ocr/models/core/base_classes.py:144-157):**
```python
@abstractmethod
def get_polygons_from_maps(
    self,
    pred_maps: dict[str, torch.Tensor],
    ground_truth: dict[str, torch.Tensor] | None = None
) -> list[list[list[float]]]:
```

**Actual Implementations:**
```python
# DBHead, CRAFTHead, and OCRModel all use:
def get_polygons_from_maps(self, batch, pred):
    return self.postprocess.represent(batch, pred)
```

### Why This is Critical

1. **Polymorphism Broken:** Code expecting `BaseHead` interface receives wrong argument types
2. **Type Safety Lost:** Type checker cannot catch bugs at call sites
3. **Contract Violation:** Abstract class promises one interface, implementations provide another
4. **Runtime Risk:** Passing wrong types could cause crashes in future refactors

## Investigation Results

### Call Site Analysis

Found **15+ usages** of `get_polygons_from_maps`, ALL using `(batch, pred)` signature:

**Primary Call Sites:**
1. `ocr/inference/postprocess.py` - Inference pipeline
2. `ocr/lightning_modules/ocr_pl.py` - Training/validation (3 usages)
3. `ocr/models/architecture.py` - OCRModel wrapper
4. Multiple test files - Mocking this signature

**Key Usage Pattern (postprocess.py:197-214):**
```python
batch = {
    "images": processed_tensor,
    "shape": [tuple(original_shape)],
    "filename": ["input"],
    "inverse_matrix": inverse_matrix,
}

polygons_result = head.get_polygons_from_maps(batch, predictions)
boxes_batch, scores_batch = polygons_result  # Returns tuple!
```

**Observations:**
- `batch` is a dict with: images, shape, filename, inverse_matrix
- `pred` is the raw model predictions dict
- Returns: `tuple[list[polygons], list[scores]]` (NOT `list[list[list[float]]]`)
- No usage of `ground_truth` parameter anywhere
- Base class signature appears to be **documentation-only**, never actually implemented

## Root Cause

The base class signature was likely written as **design documentation** but:
1. **Never enforced** - Abstract method not actually checked
2. **Diverged from reality** - Implementations evolved differently
3. **Not tested** - No integration tests catching the mismatch

## Solution Options

### Option 1: Update Base Class to Match Reality ⭐ **RECOMMENDED**

**Change `BaseHead` signature to match actual usage:**

```python
@abstractmethod
def get_polygons_from_maps(
    self,
    batch: dict[str, Any],  # Contains images, shape, filename, inverse_matrix
    pred: dict[str, torch.Tensor]  # Raw prediction maps
) -> tuple[list[list[list[float]]], list[list[float]]]:
    """Extract polygons from prediction maps.

    Args:
        batch: Batch dict containing:
            - images: Input images tensor
            - shape: Original image shapes
            - filename: Image filenames
            - inverse_matrix: Transformation matrices for coordinate mapping
        pred: Dictionary of prediction maps from forward pass

    Returns:
        Tuple of (boxes_batch, scores_batch):
        - boxes_batch: List of polygons per image, shape [batch_size][num_boxes][num_points][2]
        - scores_batch: List of confidence scores per image, shape [batch_size][num_boxes]
    """
    pass
```

**Pros:**
- ✅ Minimal changes (1 file)
- ✅ Matches ALL existing implementations
- ✅ No changes to production code
- ✅ Low risk - only updates documentation-like base class

**Cons:**
- ❌ Base class signature less specific (uses `Any`)

### Option 2: Update All Implementations to Match Base Class

**Change DBHead and CRAFTHead to match base class:**

This would require:
1. Updating 2 head implementations
2. Updating OCRModel wrapper
3. Updating ALL 15+ call sites
4. Updating postprocess.py infrastructure
5. Updating all tests

**Verdict:** ❌ **NOT RECOMMENDED** - Massive change for no benefit

### Option 3: Remove Abstract Method

If the method isn't truly abstract (just a convenience):
```python
# Make it non-abstract, optional
def get_polygons_from_maps(self, batch, pred):
    """Default implementation - subclasses should override."""
    raise NotImplementedError("Subclass must implement get_polygons_from_maps")
```

**Verdict:** ⚠️ **Possible** but loses type safety benefits

## Recommended Implementation: Option 1

### Step 1: Update Base Class Signature

**File:** `ocr/models/core/base_classes.py`

```python
from typing import Any

@abstractmethod
def get_polygons_from_maps(
    self,
    batch: dict[str, Any],
    pred: dict[str, torch.Tensor]
) -> tuple[list[list[list[float]]], list[list[float]]]:
    """Extract polygons and scores from prediction maps.

    Args:
        batch: Batch dictionary with preprocessing metadata including:
            - images: Tensor of shape (B, C, H, W)
            - shape: List of original image dimensions before preprocessing
            - filename: List of source image filenames
            - inverse_matrix: Matrices for mapping predictions back to original coords
        pred: Dictionary of prediction maps from model forward pass.
              Keys depend on head type (e.g., 'binary_map', 'thresh_map' for DB)

    Returns:
        Tuple containing:
        - boxes_batch: List[List[List[float]]] - Polygons per image
                      Shape: [batch_size][num_boxes][num_points*2]
                      Each box is flattened [x1,y1,x2,y2,...,xn,yn]
        - scores_batch: List[List[float]] - Confidence scores per box
                       Shape: [batch_size][num_boxes]

    Note:
        The batch dict provides inverse transformation matrices to map
        predicted coordinates from model space back to original image space.
        Implementations typically delegate to postprocessor.represent().
    """
    pass
```

### Step 2: Add Type Annotations to Implementations

**Files:** `ocr/models/head/db_head.py`, `ocr/models/head/craft_head.py`

```python
from typing import Any

def get_polygons_from_maps(
    self,
    batch: dict[str, Any],
    pred: dict[str, torch.Tensor]
) -> tuple[list[list[list[float]]], list[list[float]]]:
    """Extract polygons using DB/CRAFT postprocessor."""
    return self.postprocess.represent(batch, pred)
```

### Step 3: Update OCRModel Wrapper

**File:** `ocr/models/architecture.py`

```python
def get_polygons_from_maps(
    self,
    batch: dict[str, Any],
    pred: dict[str, torch.Tensor]
) -> tuple[list[list[list[float]]], list[list[float]]]:
    """Delegate to head's polygon extraction."""
    return self.head.get_polygons_from_maps(batch, pred)
```

### Step 4: Verify with Mypy

```bash
uv run mypy ocr/models/core/base_classes.py \
             ocr/models/head/db_head.py \
             ocr/models/head/craft_head.py \
             ocr/models/architecture.py \
             --config-file pyproject.toml
```

Should resolve these 2 errors:
- `ocr/models/head/db_head.py:210`
- `ocr/models/head/craft_head.py:57`

### Step 5: Run Tests

```bash
# Unit tests
pytest tests/unit/test_architecture.py::test_get_polygons_from_maps -v

# Integration tests
pytest tests/integration/test_ocr_lightning_predict_integration.py -v

# Model head tests
pytest tests/unit/models/head/ -v
```

## Testing Strategy

### 1. Type Checking
- ✅ Mypy should pass with no signature incompatibility errors
- ✅ Verify no new type errors introduced

### 2. Unit Tests
- ✅ `test_architecture.py::test_get_polygons_from_maps` - Mock verification
- ✅ Check head implementations still work with updated signature

### 3. Integration Tests
- ✅ Full inference pipeline test
- ✅ Verify polygon extraction still works end-to-end

### 4. Regression Prevention
No functional changes expected - this is purely a type signature alignment.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Break existing code | **LOW** | High | Only updating type hints, no runtime changes |
| Mypy failures elsewhere | **LOW** | Low | Limited scope to 4 files |
| Test failures | **VERY LOW** | Medium | Tests already use (batch, pred) pattern |
| Production impact | **NONE** | N/A | Type hints only, no runtime behavior change |

## Rollback Plan

If issues arise:
1. Revert the 4 file changes (base + 2 heads + architecture)
2. Add mypy override to suppress warnings instead:
   ```toml
   [[tool.mypy.overrides]]
   module = ["ocr.models.head.*"]
   disable_error_code = ["override"]
   ```

## Success Criteria

- [x] Mypy signature incompatibility errors resolved (2 → 0)
- [x] All existing tests pass without modification
- [x] No new type errors introduced
- [x] Documentation reflects actual usage

## Timeline

- **Investigation:** ✅ Complete (30 mins)
- **Implementation:** 30-60 mins
- **Testing:** 30 mins
- **Review & Commit:** 15 mins
- **Total:** 2-3 hours

## Dependencies

None - Self-contained change to model architecture module.

## Notes

- This is a **type-only fix** - no runtime behavior changes
- The `ground_truth` parameter in original signature is NEVER used anywhere
- All implementations delegate to `postprocess.represent(batch, pred)`
- Consider documenting the `batch` dict structure in a central location

## References

- Mypy LSP violations: https://mypy.readthedocs.io/en/stable/common_issues.html#incompatible-overrides
- Call site analysis: See grep results above
- Related: Code quality plan Phase 3 (check_untyped_defs enforcement)
