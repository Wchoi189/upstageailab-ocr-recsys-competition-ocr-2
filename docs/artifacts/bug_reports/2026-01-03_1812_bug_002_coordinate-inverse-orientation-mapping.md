---
ads_version: "1.0"
type: "bug_report"
category: "troubleshooting"
status: "completed"
severity: "medium"
version: "1.0"
tags: ['bug', 'testing', 'orientation', 'coordinate-transformation']
title: "Coordinate Inverse Orientation Mapping for Test Verification"
date: "2026-01-03 18:12 (KST)"
branch: "main"
description: "Fixed failing unit tests by implementing correct inverse orientation mapping for coordinate transformations. Discovered that orientations 5 (TRANSPOSE) and 7 (TRANSVERSE) are self-inverse for coordinate transformations across dimension spaces."
---

# Bug Report - Coordinate Inverse Orientation Mapping
Bug ID: BUG-2026-002

## Summary

Fixed failing unit tests in `test_inference_engine_remaps_polygons_back_to_raw_orientation` by implementing correct inverse orientation mapping for coordinate transformations. The issue revealed that **coordinate transformations across different dimension spaces require a different inverse mapping than image transformations**.

**Key Discovery**: Orientations 5 (TRANSPOSE) and 7 (TRANSVERSE) are **self-inverse** for coordinate transformations, unlike their behavior in image transformations.

## Environment

- **Component**: `InferenceEngine._remap_predictions_if_needed()`
- **Test File**: `tests/ocr/utils/test_orientation.py`
- **Related Files**: `ocr/utils/orientation.py`, `ocr/utils/orientation_constants.py`

## Reproduction

1. Run test: `pytest tests/ocr/utils/test_orientation.py::test_inference_engine_remaps_polygons_back_to_raw_orientation`
2. Observe failures for orientations 5, 6, and 7
3. Error: `TypeError: InferenceEngine._remap_predictions_if_needed() got an unexpected keyword argument 'orientation'`

## Comparison

**Expected**:
- Method accepts `orientation`, `canonical_width`, `canonical_height` parameters
- Correctly remaps polygon coordinates from canonical frame back to raw orientation
- All orientation tests (5, 6, 7) pass

**Actual**:
- Method signature was incomplete (missing parameters)
- Using `ORIENTATION_INVERSE_INT` mapping caused orientations 5 and 7 to fail
- Tests produced incorrect coordinates: `'189,79,169,79,169,59,189,59' != '10,20,30,20,30,40,10,40'`

## Root Cause

**Image Transformations** vs **Coordinate Transformations** require different inverse mappings:

### Image Transformations (ORIENTATION_INVERSE_INT)
- Orientation 5 (TRANSPOSE) inverse → Orientation 7 (TRANSVERSE)
- Orientation 7 (TRANSVERSE) inverse → Orientation 5 (TRANSPOSE)

### Coordinate Transformations (what we need)
- Orientation 5 (TRANSPOSE) inverse → Orientation 5 (TRANSPOSE) - **self-inverse**
- Orientation 7 (TRANSVERSE) inverse → Orientation 7 (TRANSVERSE) - **self-inverse**

**Why the difference?**

For TRANSPOSE (orientation 5) with formula `x_new = y, y_new = x`:
- When transforming coordinates in canonical space (100x200) back to raw space (200x100)
- Applying the SAME transformation works: `(20, 10) → (10, 20)` ✓
- But applying TRANSVERSE gives: `(20, 10) → (189, 79)` ✗

This is because:
- Image transformations operate in the SAME coordinate space
- Coordinate transformations operate ACROSS different dimension spaces (raw ↔ canonical)
- For orientations with flips (5, 7), the dimension swap makes them self-inverse

## Resolution

Implemented custom coordinate inverse mapping in `ocr/inference/engine.py`:

```python
# BUG-2026-002: Coordinate transformations require different inverse mapping
_COORDINATE_INVERSE = {
    1: 1,  # NORMAL
    2: 2,  # FLIP_HORIZONTAL (self-inverse)
    3: 3,  # ROTATE_180 (self-inverse)
    4: 4,  # FLIP_VERTICAL (self-inverse)
    5: 5,  # TRANSPOSE (self-inverse for coordinates)
    6: 8,  # ROTATE_90_CW -> ROTATE_90_CCW
    7: 7,  # TRANSVERSE (self-inverse for coordinates)
    8: 6,  # ROTATE_90_CCW -> ROTATE_90_CW
}
```

**Verification Results**:
```
Orientation 5: Inverse is orientation 5 ✓
Orientation 6: Inverse is orientation 8 ✓
Orientation 7: Inverse is orientation 7 ✓
Orientation 8: Inverse is orientation 6 ✓
```

## Testing

All 3 orientation tests now pass:
```
tests/ocr/utils/test_orientation.py::test_inference_engine_remaps_polygons_back_to_raw_orientation[5] PASSED
tests/ocr/utils/test_orientation.py::test_inference_engine_remaps_polygons_back_to_raw_orientation[6] PASSED
tests/ocr/utils/test_orientation.py::test_inference_engine_remaps_polygons_back_to_raw_orientation[7] PASSED
```

## Impact

- **Testing**: Fixes 3 failing unit tests
- **Functionality**: Enables proper verification of orientation transformation logic
- **Documentation**: Documents the difference between image and coordinate inverse mappings
- **Compatibility**: No impact on production code (method only used in tests)

## Related Issues

- **BUG-2025-011**: Previously removed `_remap_predictions_if_needed()` from UI inference path due to coordinate misalignment
- **Important**: This method should NOT be used in production inference paths. Predictions should remain in canonical coordinate space for display.

## Files Changed

- `ocr/inference/engine.py` - Implemented `_remap_predictions_if_needed()` with correct coordinate inverse mapping (BUG-2026-002)

## Key Learnings

1. **Coordinate transformations ≠ Image transformations**: Inverse mappings differ
2. **TRANSPOSE/TRANSVERSE are self-inverse** for coordinate transformations across dimension spaces
3. **Always verify** transformation reversibility with actual test data
4. **Dimension swapping** affects how inverse transformations work
