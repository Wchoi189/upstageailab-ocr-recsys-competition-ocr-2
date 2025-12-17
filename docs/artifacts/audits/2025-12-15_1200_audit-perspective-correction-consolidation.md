---
ads_version: "1.0"
title: "Audit Perspective Correction Consolidation"
date: "2025-12-16 00:11 (KST)"
type: "audit"
category: "compliance"
status: "active"
version: "1.0"
tags: ['audit', 'compliance']
---



# Perspective Correction Consolidation Audit

## Executive Summary

**Finding:** Duplicate perspective correction implementations identified across codebase.

**Scope:** 86 files reference perspective correction functionality.

**Recommendation:** Consolidate into single canonical implementation in Phase 2+.

---

## Duplicate Implementations

### 1. ocr/utils/perspective_correction.py (460 lines)
**Purpose:** Rembg-based mask perspective correction for inference

**Key Functions:**
- `fit_mask_rectangle()` - fits rectangle to binary foreground mask
- `four_point_transform()` - applies perspective warp
- `calculate_target_dimensions()` - computes output dimensions from corners
- `correct_perspective_from_mask()` - high-level mask-based correction
- `remove_background_and_mask()` - rembg background removal

**Features:**
- Mask-based corner detection with fallbacks
- Convex hull approximation with iterative epsilon tuning
- INTER_LANCZOS4 interpolation for quality
- Optional rembg integration
- Comprehensive error handling and diagnostics

**Used By:**
- `ocr/inference/preprocess.py::apply_optional_perspective_correction()`
- Inference pipeline (optional preprocessing step)

---

### 2. ocr/datasets/preprocessing/perspective.py (108 lines)
**Purpose:** Dataset preprocessing with docTR or OpenCV perspective correction

**Key Functions:**
- `PerspectiveCorrector.correct()` - main entry point
- `_opencv_perspective_correction()` - OpenCV-based warp
- `_doctr_perspective_correction()` - docTR extract_rcrops approach
- `_compute_perspective_targets()` - computes output dimensions
- `_normalize_corners()` - normalizes corners for docTR

**Features:**
- Dual-mode: docTR geometry OR OpenCV fallback
- Pre-provided corners (not mask-based detection)
- INTER_LINEAR interpolation
- Configurable via AdvancedPreprocessor config

**Used By:**
- `ocr/datasets/preprocessing/` pipeline components
- Training dataset preprocessing

---

## Functional Differences

| Feature | ocr/utils | ocr/datasets |
|---------|-----------|--------------|
| **Input** | Image + mask | Image + corners |
| **Corner Detection** | Mask-based fitting | Pre-provided |
| **Methods** | Rembg + OpenCV | docTR + OpenCV |
| **Interpolation** | LANCZOS4 | LINEAR |
| **Primary Use** | Inference | Training |
| **Error Handling** | Extensive diagnostics | Basic fallback |

---

## Code Duplication Analysis

### Duplicate Logic: Dimension Calculation

**ocr/utils/perspective_correction.py:47-68**
```python
def calculate_target_dimensions(pts: np.ndarray) -> Tuple[int, int]:
    (tl, tr, br, bl) = pts
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = int(max(width_a, width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = int(max(height_a, height_b))
    return max_width, max_height
```

**ocr/datasets/preprocessing/perspective.py:81-92**
```python
def _compute_perspective_targets(corners: np.ndarray):
    tl, tr, br, bl = corners.astype(np.float32)
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(round(width_a)), int(round(width_b)), 1)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(round(height_a)), int(round(height_b)), 1)
    return src_points, (max_width, max_height), dst_points
```

**Difference:** Identical algorithm, different API (np.sqrt vs np.linalg.norm).

---

### Duplicate Logic: OpenCV Perspective Warp

**ocr/utils/perspective_correction.py:71-102**
```python
def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.dtype != np.float32:
        pts = pts.astype(np.float32)
    max_width, max_height = calculate_target_dimensions(pts)
    dst = np.array([[0, 0], [max_width - 1, 0],
                    [max_width - 1, max_height - 1], [0, max_height - 1]],
                   dtype="float32")
    m = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height),
                                 flags=cv2.INTER_LANCZOS4)
    return warped
```

**ocr/datasets/preprocessing/perspective.py:66-70**
```python
def _opencv_perspective_correction(self, image: np.ndarray, corners: np.ndarray):
    src_points, (max_width, max_height), dst_points = self._compute_perspective_targets(corners)
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected = cv2.warpPerspective(image, perspective_matrix, (max_width, max_height),
                                   flags=cv2.INTER_LINEAR)
    return corrected, perspective_matrix
```

**Difference:** Identical OpenCV calls, different interpolation (LANCZOS4 vs LINEAR).

---

## Over-Engineered Exception Handling

### Location: ocr/inference/preprocess.py:148-153

```python
except Exception as exc:  # noqa: BLE001
    LOGGER.warning("Perspective correction failed or unavailable: %s", exc)
    if return_matrix:
        import numpy as np
        return image_bgr, np.eye(3, dtype=np.float32)
    return image_bgr
```

**Issue:** Bare `Exception` catch suppresses all errors including programming errors.

**Recommendation:** Catch specific exceptions (RuntimeError, ImportError, cv2.error).

---

## Impact Analysis

### Files Affected by Consolidation

**Direct Usage (2 files):**
- `ocr/inference/preprocess.py`
- `ocr/datasets/preprocessing/enhanced_pipeline.py`

**Indirect References (84 files):**
- Documentation (30+ files)
- Tests and demos (20+ files)
- Legacy/archived code (30+ files)
- Configuration files (4 files)

**High-Risk Files:**
- `ocr/inference/engine.py` - core inference logic
- `ocr/datasets/preprocessing/pipeline.py` - training pipeline
- Backend APIs using perspective correction flag

---

## Consolidation Strategy

### Phase 1: Create Unified Module (New File)

**Location:** `ocr/preprocessing/perspective_correction.py`

**Unified API:**
```python
class PerspectiveCorrector:
    """Unified perspective correction for inference and training."""

    @staticmethod
    def correct_from_corners(
        image: np.ndarray,
        corners: np.ndarray,
        interpolation: int = cv2.INTER_LANCZOS4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply perspective correction given pre-detected corners."""
        ...

    @staticmethod
    def correct_from_mask(
        image: np.ndarray,
        mask: np.ndarray,
        return_diagnostics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict | None]:
        """Apply perspective correction via mask-based corner detection."""
        ...

    @staticmethod
    def remove_background(
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove background using rembg (optional dependency)."""
        ...
```

---

### Phase 2: Migration Path

**Step 1:** Create consolidated module with tests
**Step 2:** Update inference module imports
**Step 3:** Update datasets module imports
**Step 4:** Deprecate old implementations
**Step 5:** Remove after 1 release cycle

**Testing:**
- Unit tests for each method
- Integration tests for inference pipeline
- Regression tests for training pipeline

---

## Recommendations

### Immediate Actions (Phase 1.5)

1. ✅ **Audit Complete** - Document duplicate implementations
2. ⏳ **Defer Consolidation** - Too broad for Phase 1 (affects 86 files)
3. ⏳ **Plan Separate Phase** - Create dedicated consolidation plan for Phase 2+

### Medium-Term Actions (Phase 2-3)

1. Create `ocr/preprocessing/perspective_correction.py` with unified API
2. Migrate inference module to use consolidated implementation
3. Migrate datasets module to use consolidated implementation
4. Update 86 affected files (prioritize active code over archived docs)

### Long-Term Actions (Post-Refactor)

1. Remove `ocr/utils/perspective_correction.py` (redundant)
2. Remove `ocr/datasets/preprocessing/perspective.py` (redundant)
3. Update all documentation references
4. Archive legacy implementations

---

## Risk Assessment

**Risk Level:** HIGH
**Complexity:** 8/10
**Impact Scope:** 86 files across inference, training, and documentation

**Critical Paths:**
- Inference API must maintain backward compatibility
- Training pipeline must produce identical outputs
- Perspective correction is optional (can fail gracefully)

**Mitigation:**
- Comprehensive test coverage before migration
- Parallel implementation during transition
- Feature flag for rollback capability

---

## Conclusion

Perspective correction duplication is **significant** but **not blocking** for Phase 1 completion.

**Audit Status:** ✅ Complete
**Consolidation Status:** ⏳ Deferred to Phase 2+
**Reason:** Scope too large (86 files, high complexity, requires dedicated phase)

**Next Steps:**
1. Mark Phase 1.5 as "Audit Complete"
2. Update implementation plan with audit findings
3. Create separate consolidation plan for Phase 2+ if needed
4. Continue with Phase 1 remaining tasks (if any)

---

**Document Status:** Final
**Author:** AI Code Assistant
**Review Required:** Before Phase 2 consolidation
**Related Issues:** None

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
