---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
priority: "medium"
version: "1.0"
tags: ['refactor', 'modularization', 'perspective-correction', 'code-quality']
title: "Perspective Correction Module Modularization"
date: "2025-12-20 02:44 (KST)"
branch: "main"
related_assessment: "2025-12-20_0234_assessment-perspective-correction-refactor"
source_file: "ocr/utils/perspective_correction.py"
effort_estimate: "3-5 hours"
worker_handoff: true
---

# Perspective Correction Module Modularization

## Objective

Refactor monolithic `ocr/utils/perspective_correction.py` (1551 lines) into modular package structure for improved testability and maintainability.

**Handoff Note**: This plan is designed for independent worker execution. Each step is self-contained with precise file locations and code blocks.

---

## Current State

**File**: `ocr/utils/perspective_correction.py`
- 1551 lines, 26 functions, 2 dataclasses
- All functionality in single file
- Low cohesion (geometry, validation, metrics, fitting mixed)

**Consumers** (must remain compatible):
- `ocr/inference/preprocess.py` (imports `correct_perspective_from_mask`, `remove_background_and_mask`)
- `ocr/inference/orchestrator.py` (imports `transform_polygons_inverse`)

---

## Target Structure

```
ocr/utils/perspective_correction/
├── __init__.py                    # Public API exports
├── types.py                       # Dataclasses
├── core.py                        # Public transform functions
├── fitting.py                     # fit_mask_rectangle + strategies
├── geometry.py                    # Geometric calculations
├── validation.py                  # Validators
└── quality_metrics.py             # Quality metrics
```

**Backward Compatibility**: All imports from `ocr.utils.perspective_correction` must continue working.

---

## Implementation Steps

### Step 1: Create Package Directory Structure

**Action**: Create new package directory

```bash
mkdir -p ocr/utils/perspective_correction
touch ocr/utils/perspective_correction/__init__.py
touch ocr/utils/perspective_correction/types.py
touch ocr/utils/perspective_correction/core.py
touch ocr/utils/perspective_correction/fitting.py
touch ocr/utils/perspective_correction/geometry.py
touch ocr/utils/perspective_correction/validation.py
touch ocr/utils/perspective_correction/quality_metrics.py
```

**Verification**: Confirm 7 files created in `ocr/utils/perspective_correction/`

---

### Step 2: Create `types.py` - Dataclasses

**File**: `ocr/utils/perspective_correction/types.py`

**Content**:
```python
from __future__ import annotations

"""Data types for perspective correction."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LineQualityReport:
    """Advanced quality metrics for fitted rectangle."""

    decision: str
    metrics: dict[str, Any]
    passes: dict[str, bool]
    fail_reasons: list[str]


@dataclass
class MaskRectangleResult:
    """Result of fitting a rectangle to a foreground mask."""

    corners: np.ndarray | None
    raw_corners: np.ndarray | None
    contour_area: float
    hull_area: float
    mask_area: float
    contour: np.ndarray | None
    hull: np.ndarray | None
    reason: str | None = None
    line_quality: LineQualityReport | None = None
    used_epsilon: float | None = None


__all__ = ["LineQualityReport", "MaskRectangleResult"]
```

**Source**: Extract from lines 33-56 of original `perspective_correction.py`

---

### Step 3: Create `geometry.py` - Geometric Utilities

**File**: `ocr/utils/perspective_correction/geometry.py`

**Functions to migrate** (from original file):
- `_order_points()` (lines ~247-260)
- `_compute_edge_vectors()` (lines ~263-267)
- `_intersect_lines()` (lines ~539-563)
- `_blend_corners()` (lines ~1067-1076)
- `_geometric_synthesis()` (lines ~140-236)

**Template**:
```python
from __future__ import annotations

"""Geometric calculations for perspective correction."""

import math

import cv2
import numpy as np


def _order_points(points: np.ndarray) -> np.ndarray:
    """Order quadrilateral corners as TL, TR, BR, BL."""
    # Copy lines 247-260 from original
    ...


def _compute_edge_vectors(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute edge vectors and lengths from ordered corners."""
    # Copy lines 263-267 from original
    ...


def _intersect_lines(...) -> np.ndarray | None:
    """Find intersection of two lines."""
    # Copy lines 539-563 from original
    ...


def _blend_corners(...) -> np.ndarray:
    """Blend two ordered quadrilaterals."""
    # Copy lines 1067-1076 from original
    ...


def _geometric_synthesis(...) -> np.ndarray | None:
    """Geometric Synthesis: Intersect fitted_quad with bbox_corners."""
    # Copy lines 140-236 from original
    # NOTE: This function calls _order_points, ensure it's imported locally
    ...


__all__ = [
    "_order_points",
    "_compute_edge_vectors",
    "_intersect_lines",
    "_blend_corners",
    "_geometric_synthesis",
]
```

**Worker Instructions**:
1. Copy each function verbatim from original file
2. Preserve all docstrings, comments, type hints
3. Add `import math` at top if using `math.atan2`
4. Internal calls like `_order_points()` within `_geometric_synthesis()` work automatically (same module)

---

### Step 4: Create `validation.py` - Validators

**File**: `ocr/utils/perspective_correction/validation.py`

**Functions to migrate**:
- `_validate_edge_angles()` (lines ~270-302)
- `_validate_edge_lengths()` (lines ~305-338)
- `_validate_contour_alignment()` (lines ~341-395)
- `_validate_contour_segments()` (lines ~398-422)

**Template**:
```python
from __future__ import annotations

"""Validation functions for perspective correction."""

import math

import cv2
import numpy as np

from .geometry import _compute_edge_vectors


def _validate_edge_angles(corners: np.ndarray, angle_tolerance_deg: float = 15.0) -> bool:
    """Validate that edges form approximately right angles."""
    # Copy lines 270-302 from original
    # NOTE: Calls _compute_edge_vectors - import from .geometry
    ...


def _validate_edge_lengths(...) -> bool:
    """Validate edge lengths form reasonable document proportions."""
    # Copy lines 305-338 from original
    ...


def _validate_contour_alignment(...) -> bool:
    """Cross-validate fitted rectangle against mask bounding box."""
    # Copy lines 341-395 from original
    ...


def _validate_contour_segments(...) -> bool:
    """Validate contour has sufficient structure."""
    # Copy lines 398-422 from original
    ...


__all__ = [
    "_validate_edge_angles",
    "_validate_edge_lengths",
    "_validate_contour_alignment",
    "_validate_contour_segments",
]
```

**Key Dependency**: Import `_compute_edge_vectors` from `.geometry`

---

### Step 5: Create `quality_metrics.py` - Quality Metrics

**File**: `ocr/utils/perspective_correction/quality_metrics.py`

**Functions to migrate**:
- `_collect_edge_support_data()` (lines ~844-896)
- `_compute_edge_support_metrics()` (lines ~899-952)
- `_compute_linearity_rmse()` (lines ~955-985)
- `_compute_solidity_metrics()` (lines ~988-1007)
- `_compute_corner_sharpness_deviation()` (lines ~1010-1048)
- `_compute_parallelism_misalignment()` (lines ~1051-1064)

**Template**:
```python
from __future__ import annotations

"""Quality metrics computation for fitted rectangles."""

import math
from typing import Any

import cv2
import numpy as np

from .geometry import _compute_edge_vectors


def _collect_edge_support_data(...) -> list[dict[str, Any]]:
    """Collect per-edge point projections/distances."""
    # Copy lines 844-896 from original
    ...


def _compute_edge_support_metrics(...) -> dict[str, Any]:
    """Compute edge coverage/support metrics."""
    # Copy lines 899-952 from original
    # NOTE: Calls _collect_edge_support_data (same module)
    ...


def _compute_linearity_rmse(...) -> dict[str, Any]:
    """Compute RMSE of contour distances to each fitted edge."""
    # Copy lines 955-985 from original
    ...


def _compute_solidity_metrics(...) -> dict[str, float]:
    """Compute solidity/rectangularity style metrics."""
    # Copy lines 988-1007 from original
    ...


def _compute_corner_sharpness_deviation(...) -> dict[str, float] | None:
    """Measure maximum and mean deviation from 90°."""
    # Copy lines 1010-1048 from original
    ...


def _compute_parallelism_misalignment(...) -> float | None:
    """Return maximum angular deviation between opposite edges."""
    # Copy lines 1051-1064 from original
    # NOTE: Calls _compute_edge_vectors - import from .geometry
    ...


__all__ = [
    "_collect_edge_support_data",
    "_compute_edge_support_metrics",
    "_compute_linearity_rmse",
    "_compute_solidity_metrics",
    "_compute_corner_sharpness_deviation",
    "_compute_parallelism_misalignment",
]
```

**Key Dependencies**: Import `_compute_edge_vectors` from `.geometry`

---

### Step 6: Create `fitting.py` - Fitting Logic

**File**: `ocr/utils/perspective_correction/fitting.py`

**Functions to migrate**:
- `_prepare_mask()` (lines ~115-120)
- `_extract_largest_component()` (lines ~123-137)
- `_fit_quadrilateral_from_hull()` (lines ~425-510)
- `_fit_quadrilateral_regression()` (lines ~566-655)
- `_fit_quadrilateral_dominant_extension()` (lines ~658-841)
- `fit_mask_rectangle()` (lines ~1079-1413) **[MAIN FUNCTION]**

**Template**:
```python
from __future__ import annotations

"""Mask-based rectangle fitting logic."""

import math
from typing import Any

import cv2
import numpy as np

from .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points
from .quality_metrics import (
    _compute_corner_sharpness_deviation,
    _compute_edge_support_metrics,
    _compute_linearity_rmse,
    _compute_parallelism_misalignment,
    _compute_solidity_metrics,
)
from .types import LineQualityReport, MaskRectangleResult
from .validation import (
    _validate_contour_alignment,
    _validate_contour_segments,
    _validate_edge_angles,
    _validate_edge_lengths,
)


def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is binary {0,255} uint8."""
    # Copy lines 115-120 from original
    ...


def _extract_largest_component(mask: np.ndarray) -> np.ndarray:
    """Return binary mask of largest connected component."""
    # Copy lines 123-137 from original
    ...


def _fit_quadrilateral_from_hull(...) -> tuple[np.ndarray | None, float]:
    """Fit quadrilateral using adaptive approxPolyDP."""
    # Copy lines 425-510 from original
    ...


def _fit_quadrilateral_regression(...) -> tuple[np.ndarray | None, float]:
    """Fit quadrilateral by side-based line regression."""
    # Copy lines 566-655 from original
    # NOTE: Calls _intersect_lines from .geometry
    ...


def _fit_quadrilateral_dominant_extension(...) -> tuple[np.ndarray | None, float]:
    """Fit quadrilateral via angle-based bucketing."""
    # Copy lines 658-841 from original
    # NOTE: Calls _intersect_lines from .geometry
    ...


def fit_mask_rectangle(...) -> MaskRectangleResult:
    """Fit rectangle corners directly from binary mask."""
    # Copy lines 1079-1413 from original (LARGEST FUNCTION - 335 lines)
    # NOTE: Imports all helpers from other modules
    ...


__all__ = ["fit_mask_rectangle"]
```

**Critical**: This module imports from ALL other modules. Ensure imports at top are correct.

---

### Step 7: Create `core.py` - Public API Functions

**File**: `ocr/utils/perspective_correction/core.py`

**Functions to migrate**:
- `calculate_target_dimensions()` (lines ~59-80)
- `four_point_transform()` (lines ~83-112)
- `correct_perspective_from_mask()` (lines ~1416-1483)
- `remove_background_and_mask()` (lines ~1486-1514)
- `transform_polygons_inverse()` (lines ~1517-1548)

**Template**:
```python
from __future__ import annotations

"""Core perspective correction functions."""

import logging
from typing import Any

import cv2
import numpy as np

from .fitting import fit_mask_rectangle
from .types import MaskRectangleResult

# Optional rembg dependency
try:
    from rembg import remove as _rembg_remove

    _REMBG_AVAILABLE = True
except Exception:
    _rembg_remove = None
    _REMBG_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def calculate_target_dimensions(pts: np.ndarray) -> tuple[int, int]:
    """Calculate target dimensions using Max-Edge rule."""
    # Copy lines 59-80 from original
    ...


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform."""
    # Copy lines 83-112 from original
    # NOTE: Calls calculate_target_dimensions (same module)
    ...


def correct_perspective_from_mask(...) -> tuple[np.ndarray, MaskRectangleResult] | tuple[np.ndarray, MaskRectangleResult, np.ndarray]:
    """High-level perspective correction from mask."""
    # Copy lines 1416-1483 from original
    # NOTE: Calls fit_mask_rectangle and four_point_transform
    ...


def remove_background_and_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove background using rembg."""
    # Copy lines 1486-1514 from original
    ...


def transform_polygons_inverse(...) -> list[np.ndarray]:
    """Apply inverse perspective transform to polygons."""
    # Copy lines 1517-1548 from original
    ...


__all__ = [
    "calculate_target_dimensions",
    "four_point_transform",
    "correct_perspective_from_mask",
    "remove_background_and_mask",
    "transform_polygons_inverse",
]
```

**Key Dependencies**: Import `fit_mask_rectangle` from `.fitting`, `MaskRectangleResult` from `.types`

---

### Step 8: Create `__init__.py` - Public API Exports

**File**: `ocr/utils/perspective_correction/__init__.py`

**Purpose**: Re-export all public symbols for backward compatibility

**Content**:
```python
"""Perspective correction utilities for OCR images.

This module provides mask-based rectangle fitting and perspective transformation.
"""

from .core import (
    calculate_target_dimensions,
    correct_perspective_from_mask,
    four_point_transform,
    remove_background_and_mask,
    transform_polygons_inverse,
)
from .fitting import fit_mask_rectangle
from .types import LineQualityReport, MaskRectangleResult

__all__ = [
    # Types
    "LineQualityReport",
    "MaskRectangleResult",
    # Core functions
    "calculate_target_dimensions",
    "four_point_transform",
    "correct_perspective_from_mask",
    "remove_background_and_mask",
    "transform_polygons_inverse",
    # Fitting
    "fit_mask_rectangle",
]
```

**Verification**: Existing imports like `from ocr.utils.perspective_correction import correct_perspective_from_mask` must still work.

---

### Step 9: Backup Original File

**Action**: Rename original file for rollback safety

```bash
mv ocr/utils/perspective_correction.py ocr/utils/perspective_correction.py.backup
```

**Note**: Keep backup until validation passes.

---

### Step 10: Validation

**Tests to run**:

```bash
# 1. Import test
python3 -c "
from ocr.utils.perspective_correction import (
    fit_mask_rectangle,
    LineQualityReport,
    MaskRectangleResult,
    calculate_target_dimensions,
    four_point_transform,
    correct_perspective_from_mask,
    remove_background_and_mask,
    transform_polygons_inverse,
)
print('✅ All imports successful')
"

# 2. Basic functionality test
python3 -c "
import numpy as np
from ocr.utils.perspective_correction import fit_mask_rectangle

mask = np.zeros((500, 800), dtype=np.uint8)
mask[50:450, 100:700] = 255

result = fit_mask_rectangle(mask, use_dominant_extension=True)
assert result.corners is not None
assert result.corners.shape == (4, 2)
print(f'✅ Functional test passed: corners shape = {result.corners.shape}')
"

# 3. Regression test (if validation script exists)
python3 validate_perspective_fix_simple.py
```

**Success Criteria**:
- All imports work
- Basic function test passes
- Regression test shows 0px data loss (if script available)

---

## Module Summary

| Module | Lines | Functions | Imports From |
|--------|-------|-----------|--------------|
| `types.py` | ~30 | 2 dataclasses | None |
| `geometry.py` | ~120 | 5 | None |
| `validation.py` | ~140 | 4 | geometry |
| `quality_metrics.py` | ~200 | 6 | geometry |
| `fitting.py` | ~550 | 7 | geometry, validation, quality_metrics, types |
| `core.py` | ~200 | 5 | fitting, types |
| `__init__.py` | ~30 | 0 (exports) | core, fitting, types |

**Total**: ~1270 lines (vs 1551 original - some duplicate docs removed)

---

## Dependency Graph

```
types.py (no dependencies)
  ↑
geometry.py (no dependencies)
  ↑
validation.py ──→ geometry
  ↑
quality_metrics.py ──→ geometry
  ↑
fitting.py ──→ geometry, validation, quality_metrics, types
  ↑
core.py ──→ fitting, types
  ↑
__init__.py ──→ core, fitting, types
```

**Migration Order**: Follow bottom-up (types → geometry → validation/quality → fitting → core → init)

---

## Rollback Plan

**If validation fails**:

```bash
# Restore original file
rm -rf ocr/utils/perspective_correction/
mv ocr/utils/perspective_correction.py.backup ocr/utils/perspective_correction.py

# Verify restoration
python3 -c "from ocr.utils.perspective_correction import fit_mask_rectangle; print('✅ Rollback successful')"
```

---

## Common Pitfalls

1. **Circular imports**: Follow dependency graph strictly (bottom-up migration)
2. **Missing imports**: Each function that calls `_order_points()` needs `from .geometry import _order_points`
3. **Type hints**: Preserve `np.ndarray | None` syntax (requires Python 3.10+)
4. **Relative imports**: Use `.geometry`, `.types` (dot prefix for same package)
5. **`__all__` exports**: Ensure every module defines `__all__` for clarity

---

## Checklist

**Pre-work**:
- [ ] Read entire plan
- [ ] Confirm Python 3.10+ environment
- [ ] Create backup of original file

**Implementation** (follow order):
- [ ] Step 1: Create directory structure
- [ ] Step 2: Create `types.py`
- [ ] Step 3: Create `geometry.py`
- [ ] Step 4: Create `validation.py`
- [ ] Step 5: Create `quality_metrics.py`
- [ ] Step 6: Create `fitting.py`
- [ ] Step 7: Create `core.py`
- [ ] Step 8: Create `__init__.py`
- [ ] Step 9: Backup original file
- [ ] Step 10: Run validation tests

**Post-work**:
- [ ] All import tests pass
- [ ] Functional test passes
- [ ] No regressions in existing code
- [ ] Remove `.backup` file (only after full validation)

---

## Estimated Effort

- **Setup (Steps 1-2)**: 15 minutes
- **Geometry/Validation (Steps 3-4)**: 45 minutes
- **Quality Metrics (Step 5)**: 30 minutes
- **Fitting (Step 6)**: 90 minutes (largest module)
- **Core + Init (Steps 7-8)**: 45 minutes
- **Validation (Steps 9-10)**: 30 minutes

**Total**: 3.5-4 hours (conservative estimate for careful work)

---

## References

- **Original File**: [`perspective_correction.py`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/utils/perspective_correction.py)
- **Assessment**: [`2025-12-20_0234_assessment-perspective-correction-refactor.md`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/artifacts/assessments/2025-12-20_0234_assessment-perspective-correction-refactor.md)
- **Recent Fix Walkthrough**: [Perspective Correction Fix](file:///home/vscode/.gemini/antigravity/brain/6ab361a9-24a6-48cc-93a4-afdf559c4250/walkthrough.md)
