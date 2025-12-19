---
ads_version: "1.0"
type: "assessment"
category: "code_quality"
status: "active"
priority: "medium"
version: "1.0"
tags: ['refactor', 'performance', 'code-quality', 'perspective-correction']
title: "Perspective Correction Module: Refactor Assessment"
date: "2025-12-20 02:34 (KST)"
branch: "main"
target_file: "ocr/utils/perspective_correction.py"
metrics:
  total_lines: 1551
  functions: 26
  classes: 2
related_incident: "20251220_0130_incident_report_perspective_correction_data_loss"
---

# Perspective Correction Module: Refactor Assessment

## Current State

**File**: `ocr/utils/perspective_correction.py`
**Size**: 1551 lines, 51.9 KB, 26 functions, 2 dataclasses
**Recent Change**: Integrated 1180-line experimental algorithm (data loss fix)

**Refactor Priority**: MEDIUM
**Performance**: ~20ms (acceptable, validated)
**Maintainability**: MEDIUM (monolithic)

---

## Refactor Opportunities

### Priority 1: Modularization

**Problem**: 1551-line monolith, low cohesion
**Proposed**:

```
ocr/utils/perspective_correction/
├── __init__.py          # Public API
├── core.py              # transform, correct_perspective
├── fitting.py           # fit_mask_rectangle + strategies
├── geometry.py          # _order_points, _intersect_lines, etc.
├── validation.py        # _validate_* functions
├── quality_metrics.py   # _compute_* functions
└── types.py             # LineQualityReport, MaskRectangleResult
```

**Effort**: 3-4h
**Risk**: Low
**Benefit**: High (testability, maintainability)

---

### Priority 2: Strategy Pattern

**Problem**: 3 fitting strategies hardcoded

```python
# Current
if use_dominant_extension:
    ...
elif use_regression:
    ...
else:
    ...

# Proposed
strategy = get_fitting_strategy(use_dominant_extension, use_regression)
fitted_quad, used_eps = strategy.fit(hull, **params)
```

**Effort**: 1-2h
**Risk**: Very low
**Benefit**: Medium (extensibility)

---

### Priority 3: Quality Metrics Extraction

**Problem**: 6 metric functions tightly coupled

```python
class QualityMetricsCalculator:
    def compute_all(corners, contour, hull, params) -> QualityMetrics:
        ...
```

**Effort**: 2h
**Risk**: Very low
**Benefit**: High (testability, reusability)

---

## Performance Analysis

**Current**: 13-20ms per image (validated)
**Bottleneck**: None identified
**Recommendation**: **NO optimization needed** - performance acceptable

**Low-priority optimizations** (if <20ms becomes requirement):
- Vectorize `_collect_edge_support_data`: ~2-3ms gain
- Early exit in validators: <1ms gain

---

## Technical Debt

| Item | Severity | Effort | Priority |
|------|----------|--------|----------|
| Monolithic structure | Medium | 3-4h | High |
| No module tests | Medium | 2-3h | Medium |
| Hardcoded strategies | Low | 1-2h | Low |

**Recommendation**: Proceed with Priority 1 modularization in next maintenance window.

---

## Migration Path

1. Create package structure (2h)
2. Move functions to modules (1h)
3. Update `__init__.py` exports (30min)
4. Update imports in codebase (30min)
5. Add unit tests (2h)

**Total**: 3-5h (modularization only), 6-8h (full refactor)

---

## Decision Matrix

| Scenario | Action |
|----------|--------|
| Immediate bug/feature | Skip refactor |
| Maintenance window | **Proceed with Priority 1** |
| Adding fitting strategy | Implement Priority 2 first |
| Performance <20ms needed | Profile, then optimize |

---

## References

- [`perspective_correction.py`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/utils/perspective_correction.py)
- [Recent fix walkthrough](file:///home/vscode/.gemini/antigravity/brain/6ab361a9-24a6-48cc-93a4-afdf559c4250/walkthrough.md)
- [Incident report](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251220_0130_incident_report_perspective_correction_data_loss.md)
