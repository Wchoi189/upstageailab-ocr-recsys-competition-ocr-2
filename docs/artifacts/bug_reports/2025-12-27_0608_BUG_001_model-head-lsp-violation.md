---
ads_version: "1.0"
type: "bug_report"
category: "troubleshooting"
status: "active"
severity: "high"
version: "1.0"
tags: ['bug', 'issue', 'troubleshooting']
title: "BaseHead.get_polygons_from_maps Signature Incompatibility (LSP Violation)"
date: "2025-12-27 15:08 (KST)"
branch: "claude/code-quality-plan-oApCV"
affected_components: "ocr.models.head, ocr.models.core.base_classes"
---

# Bug Report: BaseHead.get_polygons_from_maps Signature Incompatibility (LSP Violation)

## Bug ID
BUG-20251227-001

## Summary
Abstract base class `BaseHead.get_polygons_from_maps()` defines an abstract method signature that **does not match** any of its implementations in `DBHead` and `CRAFTHead`. This is a **Liskov Substitution Principle (LSP) violation** that breaks type safety and polymorphism.

## Environment
- **Python Version**: 3.11+
- **Type Checker**: mypy 1.x with `check_untyped_defs=true`
- **Affected Components**: `ocr.models.core.base_classes`, `ocr.models.head.{db_head,craft_head}`

## Steps to Reproduce
1. Enable `check_untyped_defs = true` in `pyproject.toml`
2. Run: `uv run mypy ocr/models/head/ --config-file pyproject.toml`
3. Observe signature incompatibility errors

## Expected Behavior
Subclass method signatures should match the abstract base class contract to enable polymorphism and type safety.

## Actual Behavior
**Base Class Signature (BaseHead:144):**
```python
@abstractmethod
def get_polygons_from_maps(
    self,
    pred_maps: dict[str, Tensor],
    ground_truth: dict[str, Tensor] | None = None
) -> list[list[list[float]]]:
```

**Actual Implementation (DBHead, CRAFTHead):**
```python
def get_polygons_from_maps(self, batch, pred):
    return self.postprocess.represent(batch, pred)
```

## Error Messages
```
ocr/models/head/db_head.py:210: error: Signature of "get_polygons_from_maps" incompatible with supertype "BaseHead"
ocr/models/head/craft_head.py:57: error: Signature of "get_polygons_from_maps" incompatible with supertype "BaseHead"
```

## Impact
- **Severity**: HIGH - Architecture violation
- **Type Safety**: Broken - mypy cannot verify correct usage
- **Polymorphism**: Fails - code expecting `BaseHead` interface receives wrong types
- **Call Sites Affected**: 15+ locations in inference, training, and testing code
- **Workaround**: Disable `check_untyped_defs` or add mypy override (not recommended)

## Investigation

### Root Cause Analysis
- **Cause**: Base class signature was written as **design documentation** but never enforced
- **Location**: `ocr/models/core/base_classes.py:144-157`
- **Trigger**: Enabling stricter type checking (`check_untyped_defs=true`)
- **Evidence**: ALL 15+ call sites use `(batch, pred)` signature, NONE use `(pred_maps, ground_truth)`

### Call Site Analysis
```python
# Inference (postprocess.py:214)
polygons_result = head.get_polygons_from_maps(batch, predictions)

# Training (ocr_pl.py:3 locations)
boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)

# Architecture wrapper (architecture.py)
return self.head.get_polygons_from_maps(batch, pred)
```

**Findings:**
- `batch` is dict with: images, shape, filename, inverse_matrix
- `pred` is raw model predictions dict
- Returns `tuple[boxes, scores]` NOT `list[list[list[float]]]`
- `ground_truth` parameter never used anywhere

### Related Issues
- Code Quality Plan Phase 3: check_untyped_defs enforcement
- Implementation Plan: `docs/artifacts/implementation_plans/2025-12-27_model_head_signature_fix.md`

## Proposed Solution

### Fix Strategy
**Option 1: Update Base Class to Match Reality** ⭐ **RECOMMENDED**

Update `BaseHead` abstract method signature to match actual implementations:
```python
@abstractmethod
def get_polygons_from_maps(
    self,
    batch: dict[str, Any],
    pred: dict[str, Tensor]
) -> tuple[list[list[list[float]]], list[list[float]]]:
```

**Rationale:**
- Minimal changes (4 files: base + 2 heads + wrapper)
- No functional changes (type hints only)
- Matches ALL existing implementations
- Low risk, high reward

### Implementation Plan
1. Update `BaseHead` signature with proper types and documentation
2. Add type annotations to `DBHead.get_polygons_from_maps()`
3. Add type annotations to `CRAFTHead.get_polygons_from_maps()`
4. Update `OCRModel` wrapper signature
5. Verify with mypy (should resolve 2 errors)
6. Run tests: `pytest tests/unit/test_architecture.py tests/integration/`

**Detailed Plan:** See `docs/artifacts/implementation_plans/2025-12-27_model_head_signature_fix.md`

### Testing Plan
- ✅ Type checking: mypy should pass with 0 signature errors
- ✅ Unit tests: `test_architecture.py::test_get_polygons_from_maps`
- ✅ Integration: Full inference pipeline test
- ✅ Regression: All existing tests should pass unchanged

### Risk Assessment
- **Breaking Changes**: NONE (type hints only)
- **Runtime Impact**: NONE
- **Test Updates**: NONE required
- **Rollback**: Simple (revert 4 files or add mypy override)

## Status
- [x] Confirmed (via mypy + code analysis)
- [ ] Investigating
- [ ] Fix in progress
- [ ] Fixed
- [ ] Verified

## Assignee
Code Quality Team

## Priority
**HIGH** - Architecture bug affecting type safety across model heads

**Justification:**
- Prevents future refactoring bugs
- Enables full type checking coverage
- Quick fix (2-3 hours) for high-value improvement

---

*This bug report follows the project's standardized format for issue tracking.*
