# OCR Directory Detailed Comparison

**Date**: 2025-11-12
**Purpose**: Detailed analysis of `ocr/` directory differences between `main` and `12_refactor/streamlit`

---

## Summary

- **Total files changed**: 24
- **Total changes**: +1,420 insertions, -569 deletions
- **New files in streamlit**: 2 (experiment_registry.py, metadata_callback.py)
- **Files with significant changes**: 8

---

## File-by-File Analysis

### 1. `ocr/experiment_registry.py` ⭐ NEW IN STREAMLIT
- **Status**: Does not exist in main
- **Size**: 126 lines
- **Purpose**: Experiment management with unique IDs and metadata tracking
- **Decision**: ✅ **KEEP FROM STREAMLIT** - New feature, no conflict

---

### 2. `ocr/lightning_modules/callbacks/metadata_callback.py` ⭐ NEW IN STREAMLIT
- **Status**: Does not exist in main
- **Size**: 509 lines
- **Purpose**: Metadata callback for experiment tracking
- **Decision**: ✅ **KEEP FROM STREAMLIT** - New feature, no conflict

---

### 3. `ocr/models/head/db_head.py` ⚠️ CRITICAL
- **Main**: Has numerical stability fix (PLAN-001)
  - Uses `torch.sigmoid()` with input clamping
  - Prevents NaN/Inf from extreme values
- **Streamlit**: Original buggy code
  - Uses `torch.reciprocal(1 + exp(-k*(x-y)))`
  - Can cause numerical overflow
- **Impact**: High - Fixes BUG-20251110-002
- **Decision**: ✅ **TAKE FROM MAIN** - Critical bug fix

---

### 4. `ocr/models/loss/dice_loss.py` ⚠️ IMPORTANT
- **Main**: Basic validation with separate checks
- **Streamlit**: More defensive validation
  - Combined NaN/Inf checks
  - Better error messages
  - Guards against degenerate unions
  - Tolerates minor numeric overshoot
- **Impact**: Medium - Better edge case handling
- **Decision**: ✅ **TAKE FROM STREAMLIT** - More robust

---

### 5. `ocr/lightning_modules/callbacks/unique_checkpoint.py`
- **Main**: Lazy wandb imports (PLAN-003)
- **Streamlit**: Enhanced version + lazy imports
  - More features
  - Better checkpoint handling
- **Impact**: Medium
- **Decision**: ✅ **TAKE FROM STREAMLIT** - Has improvements

---

### 6. `ocr/lightning_modules/callbacks/wandb_completion.py`
- **Main**: Lazy wandb imports (PLAN-003)
- **Streamlit**: Lazy imports + directory creation fix
  - Fixes FileNotFoundError in exception handler
  - Addresses BUG-20251112-014
- **Impact**: High - Fixes bug
- **Decision**: ✅ **TAKE FROM STREAMLIT** - Has the fix

---

### 7. `ocr/utils/polygon_utils.py`
- **Main**: PLAN-002 improvements (polygon validation consolidation)
- **Streamlit**: Different improvements
- **Impact**: Medium - Both have value
- **Decision**: ⚠️ **MERGE MANUALLY** - Need to combine both

---

### 8. `ocr/datasets/db_collate_fn.py`
- **Main**: PLAN-002 improvements (shared polygon validators)
- **Streamlit**: Different improvements
- **Impact**: Medium - Both have value
- **Decision**: ⚠️ **MERGE MANUALLY** - Need to combine both

---

### 9. `ocr/utils/orientation.py`
- **Main**: Unknown changes
- **Streamlit**: Refactored (83 lines removed)
- **Impact**: Low-Medium
- **Decision**: ⚠️ **REVIEW** - Check what was removed

---

### 10. `ocr/utils/orientation_constants.py` ⭐ NEW IN STREAMLIT
- **Status**: New file in streamlit (154 lines)
- **Purpose**: Orientation constants extracted from orientation.py
- **Decision**: ✅ **KEEP FROM STREAMLIT** - Part of refactoring

---

### 11. `ocr/utils/polygon_utils.py`
- **Main**: PLAN-002 improvements
- **Streamlit**: Refactored (167 lines removed)
- **Impact**: Medium
- **Decision**: ⚠️ **REVIEW CAREFULLY** - Significant refactoring

---

### 12. `ocr/datasets/preprocessing/advanced_noise_elimination.py`
- **Changes**: 39 lines modified
- **Impact**: Low-Medium
- **Decision**: ⚠️ **REVIEW** - Check what changed

---

### 13. `ocr/datasets/preprocessing/contracts.py`
- **Changes**: 28 lines modified
- **Impact**: Low-Medium
- **Decision**: ⚠️ **REVIEW** - Check what changed

---

### 14. `ocr/datasets/preprocessing/document_flattening.py`
- **Changes**: 77 lines modified
- **Impact**: Medium
- **Decision**: ⚠️ **REVIEW** - Significant changes

---

### 15. `ocr/datasets/preprocessing/external.py`
- **Changes**: 17 lines added
- **Impact**: Low
- **Decision**: ⚠️ **REVIEW** - Check what was added

---

### 16. `ocr/datasets/preprocessing/intelligent_brightness.py`
- **Changes**: 4 lines removed
- **Impact**: Low
- **Decision**: ⚠️ **REVIEW** - Check what was removed

---

### 17. `ocr/lightning_modules/callbacks/wandb_image_logging.py`
- **Main**: Lazy wandb imports (PLAN-003)
- **Streamlit**: Different improvements (122 lines changed)
- **Impact**: Medium
- **Decision**: ⚠️ **REVIEW** - Check streamlit's improvements

---

### 18. `ocr/lightning_modules/ocr_pl.py`
- **Changes**: 19 lines modified
- **Impact**: Medium
- **Decision**: ⚠️ **REVIEW** - Check what changed

---

### 19. `ocr/lightning_modules/utils/model_utils.py`
- **Changes**: 50 lines modified
- **Impact**: Medium
- **Decision**: ⚠️ **REVIEW** - Check what changed

---

### 20. `ocr/utils/convert_submission.py`
- **Changes**: 114 lines modified
- **Impact**: Medium-High
- **Decision**: ⚠️ **REVIEW** - Significant changes

---

### 21. `ocr/utils/path_utils.py`
- **Changes**: 60 lines modified
- **Impact**: Medium
- **Decision**: ⚠️ **REVIEW** - Check what changed

---

### 22. `ocr/utils/wandb_utils.py`
- **Main**: Lazy wandb imports (PLAN-003)
- **Streamlit**: Different changes (6 lines)
- **Impact**: Low-Medium
- **Decision**: ⚠️ **REVIEW** - Check streamlit's changes

---

### 23. `ocr/datasets/__init__.py`
- **Changes**: 13 lines modified
- **Impact**: Low
- **Decision**: ⚠️ **REVIEW** - Check what changed

---

### 24. `ocr/lightning_modules/__init__.py`
- **Changes**: 2 lines modified
- **Impact**: Low
- **Decision**: ⚠️ **REVIEW** - Check what changed

---

## Recommendations

### Immediate Actions (High Priority)
1. ✅ **Port `db_head.py` from main** - Critical bug fix
2. ✅ **Keep `experiment_registry.py` from streamlit** - New feature
3. ✅ **Keep `metadata_callback.py` from streamlit** - New feature
4. ✅ **Keep `wandb_completion.py` from streamlit** - Has bug fix

### Review Needed (Medium Priority)
5. ⚠️ **Review `polygon_utils.py`** - Both have improvements, need merge
6. ⚠️ **Review `db_collate_fn.py`** - Both have improvements, need merge
7. ⚠️ **Review `dice_loss.py`** - Take streamlit (more defensive)
8. ⚠️ **Review `unique_checkpoint.py`** - Take streamlit (has enhancements)

### Low Priority Review
9. ⚠️ Review remaining 16 files for context-specific changes

---

## Merge Strategy for OCR Directory

### Option A: Take Streamlit, Port Critical Fixes (RECOMMENDED)
1. Start with streamlit's `ocr/` directory
2. Port `db_head.py` fix from main
3. Manually merge `polygon_utils.py` and `db_collate_fn.py`
4. Review other files case by case

**Pros**:
- Preserves new features (experiment_registry, metadata_callback)
- Keeps streamlit's improvements
- Only need to port critical fixes

**Cons**:
- Need to manually merge some files
- May miss some main improvements

### Option B: Take Main, Port Streamlit Features
1. Start with main's `ocr/` directory
2. Port `experiment_registry.py` from streamlit
3. Port `metadata_callback.py` from streamlit
4. Port other streamlit improvements

**Pros**:
- Has all PLAN-001/002/003 fixes
- Cleaner starting point

**Cons**:
- More work to port streamlit features
- May lose streamlit-specific improvements

---

## Decision: Option A (Recommended)

**Rationale**:
- Streamlit has new features that are valuable
- Only need to port one critical fix (`db_head.py`)
- Most other files can be reviewed case by case
- Faster to implement

---

**Last Updated**: 2025-11-12

