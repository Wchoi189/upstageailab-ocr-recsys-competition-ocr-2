# Commit Analysis: Dual Architecture Restoration Issue

**Date**: 2026-01-20
**Issue**: CI fixes inadvertently reversed "Domains First" architecture refactoring
**Branch**: main (merged from claude/refactor-agentqms-framework-Wx2i3)
**Commits Analyzed**: 591729c, bd27a23, ba614b6, bb274c6, c2b8f64

---

## Timeline of Events

### 1. **Commit 7eef131** (Jan 17-18) - "Domains First" Refactor Complete ✅
**Status**: Successful architecture refactor

**Key Changes**:
- `ocr/core/lightning/ocr_pl.py` → **RENAMED** to `ocr_pl.py.deprecated`
- Functionality **SPLIT** into domain-specific modules:
  - `ocr/domains/detection/module.py` (DetectionPLModule)
  - `ocr/domains/recognition/module.py` (RecognitionPLModule)
  - `ocr/core/lightning/base.py` (shared base class)
- Created `ocr/data/datasets/__init__.py` with:
  - Lazy imports via `__getattr__`
  - `get_datasets_by_cfg()` factory function
  - Proper domain forwarding (DBCollateFN → `domains.detection.data.collate_db`)

**Architecture**: Clean domain separation achieved ✅

---

### 2. **Commit bd27a23** (Jan 20, 09:39 UTC) - CI Fix That Broke Architecture ❌
**Author**: Claude (web worker)
**Intent**: Fix 16 failing test imports
**Method**: Create backward-compatibility shims

#### What Was Created:
```
ocr/data/datasets/db_collate_fn.py          (shim → domains.detection.data.collate_db)
ocr/core/utils/geometry_utils.py            (shim → domains.detection.utils.geometry)
ocr/core/utils/polygon_utils.py             (shim → domains.detection.utils.polygons)
ocr/core/inference/engine.py                (shim → pipelines.engine)
ocr/core/evaluation/evaluator.py            (shim → domains.detection.evaluation)
ocr/core/lightning/ocr_pl.py                (RESTORED from .deprecated)
experiment_manager/src/etk/compass.py       (shim → project_compass.src.core)
```

#### What Was DESTROYED:
```diff
- ocr/data/datasets/__init__.py  (88 lines → 1 line)

BEFORE (functional):
  • Lazy imports via __getattr__
  • get_datasets_by_cfg() factory (critical!)
  • Domain forwarding

AFTER (gutted):
  """Legacy data.datasets package - compatibility shim."""
```

**Critical Loss**: `get_datasets_by_cfg()` function deleted! This was likely used by:
- `ocr/data/lightning_data.py` (OCRDataPLModule)
- `ocr/pipelines/orchestrator.py` (OCRProjectOrchestrator)
- Training scripts

---

### 3. **Commit ba614b6** (Jan 20, 11:51 UTC) - Nuclear Refactor ⚠️
**Intent**: Remove ALL legacy AgentQMS tools
**Scope**: AgentQMS/* (NOT ocr/*)

**OCR-Related Changes**:
- Added **deprecation warnings** to compatibility shims
- Did **NOT** delete the shims (kept for backward compatibility)
- Documented shims will be removed in v0.4.0

**Issue**: The "nuclear refactor" was scoped to AgentQMS, but the OCR shims created in bd27a23 were left in place with deprecation warnings. This means:
1. Legacy architecture still active
2. Dual import paths still exist
3. `get_datasets_by_cfg()` still missing

---

### 4. **Commits bb274c6 + c2b8f64** (Jan 20, 12:08 + later) - AgentQMS Fixes
**Scope**: AgentQMS automation and CI workflows
**OCR Impact**: None (these were about restoring AgentQMS tool implementations for qms CLI)

---

## Root Cause Analysis

### Why Did This Happen?

1. **CI Worker Had No Context on Architecture Goals**
   - Worker was asked: "Fix failing CI tests"
   - Worker did NOT have access to:
     - `project_compass/roadmap/ocr-domain-refactor.yml`
     - `docs/artifacts/implementation_plans/2026-01-17_0340_implementation_plan_ocr-proposed-directory-tree.md`
     - Understanding of "Domains First" principles

2. **Shims Were the "Quick Fix"**
   - Creating shims took ~10 minutes
   - Updating 16 test files would take ~30-60 minutes
   - Worker optimized for speed, not architecture

3. **get_datasets_by_cfg() Deletion Was Unintentional**
   - Worker replaced entire `ocr/data/datasets/__init__.py` (88 lines → 1 line)
   - Likely didn't realize this file contained critical factory function
   - May have assumed function was moved elsewhere during refactor

---

## Current State Assessment

### What's Broken Now?

1. ✅ **Tests Pass** (via compatibility shims)
2. ❌ **Architecture Violated** (legacy paths active)
3. ❌ **`get_datasets_by_cfg()` Missing** (may break training)
4. ❌ **Dual Import Paths** (confusion for developers)
5. ❌ **`ocr_pl.py` Restored** (contradicts domain separation)

### Critical Question: Is Training Still Working?

Need to check if `get_datasets_by_cfg()` was already replaced by:
- Direct Hydra instantiation in orchestrator?
- New factory method elsewhere?

**Action Required**: Check these files:
```bash
git log --all --oneline -S "get_datasets_by_cfg" | head -10
grep -r "get_datasets_by_cfg" ocr/ runners/
```

---

## Impact Assessment

### Files That Should NOT Exist (Legacy Shims):
| File | Re-exports From | Lines | Clutter Score |
|------|----------------|-------|--------------|
| `ocr/core/lightning/ocr_pl.py` | ❌ Full implementation (516 lines!) | 516 | 🔴 CRITICAL |
| `ocr/data/datasets/db_collate_fn.py` | domains.detection.data.collate_db | 15 | 🟡 Medium |
| `ocr/core/utils/geometry_utils.py` | domains.detection.utils.geometry | 25 | 🟡 Medium |
| `ocr/core/utils/polygon_utils.py` | domains.detection.utils.polygons | 31 | 🟡 Medium |
| `ocr/core/inference/engine.py` | pipelines.engine | 23 | 🟡 Medium |
| `ocr/core/evaluation/evaluator.py` | domains.detection.evaluation | 15 | 🟡 Medium |
| `experiment_manager/src/etk/compass.py` | project_compass.src.core | 39 | 🟡 Medium |

**Total Clutter**: 664 lines of legacy code

### Files That Were Destroyed:
| File | Status | Impact |
|------|--------|--------|
| `ocr/data/datasets/__init__.py` | 88 lines → 1 line | 🔴 `get_datasets_by_cfg()` lost |

---

## Recommended Fix Strategy

### Option 1: Clean Revert (Your Preference)
```bash
# Revert the CI fix commit
git revert bd27a23

# This will:
# 1. Restore ocr/data/datasets/__init__.py (88 lines)
# 2. Delete all compatibility shims
# 3. Re-break the 16 test imports

# Then: Update test imports properly (30-60 min)
```

### Option 2: Surgical Fix (Preserve What Works)
```bash
# Keep deprecation warnings from ba614b6
# But restore get_datasets_by_cfg() and fix tests properly

# Step 1: Restore ocr/data/datasets/__init__.py
git show bd27a23^:ocr/data/datasets/__init__.py > ocr/data/datasets/__init__.py

# Step 2: Delete ocr/core/lightning/ocr_pl.py (most critical)
rm ocr/core/lightning/ocr_pl.py

# Step 3: Update test imports to use domain paths
# (Can keep small shims for geometry_utils etc. temporarily)

# Step 4: Test and commit
```

### Option 3: Verify Training First
```bash
# Before any revert, check if training still works
python runners/train.py domain=detection trainer.fast_dev_run=true

# If it works: get_datasets_by_cfg() wasn't critical
# If it breaks: Need urgent restoration
```

---

## Test Import Fixes Required (If Reverting)

When bd27a23 is reverted, these 16 test files will break:

### Detection Domain Tests (10 files):
```python
# tests/integration/test_collate_integration.py
- from ocr.data.datasets.db_collate_fn import DBCollateFN
+ from ocr.domains.detection.data.collate_db import DBCollateFN

# tests/ocr/callbacks/test_wandb_image_logging.py
- from ocr.core.utils.geometry_utils import apply_padding_offset_to_polygons, compute_padding_offsets
+ from ocr.domains.detection.utils.geometry import apply_padding_offset_to_polygons, compute_padding_offsets

# tests/ocr/datasets/test_polygon_filtering.py
- from ocr.core.utils.polygon_utils import filter_degenerate_polygons
+ from ocr.domains.detection.utils.polygons import filter_degenerate_polygons

# ... (similar for other 7 files)
```

### Inference Tests (5 files):
```python
# tests/unit/test_image_loader.py
- from ocr.core.inference.engine import InferenceEngine
+ from ocr.pipelines.engine import InferenceEngine
```

### Lightning Module Tests (3 files):
```python
# tests/unit/test_lightning_module.py
- from ocr.core.lightning.ocr_pl import OCRPLModule
+ from ocr.domains.detection.module import DetectionPLModule
+ from ocr.domains.recognition.module import RecognitionPLModule
```

### Experiment Manager Test (1 file):
```python
# tests/test_etk_compass.py
- from etk.compass import CompassPaths
+ from project_compass.src.core import CompassPaths
```

---

## Conclusion

### What Happened:
1. ✅ Domain separation refactor completed (commit 7eef131)
2. ❌ CI worker reverted architecture with shims (commit bd27a23)
3. ⚠️ Nuclear refactor didn't catch OCR scope creep (commit ba614b6)
4. ❌ Critical function `get_datasets_by_cfg()` lost

### What Should Happen:
1. **Immediate**: Verify training still works
2. **If broken**: Restore `get_datasets_by_cfg()` urgently
3. **Strategic**: Delete all shims and fix test imports properly
4. **Long-term**: Add CI check to prevent legacy imports

### Key Insight:
**The CI "fix" was tactically correct (tests pass) but strategically wrong (architecture violated).**

This is a textbook case of why architectural documentation must be accessible during CI fixes.

---

## Next Steps (Your Decision)

Which would you prefer?

**A. Immediate Revert**: Revert bd27a23, fix tests properly (clean slate)
**B. Verify First**: Check if training works, then decide based on `get_datasets_by_cfg()` impact
**C. Surgical Fix**: Restore `__init__.py`, delete `ocr_pl.py`, update critical tests first
**D. Document Only**: Keep current state, create migration plan for v0.4.0 (least disruptive)

I recommend **Option B** (verify training) → then **Option C** (surgical fix) to minimize disruption while restoring architecture.
