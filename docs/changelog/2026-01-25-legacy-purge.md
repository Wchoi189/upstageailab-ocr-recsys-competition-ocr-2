# Legacy Purge - 2026-01-25

## Overview

Comprehensive removal of legacy support patterns, deprecation shims, and compatibility layers to enforce V5.0 "Domains First" architecture.

## Critical Changes

### 🔴 Optimizer Configuration (BREAKING)

**Removed:**
- Alternative config path: `config.model.optimizer` ❌
- Fallback to model methods: `model.get_optimizers()` ❌
- Exception handler with Adam fallback ❌
- Manual AdamW/Adam instantiation logic ❌

**New Standard:**
- **ONLY** `config.train.optimizer` with Hydra `_target_` ✅
- Fail-fast with clear error messages ✅
- No fallbacks, no silent failures ✅

**Migration:**
```yaml
# Old (NO LONGER WORKS)
model:
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001

# New (V5 Standard)
# Place in configs/train/optimizer/adam.yaml with:
# @package train.optimizer

# Or use defaults in domain config:
defaults:
  - /train/optimizer: adam
```

**Files Changed:**
- [ocr/core/lightning/base.py](ocr/core/lightning/base.py#L96-L116) - configure_optimizers() simplified
- [configs/train/optimizer/adam.yaml](configs/train/optimizer/adam.yaml) - @package updated
- [configs/train/optimizer/adamw.yaml](configs/train/optimizer/adamw.yaml) - @package updated

### 🔴 Model Architecture (BREAKING)

**Removed:**
- `OCRModel._get_optimizers_impl()` method (46 lines) ❌
- Config path detection logic in models ❌
- Model-level optimizer instantiation ❌

**New Standard:**
- Models are optimizer-agnostic ✅
- Lightning modules handle ALL optimizer config ✅
- Clear separation of concerns ✅

**Files Changed:**
- [ocr/core/models/architecture.py](ocr/core/models/architecture.py#L154-L200) - Method deleted

### 🔴 KIE Domain (ARCHIVED)

**Action:** Entire domain archived to `archive/kie_domain_2026_01_25/`

**Reason:**
- Not actively used in experiments
- Required ~300 lines of migration code
- Used legacy optimizer patterns
- Violated V5 architecture

**Archived Components:**
- `ocr/domains/kie/` (all files)
- `configs/domain/kie.yaml`
- `tests/unit/test_receipt_extraction.py`

**Restoration:** See [archive/kie_domain_2026_01_25/ARCHIVE_README.md](archive/kie_domain_2026_01_25/ARCHIVE_README.md)

### 🟡 Checkpoint Loading (SIMPLIFIED)

**Removed:**
- 4-level fallback chain ❌
- `remove_prefix` parameter ❌
- "model." prefix auto-removal ❌
- `strict=False` final fallback ❌
- Recursion protection (no longer needed) ❌

**New Standard:**
- 2-level fallback ONLY:
  1. Direct load (strict=True) ✅
  2. torch.compile prefix handling ✅
- Fail-fast for incompatible checkpoints ✅
- Clear migration path in error message ✅

**Files Changed:**
- [ocr/core/lightning/utils/model_utils.py](ocr/core/lightning/utils/model_utils.py#L9-L47) - Simplified function

### 🟡 Detection Inference (STRICTER)

**Changed:**
- `model.load_state_dict(filtered_state, strict=False)` ❌
- `model.load_state_dict(filtered_state, strict=True)` ✅

**Added:**
- Error handling for missing keys
- Warning for unexpected keys
- Helpful migration guidance

**Files Changed:**
- [ocr/domains/detection/inference/model_loader.py](ocr/domains/detection/inference/model_loader.py#L93)

### 🟢 Code Cleanup

**Removed:**
- Redundant `load_state_dict()` override in OCRPLModule (3 lines) ❌

**Files Changed:**
- [ocr/core/lightning/base.py](ocr/core/lightning/base.py#L96-L98) - Deleted

## Impact Assessment

### What Breaks

**1. Old Experiments Using `config.model.optimizer`**
```bash
# Error will show:
# ValueError: V5 Hydra config missing: config.train.optimizer is required.
# Legacy model.get_optimizers() is no longer supported.
# See configs/train/optimizer/adam.yaml for template.
```

**Fix:** Update experiment config defaults:
```yaml
defaults:
  - /train/optimizer: adam  # or adamw
```

**2. Models with `get_optimizers()` or `_get_optimizers_impl()`**

Models should NOT implement these methods. If you have custom models with these methods, delete them.

**3. Legacy Checkpoint Formats**

Checkpoints with "model." prefix or other incompatible formats will now fail with:
```
RuntimeError: Checkpoint incompatible with current model architecture.
For legacy checkpoints, use: scripts/checkpoints/convert_legacy_checkpoints.py
```

### What Still Works

- ✅ Detection pipeline (`experiment=det_resnet50_v1`)
- ✅ Recognition pipeline (`experiment=rec_baseline_v1`)
- ✅ torch.compile workflows (prefix handling preserved)
- ✅ All experiments using `config.train.optimizer`

## Statistics

**Code Removed:**
- Optimizer fallback chains: 75 lines
- Model optimizer method: 46 lines
- load_state_dict shim: 3 lines
- Checkpoint fallback levels: ~50 lines
- KIE domain: 300+ lines
- **Total:** ~474 lines of legacy code eliminated

**Complexity Reduction:**
- Optimizer config paths: 2 → 1 (-50%)
- Checkpoint fallback levels: 4 → 2 (-50%)
- Error handling: Silent → Explicit (+100% clarity)

## Next Steps (Deferred to Session 2)

### Low Priority Cleanup

1. **Schema Deprecation Fields** - Remove `schema_version` from AgentQMS schemas
2. **Pillow Compatibility** - Remove Pillow<9 fallback in image_processor.py

These are low-impact and can be done in a future session.

## Prevention Measures

**Pre-commit Hooks Recommended:**
- Detect `except Exception: pass` patterns
- Detect `model.get_optimizers()` calls
- Enforce `config.train.optimizer` location
- Prevent `strict=False` without explicit justification

**Code Review Checklist:**
- [ ] No fallback chains > 2 levels
- [ ] No silent error handling
- [ ] Clear error messages with migration guidance
- [ ] Models are optimizer-agnostic

## References

- **Original Audit:** [legacy-purge-audit-2026-01-25.md](docs/artifacts/audits/legacy-purge-audit-2026-01-25.md)
- **KIE Archive:** [archive/kie_domain_2026_01_25/](archive/kie_domain_2026_01_25/)
- **Hydra Debugging Session:** [__DEBUG__/2026-01-22_hydra_configs_legacy_imports/](DEBUG__/2026-01-22_hydra_configs_legacy_imports/)
- **V5 Architecture:** [configs/README.md](configs/README.md)

---

**Date:** 2026-01-25  
**Type:** Breaking Change - Legacy Purge  
**Status:** Complete  
**Impact:** High (architecture cleanup)  
**Risk:** Low (clear error messages, active pipelines verified)
