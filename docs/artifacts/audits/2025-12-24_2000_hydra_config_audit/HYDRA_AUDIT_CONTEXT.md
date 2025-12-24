# Hydra Configuration Audit - Context & Previous Work

## Previous Work Summary

### Legacy Cleanup & Config Consolidation (2025-11-11) - COMPLETED ✅

**Plan**: `archive/archive_docs/docs/completed_plans/2025-11/2025-11-11_1439_implementation_plan_legacy-cleanup-config-consolidation.md`

**Phase 3 Findings**:
- ✅ Checked for duplicate config files
- ✅ Found files with same names but different purposes (e.g., `default.yaml` in different directories)
- ✅ Verified no true duplicates using md5sum
- ✅ Example: `configs/model/optimizer.yaml` vs `configs/model/optimizers/adam.yaml` - similar but serve different purposes (simplified vs complete)
- ✅ **Result**: No configs were removed - all configs serve different purposes

**Key Insight**: Files with same names in different directories are **not duplicates** - they serve different purposes in different contexts.

## Current Audit Focus

### What This Audit Should Do

1. **Architecture Classification** (NOT duplicate removal)
   - Identify which configs use new architecture (`configs/_base/` foundation)
   - Identify which configs use legacy architecture (old system)
   - Identify hybrid configs (mixed references)

2. **Override Pattern Analysis**
   - Document which configs require `+` prefix
   - Document which configs don't require `+`
   - Create override pattern guide

3. **Legacy Containerization**
   - Move legacy configs to `__LEGACY__/` folder
   - Maintain compatibility
   - Preserve configuration options

### What This Audit Should NOT Do

- ❌ Look for duplicate files (already done in 2025-11-11 plan)
- ❌ Remove configs just because names are similar (they serve different purposes)
- ❌ Duplicate previous analysis

## Architecture Context

### New Architecture (Current Default)
- Foundation: `configs/_base/` (core, data, logging, model, preprocessing, trainer)
- Entry points: `train.yaml`, `test.yaml`, `predict.yaml`
- Uses Hydra defaults composition

### Legacy Architecture
- Standalone configs without `_base/` foundation
- May have different structure
- May use different override patterns

### Hybrid Configs
- Mix of new and legacy patterns
- May reference both systems
- Need migration to new architecture

## Override Pattern Context

### Known Issues (from test suite)
- `+logger=wandb` fails (logger in defaults - use `logger=wandb`)
- `+data=canonical` fails (data in defaults - use `data=canonical`)
- `+ablation=model_comparison` works (ablation NOT in defaults)

### Rule
- **In `base.yaml` defaults** → Use without `+` (e.g., `logger=wandb`)
- **NOT in defaults** → Use with `+` (e.g., `+ablation=model_comparison`)

## Migration Strategy

### Option 1: `__LEGACY__/` Folder (Recommended)
- Move legacy configs to `configs/__LEGACY__/`
- Maintain compatibility (Hydra can still find them)
- Clear separation of old vs new
- Preserves configuration history

### Option 2: Archive
- Move to `archive/archive_docs/` (like previous work)
- Less accessible but preserves history

### Option 3: Remove
- Only if truly unused and no references
- **Risky** - may lose configuration options

## Key Questions for Audit

1. Which configs use `configs/_base/` foundation? (New architecture)
2. Which configs don't use `_base/`? (Legacy architecture)
3. Which configs are in `base.yaml` defaults? (Override without `+`)
4. Which configs are NOT in defaults? (Override with `+`)
5. Can legacy configs be moved to `__LEGACY__/` without breaking?
6. What would be lost if legacy configs are removed?

## References

- Previous plan: `archive/archive_docs/docs/completed_plans/2025-11/2025-11-11_1439_implementation_plan_legacy-cleanup-config-consolidation.md`
- Test suite: `tests/unit/test_hydra_overrides.py`

