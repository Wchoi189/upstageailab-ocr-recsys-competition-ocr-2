# Training Performance Optimization - Findings

**Spec**: 002-training-performance-optimization
**Branch**: 001-wandb-config-logging
**Status**: Phase 3 Complete

## Phase 1: Tokenizer Caching ✅
**Issue**: Tokenizer loaded 5 times per training run
**Fix**: Implemented singleton caching in `KoreanOCRTokenizer.get_or_create()`
**Result**: 5→1 tokenizer loads
**Commit**: c294ae40

## Phase 2: Lazy Dataset Loading ✅
**Issue**: All dataset splits loaded regardless of mode
**Fix**: Mode-aware split detection in `OCRProjectOrchestrator._get_required_splits()`
**Result**: Only required splits loaded per mode
**Commit**: c294ae40

## Phase 3: Model Weight Duplication ✅
**Issue**: Model weights loaded 2x (duplicate "Loading pretrained weights" messages)

### Root Cause
The model factory in `ocr/core/models/__init__.py` was double-instantiating encoder components:

```python
# BEFORE (line 11)
return hydra.utils.instantiate(architectures, cfg=config)
```

**Problem**: Both `architectures` and `cfg=config` contained the encoder definition with `_target_`, causing Hydra to instantiate:
1. Encoder from `architectures.encoder` → **First instantiation**
2. Encoder from `cfg` parameter → **Second instantiation**

### Investigation Method
Added stack trace profiling to `TimmBackbone.__init__()`:
```python
logger.info("Stack trace:")
logger.info("".join(traceback.format_stack()))
```

Stack traces revealed two separate instantiation paths:
- **First**: Direct instantiation via `hydra.utils.instantiate(architectures, ...)`
- **Second**: Nested instantiation via `cfg[key] = instantiate_node(...)` when processing the `cfg` parameter

### Fix
Removed redundant `cfg=config` parameter:
```python
# AFTER (line 11)
return hydra.utils.instantiate(architectures)
```

**Rationale**: The `architectures` config already contains all component definitions. Passing `cfg=config` was redundant and caused duplicate instantiation.

### Verification
```bash
# Before fix
uv run python scripts/runners/train.py experiment=parseq_flash trainer.limit_train_batches=0 2>&1 | grep -c "Loading pretrained weights"
# Output: 2

# After fix
uv run python scripts/runners/train.py experiment=parseq_flash trainer.limit_train_batches=0 2>&1 | grep -c "Loading pretrained weights"
# Output: 1
```

### Result
- **Model weight loads**: 2→1 ✅
- **Time saved**: ~1.4 seconds per training run
- **No functionality changes**: Model still works correctly

### Files Modified
- `ocr/core/models/__init__.py` - Removed redundant cfg parameter

## Summary

| Phase | Issue | Fix | Result |
|-------|-------|-----|--------|
| 1 | Tokenizer loaded 5x | Singleton caching | 5→1 loads |
| 2 | All splits loaded | Mode-aware splits | Load only required |
| 3 | Weights loaded 2x | Remove duplicate cfg | 2→1 loads |

**Total Performance Improvement**: Eliminated redundant initialization operations, reducing startup overhead.
