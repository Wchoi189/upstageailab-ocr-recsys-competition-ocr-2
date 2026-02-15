# Findings: Training Performance Optimization

**Spec**: `002-training-performance-optimization`
**Date**: February 15, 2026
**Status**: Phase 1 & 2 Complete

## Summary

Successfully implemented tokenizer caching and lazy dataset loading optimizations, achieving the target performance improvements.

## Optimizations Implemented

### ✅ Phase 1: Tokenizer Singleton Cache (COMPLETED)

**Changes**:
1. Added module-level cache `_TOKENIZER_CACHE` in [tokenizer.py](../../ocr/domains/recognition/data/tokenizer.py)
2. Implemented `KoreanOCRTokenizer.get_or_create()` classmethod with cache key `(resolved_path, max_len)`
3. Updated vocab injection in [recognition_config.py](../../ocr/pipelines/strategies/recognition_config.py) to use `get_or_create()`
4. Modified dataset factory in [datasets/__init__.py](../../ocr/data/datasets/__init__.py) to pre-instantiate tokenizer and pass as override to Hydra's `instantiate()`
5. Added comprehensive unit tests in [test_tokenizer_cache.py](../../tests/ocr/domains/recognition/test_tokenizer_cache.py)

**Results**:
- ✅ Tokenizer load count: **5 → 1** (80% reduction)
- ✅ All 8 unit tests passing
- ✅ Backward compatible (no breaking changes)

**Technical Notes**:
- Used `Path.resolve()` to ensure cache keys are consistent across relative/absolute paths
- Leveraged Hydra's `instantiate(config, **overrides)` to bypass OmegaConf's restriction on non-primitive types
- Cache is module-level and persists for process lifetime

### ✅ Phase 2: Lazy Dataset Loading (COMPLETED)

**Changes**:
1. Added `splits` parameter to `get_datasets_by_cfg()` function (default: `None` creates all - backward compatible)
2. Implemented `_get_required_splits()` in [orchestrator.py](../../ocr/pipelines/orchestrator.py) to map modes to required datasets:
   - `train` → `["train", "val"]`
   - `eval` → `["val"]`
   - `test` → `["test"]`
   - `predict` → `["predict"]`
3. Updated Lightning DataModule in [lightning_data.py](../../ocr/data/lightning_data.py) to return `None` for missing dataloaders
4. Created test script [test_lazy_datasets.sh](../../scripts/test_lazy_datasets.sh)

**Results**:
- ✅ **mode=train**: Creates only train + val (skips test + predict)
- ✅ **mode=eval**: Creates only val (skips train + test + predict)
- ✅ **mode=test**: Creates only test (skips train + val + predict)
- ✅ All modes tested and functional

**Technical Notes**:
- Datasets set to `None` in dict when not needed
- Lightning DataModule checks for `None` before creating dataloaders
- LMDB connections only opened for required splits (saves memory and init time)

## Performance Impact

### Before Optimization
- Tokenizer loaded **5 times** (vocab injection + 4 dataset splits)
- All 4 dataset splits created regardless of mode
- Estimated startup overhead: **2-3 seconds**

### After Optimization
- Tokenizer loaded **1 time** (cached and reused)
- Only required dataset splits created per mode
- Observed improvements:
  - **mode=train**: 2 datasets instead of 4 (50% reduction)
  - **mode=eval**: 1 dataset instead of 4 (75% reduction)
  - **mode=test**: 1 dataset instead of 4 (75% reduction)

### Measured Startup Time
- Full training initialization (limit_train_batches=0): **~25 seconds**
  - Note: Includes Python startup, Hydra, model weight loading, WandB init
  - Pure dataset/tokenizer overhead significantly reduced

## Code Quality

### Tests Added
- [test_tokenizer_cache.py](../../tests/ocr/domains/recognition/test_tokenizer_cache.py): 8 comprehensive unit tests
  - Same params → same instance
  - Different params → different instances
  - Path resolution (relative/absolute)
  - Cache key structure
  - Multiple retrievals

### Scripts Added
- [test_lazy_datasets.sh](../../scripts/test_lazy_datasets.sh): Integration test for all modes
- [benchmark_startup_time.sh](../../scripts/benchmark_startup_time.sh): Performance benchmark script

### Backward Compatibility
- ✅ Default behavior unchanged (splits=None creates all datasets)
- ✅ Existing code continues to work without modifications
- ✅ Can still instantiate tokenizer directly with `KoreanOCRTokenizer(...)` if needed

## Issues Encountered & Solutions

### Issue 1: OmegaConf Non-Primitive Type Restriction
**Problem**: Cannot inject Python object (tokenizer instance) into OmegaConf DictConfig
**Error**: `UnsupportedValueType: Value 'KoreanOCRTokenizer' is not a supported primitive type`

**Solution**: Use Hydra's `instantiate(config, **overrides)` to pass tokenizer as kwarg override, bypassing config serialization.

```python
# This works:
instantiate(dataset_cfg, tokenizer=tokenizer_instance)

# This fails:
cfg.tokenizer = tokenizer_instance  # OmegaConf error
instantiate(cfg)
```

### Issue 2: Missing Required Parameter After Config Modification
**Problem**: Removing tokenizer from config caused instantiation to fail with missing required argument

**Solution**: Pass tokenizer as override parameter instead of removing from config

## Remaining Optimizations (Deferred)

### OPT-003: Model Weight Loading Investigation (P2)
**Status**: Not started
**Reason**: Requires profiling to confirm if duplication exists
**Evidence**: Still seeing 2x "Loading pretrained weights" messages
**Next Steps**: Add stack trace logging to identify duplicate creation points

### OPT-004: Config Serialization Caching (P3)
**Status**: Not started
**Reason**: Low impact (only affects `log_config=true` case)
**Next Steps**: Implement if config logging becomes common

### OPT-005: Debug Logging Cleanup (P2)
**Status**: Not started
**Location**: `ocr/domains/recognition/module.py:114-128`
**Action**: Replace `print()` with `logger.debug()`

## Recommendations

1. **Monitor Production**: Track startup time in production to ensure improvements persist
2. **Extend Pattern**: Apply tokenizer caching pattern to other domains if applicable
3. **Profile Model Loading**: Investigate if model weight duplication can be eliminated
4. **Add Metrics**: Consider adding startup time metrics to CI/CD pipeline

## Success Criteria Met

✅ **Tokenizer load count**: 5 → 1
✅ **Lazy dataset loading**: Only required splits created
✅ **No regressions**: All modes functional
✅ **Backward compatible**: Default behavior unchanged
✅ **Tests added**: 8 unit tests + integration tests
✅ **Clean logs**: Mode-specific dataset creation logged

## Files Changed

**Core Changes**:
- `ocr/domains/recognition/data/tokenizer.py` - Added caching
- `ocr/pipelines/strategies/recognition_config.py` - Use cached tokenizer
- `ocr/data/datasets/__init__.py` - Lazy loading + tokenizer injection
- `ocr/pipelines/orchestrator.py` - Mode-specific splits
- `ocr/data/lightning_data.py` - Handle missing dataloaders
- `ocr/domains/recognition/data/lmdb_dataset.py` - Support dict tokenizer config

**Tests & Scripts**:
- `tests/ocr/domains/recognition/test_tokenizer_cache.py` - New
- `scripts/test_lazy_datasets.sh` - New
- `scripts/benchmark_startup_time.sh` - New

**Documentation**:
- `specs/002-training-performance-optimization/findings.md` - This file

## Next Steps

1. **Commit changes**: Create feature branch commit with all optimizations
2. **Profile model loading**: Investigate OPT-003 (model weight duplication)
3. **Cleanup debug prints**: Implement OPT-005
4. **Merge to main**: After validation and review
5. **Monitor metrics**: Track startup time in production

## Conclusion

Phase 1 (Tokenizer Caching) and Phase 2 (Lazy Dataset Loading) successfully implemented and tested. The optimizations achieve the target performance improvements while maintaining backward compatibility and code quality. Ready for commit and merge.
