# Session Handover: Training Performance Optimization

**Spec**: `002-training-performance-optimization`
**Date**: February 15, 2026
**Session Status**: Implementation Complete, Commit Pending

## Summary

✅ **All optimizations implemented and tested successfully on branch `001-wandb-config-logging`**

**Key Achievements**:
- Tokenizer caching: 5 loads → 1 load (80% reduction)
- Lazy dataset loading: Mode-specific dataset creation working
- All 8 unit tests passing
- All modes (train/eval/test/predict) tested and functional

## What Was Done

### Phase 1: Tokenizer Singleton Cache ✅

**Files Modified**:
1. `/workspaces/ocr/domains/recognition/data/tokenizer.py`
   - Added `_TOKENIZER_CACHE` module-level dict
   - Implemented `get_or_create()` classmethod
   - Cache key: `(resolved_path, max_len)`

2. `/workspaces/ocr/pipelines/strategies/recognition_config.py` (or orchestrator on main)
   - Updated vocab injection to use `KoreanOCRTokenizer.get_or_create()`
   - Removed `hydra.utils.instantiate()` for tokenizer

3. `/workspaces/ocr/data/datasets/__init__.py`
   - Pre-instantiate tokenizer once using `get_or_create()`
   - Pass as override to `hydra.instantiate(dataset_cfg, tokenizer=instance)`
   - Avoids OmegaConf non-primitive type restriction

4. `/workspaces/ocr/domains/recognition/data/lmdb_dataset.py`
   - Added type hint support for dict tokenizer config (not used but kept for flexibility)

**Tests Added**:
- `/workspaces/tests/ocr/domains/recognition/test_tokenizer_cache.py` (8 tests, all passing)

### Phase 2: Lazy Dataset Loading ✅

**Files Modified**:
1. `/workspaces/ocr/data/datasets/__init__.py`
   - Added `splits` parameter (default: `None` for backward compatibility)
   - Only instantiate datasets in requested splits
   - Return `None` for unused splits

2. `/workspaces/ocr/pipelines/orchestrator.py`
   - Added `_get_required_splits()` method
   - Mode mapping:
     - `train` → `["train", "val"]`
     - `eval` → `["val"]`
     - `test` → `["test"]`
     - `predict` → `["predict"]`
   - Pass `splits=required_splits` to dataset factory

3. `/workspaces/ocr/data/lightning_data.py`
   - Handle `None` datasets gracefully
   - Return `None` from dataloader methods when dataset missing

**Scripts Created**:
- `/workspaces/scripts/test_lazy_datasets.sh` - Integration test for all modes
- `/workspaces/scripts/benchmark_startup_time.sh` - Performance benchmark

**Documentation**:
- `/workspaces/specs/002-training-performance-optimization/findings.md` - Detailed results

## Verification

All functionality tested and working:
```bash
# Tokenizer caching verified
uv run python scripts/runners/train.py experiment=parseq_flash_fast trainer.limit_train_batches=0 trainer.enable_checkpointing=false 2>&1 | grep "Loaded tokenizer"
# Output: 1 line (was 5)

# Lazy loading verified
uv run python scripts/runners/train.py experiment=parseq_flash_fast mode=train trainer.limit_train_batches=0 2>&1 | grep "Creating datasets"
# Output: Creating datasets for splits: ['train', 'val']

uv run python scripts/runners/train.py experiment=parseq_flash_fast mode=eval trainer.limit_val_batches=1 2>&1 | grep "Creating datasets"
# Output: Creating datasets for splits: ['val']

uv run python scripts/runners/train.py experiment=parseq_flash_fast mode=test trainer.limit_test_batches=1 2>&1 | grep "Creating datasets"
# Output: Creating datasets for splits: ['test']
```

## Current Situation

**Problem**: Branch management complexity
- All changes implemented and tested on branch `001-wandb-config-logging`
- Attempted to create new branch `002-training-performance-optimization` from main
- Main branch has diverged significantly → merge conflicts
- Need to cleanly apply only performance optimization changes to new branch

**Why This Happened**:
- Started work on wrong branch (001 instead of 002)
- Main branch underwent significant refactoring (file moves, deletions)
- Stash contained changes from multiple features

## Next Steps for New Session

### Option 1: Commit on Current Branch (Recommended)

1. Switch back to `001-wandb-config-logging`:
   ```bash
   git checkout 001-wandb-config-logging
   ```

2. Commit the performance optimization changes:
   ```bash
   git add ocr/domains/recognition/data/tokenizer.py \
           ocr/data/datasets/__init__.py \
           ocr/pipelines/strategies/recognition_config.py \
           ocr/pipelines/orchestrator.py \
           ocr/data/lightning_data.py \
           ocr/domains/recognition/data/lmdb_dataset.py \
           tests/ocr/domains/recognition/test_tokenizer_cache.py \
           scripts/test_lazy_datasets.sh \
           scripts/benchmark_startup_time.sh \
           specs/002-training-performance-optimization/

   git commit -m "feat(perf): implement tokenizer caching and lazy dataset loading

Phase 1: Tokenizer Singleton Cache
- Add module-level cache in KoreanOCRTokenizer
- Implement get_or_create() classmethod
- Update vocab injection to use cached tokenizer
- Pass tokenizer as override to dataset instantiation
- Result: 5 tokenizer loads → 1 (80% reduction)

Phase 2: Lazy Dataset Loading
- Add splits parameter to get_datasets_by_cfg()
- Implement mode-specific dataset creation
- Update orchestrator to pass required splits
- Handle None datasets in Lightning DataModule
- Result: Only create datasets needed for current mode

Testing:
- 8 unit tests for tokenizer caching (all passing)
- Integration tests for all modes (train/eval/test/predict)
- Performance validated (tokenizer + lazy loading working)

Closes: #002-training-performance-optimization

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
   ```

3. Verify commit:
   ```bash
   git log -1 --stat
   git show HEAD
   ```

4. Optionally cherry-pick to new branch:
   ```bash
   git checkout main
   git checkout -b 002-training-performance-optimization
   git cherry-pick <commit-hash>
   ```

### Option 2: Manual File-by-File Reapplication on Clean Branch

1. Stay on clean `002-training-performance-optimization` branch

2. Apply changes file by file (code preserved in findings.md):
   - Copy tokenizer changes from findings.md or rework
   - Apply vocab injection update
   - Apply dataset factory changes
   - Apply orchestrator changes
   - Apply data module changes
   - Copy test files
   - Copy scripts

3. Test everything again

4. Commit

### Option 3: Use Git Worktree (Clean Parallel Workspace)

1. Create worktree for clean work:
   ```bash
   git worktree add ../ocr-perf-opt 002-training-performance-optimization
   cd ../ocr-perf-opt
   ```

2. Apply changes there

3. Test and commit

4. Remove worktree:
   ```bash
   cd /workspaces
   git worktree remove ../ocr-perf-opt
   ```

## Files to Commit

**Core Changes** (must include):
- `ocr/domains/recognition/data/tokenizer.py`
- `ocr/data/datasets/__init__.py`
- `ocr/pipelines/orchestrator.py` (or `strategies/recognition_config.py` if exists)
- `ocr/data/lightning_data.py`
- `ocr/domains/recognition/data/lmdb_dataset.py`

**Tests & Scripts**:
- `tests/ocr/domains/recognition/test_tokenizer_cache.py`
- `scripts/test_lazy_datasets.sh`
- `scripts/benchmark_startup_time.sh`

**Documentation**:
- `specs/002-training-performance-optimization/` (all files)

## Technical Notes for Implementation

### Key Implementation Detail: OmegaConf Workaround

**Problem**: Cannot inject Python objects into OmegaConf configs
```python
# This fails:
cfg.tokenizer = tokenizer_instance  # UnsupportedValueType error
```

**Solution**: Use Hydra's instantiate override parameter
```python
# This works:
tokenizer_instance = KoreanOCRTokenizer.get_or_create(...)
dataset = instantiate(dataset_cfg, tokenizer=tokenizer_instance)
```

The override parameter bypasses OmegaConf serialization and directly injects the tokenizer instance into the dataset constructor.

### Architecture Difference Between Branches

**Main branch structure**:
- Vocab injection in `orchestrator.py:_inject_vocab_size()`
- No `strategies/recognition_config.py`

**001 branch structure** (where work was done):
- Vocab injection may be in `strategies/recognition_config.py`

**When applying to main**: Check where `inject_vocab_size` is located and update accordingly.

## Success Criteria Verification

✅ **Tokenizer load count**: 5 → 1
✅ **Lazy dataset loading**: Only required splits created
✅ **No regressions**: All modes functional
✅ **Backward compatible**: Default behavior unchanged
✅ **Tests added**: 8 unit tests passing
✅ **Clean logs**: Mode-specific dataset creation logged

## Remaining Work (Optional - P2/P3)

- **OPT-003**: Investigate model weight duplication (still seeing 2x weight loading)
- **OPT-005**: Replace debug prints with logger.debug()
- **OPT-004**: Config serialization caching (low priority)

## Questions to Answer in Next Session

None - implementation is complete and verified. Only commit/branch management remains.

## Recommended Action

**Immediate**: Use Option 1 (commit on 001 branch, then cherry-pick to 002 if needed)

This is the quickest path to completion and avoids re-implementing already-tested code.

## Performance Impact Summary

**Before**:
- 5 tokenizer loads
- 4 dataset splits always created
- Estimated overhead: 2-3s

**After**:
- 1 tokenizer load (cached, reused)
- Only required splits created (50-75% reduction depending on mode)
- Measurable improvement in startup time

## Continuation Prompt

```
I'm continuing the training performance optimization work from the previous session.

Status: All optimizations implemented and tested successfully, but need to commit changes.

Current situation:
- On branch: 001-wandb-config-logging (all changes here)
- Need to commit performance optimization work
- Changes verified working (tokenizer caching + lazy dataset loading)

Please review SESSION_HANDOVER.md and commit the changes using Option 1 (commit on current branch).

Files to commit are listed in the handover document.
```

---

**Session End**: Implementation complete, ready for commit in next session.
