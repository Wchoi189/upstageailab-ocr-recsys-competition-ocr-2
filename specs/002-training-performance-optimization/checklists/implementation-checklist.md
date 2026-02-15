# Implementation Checklist: Training Performance Optimization

**Spec**: `002-training-performance-optimization`

## Pre-Implementation

- [ ] Read full spec: `spec.md`
- [ ] Review task breakdown: `tasks.md`
- [ ] Review performance contract: `contracts/performance-contract.md`
- [ ] Establish baseline metrics (run `scripts/benchmark_startup_time.sh`)
- [ ] Create feature branch: `git checkout -b 002-training-performance-optimization`

## Phase 1: Tokenizer Caching

### TASK-001: Implement Tokenizer Singleton
- [ ] Add `_TOKENIZER_CACHE` dict to module level
- [ ] Implement `get_or_create()` classmethod
- [ ] Add debug logging for cache hits/misses
- [ ] Test manually with Python REPL
- [ ] Verify cache key uses immutable tuple

### TASK-002: Update Vocab Injection
- [ ] Locate vocab injection in `recognition_config.py`
- [ ] Replace direct instantiation with `get_or_create()`
- [ ] Test: `uv run python scripts/runners/train.py experiment=parseq_flash_fast trainer.limit_train_batches=0`
- [ ] Verify vocab size still injected correctly
- [ ] Check log: "Reusing cached tokenizer" appears

### TASK-003: Update Dataset Configs
- [ ] Search for tokenizer instantiation: `grep -r "KoreanOCRTokenizer" configs/`
- [ ] Identify dataset factory instantiation points
- [ ] Update to use `get_or_create()` or wrapper
- [ ] Test dataset creation in isolation
- [ ] Verify tokenizer shared across datasets

### TASK-004: Add Unit Tests
- [ ] Create `tests/ocr/domains/recognition/test_tokenizer_cache.py`
- [ ] Test: Same params → same instance
- [ ] Test: Different max_len → different instance
- [ ] Test: Different charset → different instance
- [ ] Run: `pytest tests/ocr/domains/recognition/test_tokenizer_cache.py -v`
- [ ] All tests pass

### Phase 1 Validation
- [ ] Run training: `uv run python scripts/runners/train.py experiment=parseq_flash_fast`
- [ ] Count tokenizer loads: Should be 1 (grep "Loaded tokenizer")
- [ ] Measure startup time: Record improvement
- [ ] Run full test suite: No regressions
- [ ] Commit: `git commit -m "feat: implement tokenizer singleton caching"`

## Phase 2: Lazy Dataset Loading

### TASK-005: Add `splits` Parameter
- [ ] Locate `get_datasets_by_cfg()` function
- [ ] Add `splits: Optional[List[str]] = None` parameter
- [ ] Implement conditional dataset creation
- [ ] Default behavior: `splits=None` creates all (backward compatible)
- [ ] Add docstring explaining parameter

### TASK-006: Update Orchestrator
- [ ] Add `_get_required_splits()` helper method
- [ ] Map modes to required splits
- [ ] Update `setup_modules()` to pass splits
- [ ] Test: Run with `mode=train`, verify only train/val created
- [ ] Test: Run with `mode=eval`, verify only val created

### TASK-007: Update Lightning DataModule
- [ ] Handle missing splits in `train_dataloader()`
- [ ] Handle missing splits in `val_dataloader()`
- [ ] Handle missing splits in `test_dataloader()`
- [ ] Handle missing splits in `predict_dataloader()`
- [ ] Raise clear error if required split missing
- [ ] Optional: Return None for optional splits

### TASK-008: Test All Modes
- [ ] Create `scripts/test_lazy_datasets.sh`
- [ ] Test `mode=train` → train/val datasets only
- [ ] Test `mode=eval` → val dataset only
- [ ] Test `mode=test` → test dataset only
- [ ] Test `mode=predict` → predict dataset only
- [ ] Check logs: Unused datasets not created
- [ ] Verify no crashes or missing data errors

### Phase 2 Validation
- [ ] All modes functional
- [ ] Startup time improved further
- [ ] Memory usage reduced (check with `nvidia-smi`)
- [ ] Run: `bash scripts/test_lazy_datasets.sh`
- [ ] Commit: `git commit -m "feat: implement lazy dataset instantiation"`

## Phase 3: Model Weight Investigation

### TASK-009: Add Profiling
- [ ] Add stack trace logging to model creation
- [ ] Enable: `global.debug=true`
- [ ] Run training with debug enabled
- [ ] Capture model creation logs
- [ ] Identify duplicate creation points

### TASK-010: Check Vocab Injection
- [ ] Review `inject_vocab_size()` implementation
- [ ] Check if it instantiates model
- [ ] If yes: Refactor to inspect config only
- [ ] Test vocab injection still works
- [ ] Verify model not created during injection

### TASK-011: Document Findings
- [ ] Create `findings.md` in spec directory
- [ ] Document root cause of 2x loading
- [ ] Document fix (if implemented)
- [ ] Document reason (if unavoidable)
- [ ] Add recommendations for future

### Phase 3 Validation
- [ ] Model weight load count reduced (if fixable)
- [ ] Or: Documented why duplication unavoidable
- [ ] Startup time improvement measured
- [ ] Commit: `git commit -m "perf: optimize model weight loading"`

## Phase 4: Cleanup

### TASK-012: Remove Debug Prints
- [ ] Locate debug prints in `module.py`
- [ ] Replace `print()` with `logger.debug()`
- [ ] Add debug level check for expensive operations
- [ ] Test: No prints in production logs
- [ ] Test: Prints visible with `global.debug=true`
- [ ] Commit: `git commit -m "refactor: replace debug prints with logger"`

## Phase 5: Benchmarking

### TASK-013: Baseline Measurement
- [ ] Run `scripts/benchmark_startup_time.sh` (if created)
- [ ] Record baseline startup time
- [ ] Record tokenizer load count
- [ ] Record dataset creation count
- [ ] Document in `findings.md`

### TASK-014: Post-Optimization Measurement
- [ ] Run same benchmark script
- [ ] Compare results:
  - Startup time: ≥1.5s improvement ✓
  - Tokenizer loads: 5 → 1 ✓
  - Model weights: 2 → 1 ✓ (if fixed)
  - Datasets: Only required splits ✓
- [ ] Document results in `findings.md`
- [ ] Update performance contract with actual metrics

## Final Validation

### Integration Testing
- [ ] Run full training for 1 epoch: `trainer.max_epochs=1`
- [ ] Verify metrics logged correctly
- [ ] Verify checkpoints saved
- [ ] Verify WandB logging works
- [ ] Verify all callbacks execute
- [ ] No errors or warnings in logs

### Regression Testing
- [ ] Run existing test suite: `pytest tests/`
- [ ] All tests pass
- [ ] No new warnings
- [ ] No accuracy degradation on validation set
- [ ] Training curves identical to baseline

### Performance Validation
- [ ] Startup time ≤2.0s (target met)
- [ ] Tokenizer load count = 1
- [ ] Only required datasets created per mode
- [ ] Memory usage not increased
- [ ] No performance regression in training loop

### Documentation
- [ ] Update `specs/002-training-performance-optimization/findings.md`
- [ ] Update `specs/002-training-performance-optimization/contracts/performance-contract.md`
- [ ] Update relevant AgentQMS specs if patterns discovered
- [ ] Add docstrings to new functions
- [ ] Update CHANGELOG if present

## Pre-Merge

- [ ] Rebase on latest main: `git rebase main`
- [ ] Resolve conflicts (if any)
- [ ] Run full test suite on rebased branch
- [ ] Review all changed files for debug code
- [ ] Remove any temporary logging
- [ ] Ensure feature flags documented
- [ ] Verify rollback procedure works

## Post-Merge

- [ ] Update baseline metrics in CI
- [ ] Add performance regression tests to CI
- [ ] Monitor startup time in production
- [ ] Watch for user-reported issues
- [ ] Update spec status to "Completed"

## Rollback (If Needed)

- [ ] Set `runtime.optimizations.tokenizer_caching: false`
- [ ] Set `runtime.optimizations.lazy_datasets: false`
- [ ] Test rollback configuration
- [ ] Verify previous behavior restored
- [ ] Document rollback decision
