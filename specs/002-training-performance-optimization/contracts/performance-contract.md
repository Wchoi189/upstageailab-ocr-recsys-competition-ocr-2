# Performance Contract: Training Pipeline Optimization

**Spec**: `002-training-performance-optimization`
**Contract ID**: `PERF-001`

## Baseline Metrics (Pre-Optimization)

Measured on: RTX 3090, parseq_flash_fast experiment
```
Tokenizer loads: 5
Model weight loads: 2
Startup time (init → ready): ~3.5s
Dataset instantiation: All 4 splits (train/val/test/predict)
```

## Target Metrics (Post-Optimization)

### Hard Requirements (P0)
✅ **Tokenizer loads**: 1 (reduction: 80%)
✅ **Startup time**: ≤2.0s (reduction: ≥1.5s)
✅ **Mode-specific datasets**: Only required splits created

### Stretch Goals (P1)
🎯 **Model weight loads**: 1 (if root cause identified)
🎯 **Memory usage**: 10% reduction in idle GPU memory
🎯 **Log cleanliness**: No debug prints in production logs

## API Stability Contract

### Backward Compatibility Guarantees

1. **Tokenizer API**:
   ```python
   # Old instantiation still works
   tok = KoreanOCRTokenizer(charset_path="...", max_len=25)

   # New factory method (recommended)
   tok = KoreanOCRTokenizer.get_or_create(charset_path="...", max_len=25)
   ```

2. **Dataset Factory**:
   ```python
   # Old call still works (creates all splits)
   datasets = get_datasets_by_cfg(data_cfg, data_config, full_cfg)

   # New parameter (optional)
   datasets = get_datasets_by_cfg(data_cfg, data_config, full_cfg, splits=["train", "val"])
   ```

3. **Orchestrator**:
   - No public API changes
   - Internal optimizations transparent to users

### Breaking Changes (None Expected)
- No config file format changes
- No CLI argument changes
- No model checkpoint format changes

## Rollback Contract

### Feature Flags
```yaml
# configs/runtime/performance.yaml
optimizations:
  tokenizer_caching: true  # Set to false to disable
  lazy_datasets: true      # Set to false to load all splits
  debug_logging: false     # Set to true for verbose output
```

### Rollback Procedure
1. Set `optimizations.tokenizer_caching: false` in config
2. Set `optimizations.lazy_datasets: false` in config
3. Restart training
4. Previous behavior restored (5x tokenizer loads, all datasets)

### Emergency Hotfix
```bash
# Override via CLI
uv run python scripts/runners/train.py \
  experiment=parseq_flash_fast \
  runtime.optimizations.tokenizer_caching=false \
  runtime.optimizations.lazy_datasets=false
```

## Test Coverage Contract

### Unit Tests (Mandatory)
- `test_tokenizer_cache.py`: Singleton behavior
- `test_dataset_factory.py`: Splits parameter handling
- `test_orchestrator_modes.py`: Mode-specific dataset creation

### Integration Tests (Mandatory)
- All 4 modes (train/eval/test/predict) functional
- Checkpoint loading/saving works
- WandB logging unaffected
- Multi-GPU training compatible (if applicable)

### Performance Tests (Mandatory)
```bash
# Automated benchmark script
scripts/benchmark_startup_time.sh

# Regression test (fail if >2.0s startup)
pytest tests/performance/test_startup_regression.py
```

## Acceptance Criteria

### Phase 1 Complete (Tokenizer Caching)
- [ ] Unit tests pass
- [ ] Log shows "Reusing cached tokenizer" 4x per training start
- [ ] Startup time reduced by ≥0.5s
- [ ] No accuracy/loss regression on validation set

### Phase 2 Complete (Lazy Datasets)
- [ ] All modes tested (train/eval/test/predict)
- [ ] Only required datasets instantiated per mode
- [ ] Startup time reduced by additional ≥0.5s
- [ ] DataModule handles missing splits gracefully

### Phase 3 Complete (Investigation)
- [ ] Model weight duplication root cause identified
- [ ] Fix implemented (if feasible) or documented (if unavoidable)
- [ ] Findings documented in `findings.md`

### Final Acceptance
- [ ] Total startup time ≤2.0s (≥1.5s improvement)
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Performance benchmark script passes
- [ ] No debug prints in production logs
- [ ] Documentation updated

## Verification Commands

```bash
# Check tokenizer caching is working
uv run python scripts/runners/train.py experiment=parseq_flash_fast \
  trainer.limit_train_batches=0 checkpoint_path=null 2>&1 | \
  grep -c "Loaded tokenizer"
# Expected output: 1

# Check lazy dataset loading (train mode)
uv run python scripts/runners/train.py experiment=parseq_flash_fast \
  mode=train trainer.limit_train_batches=0 checkpoint_path=null 2>&1 | \
  grep "✓ Datasets created" -A 5
# Expected: Only train and val datasets mentioned

# Check startup time
time uv run python scripts/runners/train.py experiment=parseq_flash_fast \
  trainer.limit_train_batches=0 checkpoint_path=null
# Expected: ≤2.0s from init to ready
```

## Performance SLA (Post-Release)

- **Startup time**: ≤2.5s (2.0s target + 0.5s buffer)
- **Memory overhead**: <50MB additional RAM for caches
- **Regression tolerance**: ±5% variance acceptable

## Non-Functional Requirements

### Observability
- Tokenizer cache hits/misses logged at DEBUG level
- Dataset creation logged per mode
- Startup time metrics logged

### Maintainability
- Cache invalidation documented
- Performance benchmarks in CI
- Rollback procedure tested

### Security
- No new dependencies introduced
- No credential caching
- Thread-safe caching implementation
