# Specification: Training Pipeline Performance Optimization

**Feature Branch**: `002-training-performance-optimization`
**Created**: February 15, 2026
**Status**: Planning
**Priority**: P1 (Blocks efficient training iteration)

## Problem Statement

Training pipeline exhibits redundant processing during initialization:
- Tokenizer loaded 5x per training start (~0.5-1s overhead)
- Pretrained model weights loaded 2x from HuggingFace
- Unused datasets instantiated (test/predict in train mode)
- Config serialization repeated unnecessarily

**Impact**: 2-3s wasted on every training start, accumulated debug prints in hot paths.

## Root Cause Analysis

### RC-001: Tokenizer Duplication
**File**: `ocr/pipelines/orchestrator.py:97`, `ocr/data/datasets/`
**Evidence**:
```
[INFO] - Loaded tokenizer: 1023 chars, vocab_size=1027, max_len=25  # x5
```

**Cause**: Separate instantiation for:
1. Vocab injection (orchestrator.py:97)
2. Train dataset config
3. Val dataset config
4. Test dataset config
5. Predict dataset config

Each creates new `KoreanOCRTokenizer` instance from same `charset.json`.

### RC-002: Model Weight Redundancy
**File**: Model architecture instantiation
**Evidence**:
```
Loading pretrained weights from Hugging Face hub (timm/resnet18.a1_in1k)  # x2
```

**Hypothesis**: Encoder created during vocab injection validation or architecture health check, then recreated for actual model.

### RC-003: Eager Dataset Creation
**File**: `ocr/pipelines/orchestrator.py:105`
```python
dataset = get_datasets_by_cfg(self.cfg.data, data_config, self.cfg)
# Creates: train, val, test, predict regardless of mode
```

**Waste**: In `mode=train`, test/predict datasets never used but LMDB connections opened.

### RC-004: Config Serialization Overhead
**File**: `ocr/pipelines/orchestrator.py:200`
```python
yaml_str = OmegaConf.to_yaml(self.cfg, resolve=True)  # 10KB traversal
```

**When**: Only when `log_config=true` (rare)
**Issue**: Could be cached with config hash to skip if unchanged.

## Optimization Plan

### OPT-001: Tokenizer Singleton Cache
**Priority**: P0 (highest impact/effort ratio)
**Estimated Impact**: 0.5-1s per training start
**Risk**: Low

**Implementation**:
```python
# ocr/domains/recognition/data/tokenizer.py
_TOKENIZER_CACHE: Dict[Tuple[str, int], 'KoreanOCRTokenizer'] = {}

@classmethod
def get_or_create(cls, charset_path: str, max_len: int) -> 'KoreanOCRTokenizer':
    key = (charset_path, max_len)
    if key not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[key] = cls(charset_path, max_len)
    return _TOKENIZER_CACHE[key]
```

**Changes Required**:
1. Add `get_or_create()` classmethod to `KoreanOCRTokenizer`
2. Update vocab injection in `orchestrator.py:97`
3. Update dataset configs to use factory method
4. Validate tokenizer reuse doesn't break state isolation

**Validation**:
- Log count should drop from 5 to 1
- Vocab size still correctly injected
- All datasets share same tokenizer instance

### OPT-002: Lazy Dataset Instantiation
**Priority**: P1
**Estimated Impact**: 0.3-0.5s per training start (mode-dependent)
**Risk**: Medium (requires refactor)

**Implementation**:
```python
# ocr/pipelines/orchestrator.py
def _get_datasets_for_mode(self) -> List[str]:
    return {
        "train": ["train", "val"],
        "eval": ["val"],
        "test": ["test"],
        "predict": ["predict"]
    }[self.mode]

def setup_modules(self):
    needed = self._get_datasets_for_mode()
    dataset = get_datasets_by_cfg(
        self.cfg.data,
        data_config,
        self.cfg,
        splits=needed  # NEW parameter
    )
```

**Changes Required**:
1. Add `splits` parameter to `get_datasets_by_cfg()`
2. Conditional dataset creation in factory
3. Update Lightning DataModule to handle missing splits gracefully
4. Test all modes (train/eval/test/predict)

### OPT-003: Investigate Model Weight Duplication
**Priority**: P2
**Estimated Impact**: 0.5-2s per training start (network-dependent)
**Risk**: Low (investigation only)

**Tasks**:
1. Add debug logging to track encoder creation stack traces
2. Check if vocab injection calls model instantiation
3. Profile `setup_modules()` to identify duplicate weight loads
4. Fix if identified, document if unavoidable

### OPT-004: Config Serialization Caching
**Priority**: P3
**Estimated Impact**: Negligible (only affects `log_config=true` case)
**Risk**: Low

**Implementation**: Hash-based cache to skip redundant YAML serialization.

**Defer**: Low impact, optimize after P0-P2 complete.

### OPT-005: Remove Debug Logging
**Priority**: P2
**Estimated Impact**: Clean logs, minimal performance
**Risk**: None

**Files**:
- `ocr/domains/recognition/module.py:114-128` - Remove validation debug prints
- Convert to `logger.debug()` for optional verbose mode

## Success Criteria

**Quantitative**:
- Training start time reduced by ≥1.5s (measured from orchestrator init to first batch)
- Tokenizer load count: 5 → 1
- Model weight load count: 2 → 1
- Unused dataset instantiation: eliminated for train mode

**Qualitative**:
- No regression in training accuracy/loss
- All modes (train/eval/test/predict) functional
- Clean logs (no debug prints in production)

## Testing Strategy

### Unit Tests
```python
# tests/ocr/domains/recognition/test_tokenizer_cache.py
def test_tokenizer_singleton():
    tok1 = KoreanOCRTokenizer.get_or_create("charset.json", 25)
    tok2 = KoreanOCRTokenizer.get_or_create("charset.json", 25)
    assert tok1 is tok2  # Same instance

def test_tokenizer_different_params():
    tok1 = KoreanOCRTokenizer.get_or_create("charset.json", 25)
    tok2 = KoreanOCRTokenizer.get_or_create("charset.json", 30)
    assert tok1 is not tok2  # Different instances
```

### Integration Tests
```bash
# Test all modes with profiling
for mode in train eval test predict; do
    uv run python scripts/runners/train.py \
      experiment=parseq_flash_fast \
      mode=$mode \
      trainer.limit_train_batches=1 \
      trainer.limit_val_batches=1 \
      checkpoint_path=null
done
```

### Performance Baseline
```bash
# Before optimization
time uv run python scripts/runners/train.py \
  experiment=parseq_flash_fast \
  trainer.limit_train_batches=0 \
  checkpoint_path=null

# Record: Orchestrator init → Trainer ready time
```

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Tokenizer state mutation | High | Low | Make tokenizer immutable after init |
| Dataset factory API break | Medium | Medium | Backward-compatible `splits` parameter |
| Mode-specific dataset bugs | Medium | Low | Test all 4 modes in CI |
| Cache invalidation issues | Low | Low | Use immutable cache keys |

## Implementation Order

1. **Phase 1 - Quick Wins** (Day 1)
   - OPT-001: Tokenizer caching
   - OPT-005: Remove debug logging
   - Test: Verify tokenizer count drops to 1

2. **Phase 2 - Lazy Loading** (Day 2)
   - OPT-002: Lazy dataset instantiation
   - Test: All modes functional
   - Measure: Performance improvement

3. **Phase 3 - Investigation** (Day 3)
   - OPT-003: Profile model weight loading
   - Fix if issue found
   - Document findings

4. **Phase 4 - Polish** (Optional)
   - OPT-004: Config caching if needed
   - Final benchmarks

## Rollback Plan

All changes behind feature flag:
```yaml
# configs/runtime/performance.yaml
optimizations:
  tokenizer_caching: true
  lazy_datasets: true
```

If issues detected, disable via config override.

## Related Documentation

- Hydra patterns: `/workspaces/AgentQMS/specs/tier2-framework/patterns.spec.md`
- Configuration: `/workspaces/AgentQMS/specs/tier2-framework/configuration.spec.md`
- Orchestrator: `/workspaces/ocr/pipelines/orchestrator.py`

## Deferred Optimizations

- Flash Attention kernel warm-up (already documented in pulse_staging)
- DataLoader worker optimization (requires dataset profiling)
- Gradient accumulation micro-batching (model-specific)
