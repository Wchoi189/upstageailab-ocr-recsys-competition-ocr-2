# 2122 Phase 3 Handover: Model Weight Investigation

**Branch**: `001-wandb-config-logging`
**Commit**: `c294ae40` (Phase 1 & 2 complete)
**Next**: Investigate model weight duplication

## Status

✅ Phase 1: Tokenizer caching (5→1 loads)
✅ Phase 2: Lazy dataset loading (mode-specific)
⏭️ Phase 3: Model weight investigation

## Objective

Eliminate duplicate model weight loading (currently 2x):
```
Loading pretrained weights from Hugging Face hub (timm/resnet18.a1_in1k)  # x2
```

## Tasks

### TASK-009: Add Model Creation Profiling
**File**: `ocr/pipelines/orchestrator.py:100`

```python
import traceback

def setup_modules(self):
    logger.info("🏗️ Building model and datasets...")

    # Track model creation
    logger.debug("=" * 60)
    logger.debug("Model creation stack trace:")
    logger.debug("".join(traceback.format_stack()))
    logger.debug("=" * 60)

    model = get_model_by_cfg(self.cfg.model)
```

**Run**:
```bash
uv run python scripts/runners/train.py experiment=parseq_flash_fast \
  global.debug=true trainer.limit_train_batches=0 2>&1 | grep -A30 "Model creation"
```

**Look for**: Duplicate stack traces

### TASK-010: Check Vocab Injection
**File**: `ocr/pipelines/strategies/recognition_config.py:inject_vocab_size()`

**Question**: Does vocab injection instantiate model for validation?

**Check**:
```python
# Current: Only instantiates tokenizer
tokenizer = KoreanOCRTokenizer.get_or_create(...)
# Does NOT create model ✓
```

**If model created here**: Refactor to inspect config without instantiation

### TASK-011: Profile Weight Loading Points

**Add logging to model factory**:
```python
# ocr/core/models/__init__.py or factory
def get_model_by_cfg(cfg):
    logger.info(f"📦 Loading model weights...")
    import traceback
    logger.debug("".join(traceback.format_stack()))
```

**Run & compare**: Count "Loading pretrained weights" messages

## Hypothesis

**Likely**: Model weights loaded 2x due to:
1. First load during model instantiation
2. Second load during checkpoint restoration (if enabled)

**Check**: Run with `checkpoint_path=null` and count loads

## Quick Test

```bash
# Count weight loading messages
uv run python scripts/runners/train.py experiment=parseq_flash_fast \
  trainer.limit_train_batches=0 checkpoint_path=null 2>&1 | \
  grep -c "Loading pretrained weights"

# Expected: 1 (if duplication eliminated)
# Current: 2
```

## Resolution Paths

**If duplicate found**:
1. Cache model weights (similar to tokenizer)
2. Skip redundant load in second location
3. Use timm's built-in caching

**If unavoidable**:
- Document in findings.md
- Explain why (e.g., timm caching, checkpoint restore)

## Completion Criteria

- [ ] Identified source of 2x loading
- [ ] Implemented fix OR documented reason
- [ ] Verified: Weight load count ≤ 1
- [ ] Commit: "perf: eliminate model weight duplication"

## Files to Modify

- `ocr/pipelines/orchestrator.py` (add profiling)
- `ocr/core/models/__init__.py` (if caching needed)
- `specs/002-training-performance-optimization/findings.md` (update)

## Time Estimate

1-2 hours (investigation + fix)

## Continuation Command

```
Continue Phase 3: Model weight investigation for spec 002-training-performance-optimization.

Status: Phases 1 & 2 committed (c294ae40)
Branch: 001-wandb-config-logging

Task: Investigate why model weights load 2x (currently see 2x "Loading pretrained weights" messages)

Start with TASK-009 in 2122-phase3-handover.md
```
