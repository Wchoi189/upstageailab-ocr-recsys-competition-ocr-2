# Task Breakdown: Training Performance Optimization

**Spec**: `002-training-performance-optimization`
**Estimated Total**: 2-3 days

## Phase 1: Tokenizer Caching (P0)

### TASK-001: Implement Tokenizer Singleton
**File**: `ocr/domains/recognition/data/tokenizer.py`
**Priority**: P0
**Estimated**: 1h

**Changes**:
```python
# Add module-level cache
_TOKENIZER_CACHE: Dict[Tuple[str, int], 'KoreanOCRTokenizer'] = {}

class KoreanOCRTokenizer:
    @classmethod
    def get_or_create(cls, charset_path: str, max_len: int) -> 'KoreanOCRTokenizer':
        """Get cached tokenizer or create new if not exists."""
        key = (charset_path, max_len)
        if key not in _TOKENIZER_CACHE:
            logger.debug(f"Creating new tokenizer: {charset_path}, max_len={max_len}")
            _TOKENIZER_CACHE[key] = cls(charset_path, max_len)
        else:
            logger.debug(f"Reusing cached tokenizer: {charset_path}, max_len={max_len}")
        return _TOKENIZER_CACHE[key]
```

**Validation**:
- Run training, verify "Reusing cached tokenizer" appears 4x
- Verify "Creating new tokenizer" appears 1x only

---

### TASK-002: Update Vocab Injection
**File**: `ocr/pipelines/strategies/recognition_config.py`
**Priority**: P0
**Estimated**: 30m

**Before**:
```python
tokenizer = KoreanOCRTokenizer(charset_path=..., max_len=...)
```

**After**:
```python
tokenizer = KoreanOCRTokenizer.get_or_create(charset_path=..., max_len=...)
```

**Test**: Vocab injection still works, tokenizer cached.

---

### TASK-003: Update Dataset Configs
**Files**:
- `configs/data/datasets/recognition.yaml`
- Any other dataset configs with `tokenizer` field
**Priority**: P0
**Estimated**: 30m

**Strategy**: Dataset factory should use `get_or_create()` when instantiating tokenizers.

**Check**:
```bash
grep -r "KoreanOCRTokenizer(" configs/
grep -r "_target_: ocr.domains.recognition.data.tokenizer.KoreanOCRTokenizer" configs/
```

**If using Hydra instantiate**: May need wrapper factory or update dataset `__init__` to call `get_or_create()`.

---

### TASK-004: Add Unit Tests
**File**: `tests/ocr/domains/recognition/test_tokenizer_cache.py`
**Priority**: P0
**Estimated**: 30m

```python
def test_tokenizer_singleton_same_params():
    tok1 = KoreanOCRTokenizer.get_or_create("/path/charset.json", 25)
    tok2 = KoreanOCRTokenizer.get_or_create("/path/charset.json", 25)
    assert tok1 is tok2

def test_tokenizer_different_params():
    tok1 = KoreanOCRTokenizer.get_or_create("/path/charset.json", 25)
    tok2 = KoreanOCRTokenizer.get_or_create("/path/charset.json", 30)
    assert tok1 is not tok2

def test_tokenizer_different_files():
    tok1 = KoreanOCRTokenizer.get_or_create("/path/charset1.json", 25)
    tok2 = KoreanOCRTokenizer.get_or_create("/path/charset2.json", 25)
    assert tok1 is not tok2
```

---

## Phase 2: Lazy Dataset Loading (P1)

### TASK-005: Add `splits` Parameter to Dataset Factory
**File**: `ocr/data/datasets/__init__.py` (or wherever `get_datasets_by_cfg` lives)
**Priority**: P1
**Estimated**: 1h

**Signature Change**:
```python
def get_datasets_by_cfg(
    data_cfg: DictConfig,
    data_config: Optional[DictConfig],
    full_cfg: DictConfig,
    splits: Optional[List[str]] = None  # NEW
) -> Dict[str, Dataset]:
    """
    Args:
        splits: List of dataset splits to create. If None, creates all.
                Valid: ["train", "val", "test", "predict"]
    """
    if splits is None:
        splits = ["train", "val", "test", "predict"]  # Default: all

    datasets = {}
    for split in splits:
        if f"{split}_dataset" in data_cfg:
            datasets[split] = instantiate(data_cfg[f"{split}_dataset"])
    return datasets
```

**Backward Compatibility**: Default `splits=None` creates all datasets (current behavior).

---

### TASK-006: Update Orchestrator to Use Mode-Specific Datasets
**File**: `ocr/pipelines/orchestrator.py:105`
**Priority**: P1
**Estimated**: 30m

**Add Helper**:
```python
def _get_required_splits(self) -> List[str]:
    """Return dataset splits needed for current mode."""
    return {
        "train": ["train", "val"],
        "eval": ["val"],
        "test": ["test"],
        "predict": ["predict"]
    }[self.mode]
```

**Update `setup_modules()`**:
```python
dataset = get_datasets_by_cfg(
    self.cfg.data,
    data_config,
    self.cfg,
    splits=self._get_required_splits()  # NEW
)
```

---

### TASK-007: Update Lightning DataModule
**File**: `ocr/data/lightning_data.py`
**Priority**: P1
**Estimated**: 30m

**Handle Missing Splits**:
```python
def train_dataloader(self):
    if "train" not in self.dataset:
        raise ValueError("Train dataset not created for this mode")
    return DataLoader(self.dataset["train"], ...)

def val_dataloader(self):
    if "val" not in self.dataset:
        return None  # Graceful: no validation
    return DataLoader(self.dataset["val"], ...)
```

**Alternative**: Check mode early and skip dataloader creation.

---

### TASK-008: Test All Modes
**Priority**: P1
**Estimated**: 1h

**Test Script**:
```bash
#!/bin/bash
# scripts/test_lazy_datasets.sh

for mode in train eval test predict; do
    echo "Testing mode: $mode"
    uv run python scripts/runners/train.py \
      experiment=parseq_flash_fast \
      mode=$mode \
      trainer.limit_train_batches=2 \
      trainer.limit_val_batches=2 \
      checkpoint_path=null \
      trainer.max_epochs=1

    if [ $? -ne 0 ]; then
        echo "FAILED: mode=$mode"
        exit 1
    fi
done
echo "All modes passed"
```

**Log Validation**: Check that only required datasets are created per mode.

---

## Phase 3: Model Weight Investigation (P2)

### TASK-009: Add Model Creation Profiling
**File**: `ocr/pipelines/orchestrator.py:100`
**Priority**: P2
**Estimated**: 1h

**Add Debug Logging**:
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
    logger.info(f"   ✓ Model created: {type(model).__name__}")
```

**Run Training**: Check logs for duplicate stack traces indicating 2x model creation.

---

### TASK-010: Check Vocab Injection for Model Creation
**File**: `ocr/pipelines/strategies/recognition_config.py`
**Priority**: P2
**Estimated**: 30m

**Question**: Does `inject_vocab_size()` create a temporary model to validate architecture?

**Investigate**:
```python
@staticmethod
def inject_vocab_size(cfg: DictConfig):
    # Check if this instantiates model
    # If yes: refactor to inspect config only
```

**If Issue Found**: Refactor to compute vocab size without model instantiation.

---

### TASK-011: Document Findings
**File**: `specs/002-training-performance-optimization/findings.md`
**Priority**: P2
**Estimated**: 30m

Document:
- Root cause of 2x weight loading
- Fix implemented (if any)
- Reason if unavoidable (e.g., timm caching prevents duplication)

---

## Phase 4: Debug Logging Cleanup (P2)

### TASK-012: Remove Validation Debug Prints
**File**: `ocr/domains/recognition/module.py:114-128`
**Priority**: P2
**Estimated**: 15m

**Before**:
```python
if batch_idx == 0:
    print(f"\n[Validation Debug] Samples:")
    print(f"  Pred Type: {type(inference_out)}")
    # ... more prints
```

**After**:
```python
if batch_idx == 0:
    logger.debug(f"Validation samples - Pred type: {type(inference_out)}")
    if logger.isEnabledFor(logging.DEBUG):
        # Verbose output only in debug mode
        logger.debug(f"Pred keys: {list(inference_out.keys())}")
```

**Validation**: Run training, verify no prints unless `global.debug=true`.

---

## Phase 5: Performance Benchmarking (P1)

### TASK-013: Baseline Measurement
**Priority**: P1
**Estimated**: 30m

**Script**:
```bash
# scripts/benchmark_startup_time.sh
echo "=== Baseline (before optimization) ==="
for i in {1..5}; do
    echo "Run $i:"
    /usr/bin/time -v uv run python scripts/runners/train.py \
      experiment=parseq_flash_fast \
      trainer.limit_train_batches=0 \
      checkpoint_path=null 2>&1 | grep "Elapsed"
done
```

**Record**: Average time from "Orchestrator initialized" to "Trainer ready".

---

### TASK-014: Post-Optimization Measurement
**Priority**: P1
**Estimated**: 30m

**Same Script**: Run after all optimizations complete.

**Compare**:
- Tokenizer load count: 5 → 1 ✓
- Model weight loads: 2 → 1 ✓ (if fixable)
- Startup time reduction: ≥1.5s ✓
- Dataset creation: Only required splits ✓

---

## Optional: Config Caching (P3)

### TASK-015: Implement Config Hash Cache
**File**: `ocr/pipelines/orchestrator.py:200`
**Priority**: P3
**Estimated**: 1h

**Only If**: `log_config=true` becomes common.

**Implementation**:
```python
import hashlib

_CONFIG_HASH_CACHE: Dict[str, str] = {}

def _get_config_yaml(self) -> str:
    cfg_snapshot = OmegaConf.to_yaml(self.cfg, resolve=False)
    config_hash = hashlib.md5(cfg_snapshot.encode()).hexdigest()

    if config_hash not in _CONFIG_HASH_CACHE:
        yaml_str = OmegaConf.to_yaml(self.cfg, resolve=True)
        _CONFIG_HASH_CACHE[config_hash] = yaml_str

    return _CONFIG_HASH_CACHE[config_hash]
```

**Skip**: Low priority, defer until Phase 4.

---

## Summary

**Critical Path**:
1. TASK-001 → TASK-002 → TASK-003 → TASK-004 (Tokenizer caching)
2. TASK-005 → TASK-006 → TASK-007 → TASK-008 (Lazy datasets)
3. TASK-013 → TASK-014 (Benchmarking)

**Parallel Work**:
- TASK-009 → TASK-010 → TASK-011 (Investigation, can run anytime)
- TASK-012 (Cleanup, independent)

**Total Estimated Time**: 8-10 hours (2 days with testing)
