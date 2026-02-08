# Fix Multi-Worker DataLoader CUDA Failure in Recognition Pipeline

## Problem Statement

The recognition training pipeline crashes with CUDA initialization errors when `num_workers > 0`, but:
- ✅ `num_workers=0` works fine (as confirmed by user)
- ✅ Simplified DataLoader tests with CUDA PASS with 0-2 workers
- ❌ Actual recognition training FAILS with workers

**Conclusion:** Issue is **pipeline-specific**, likely in:
- LMDB dataset implementation
- Data transforms/augmentations
- Model initialization in workers
- Custom collate functions

---

## User Review Required

> [!IMPORTANT]
> **Critical Unknown**: You mentioned "num_workers 0 works, but multiple workers fail"
>
> Questions:
> 1. Does `num_workers=0` complete training successfully (beyond just starting)?
> 2. Is there a performance requirement necessitating multi-worker support?
> 3. Is `num_workers=0` acceptable as a workaround if multi-worker fix is complex?
>
> **Current Strategy**: Investigate root cause and provide both fix AND workaround options.

---

## Investigation Plan

### Phase 1: Isolate Pipeline Components

Test each component individually with workers to find the culprit:

#### 1.1 LMDB Dataset Reading
**File:** `ocr/domains/recognition/data/lmdb_dataset.py`

**Hypothesis:** LMDB environment not fork-safe or has file handle issues

**Tests:**
- Verify LMDB `readahead=False` setting
- Check if LMDB env is created per-worker vs shared
- Test with dummy LMDB database

#### 1.2 Data Transforms
**Files:** `ocr/domains/recognition/data/transforms.py`, augmentation modules

**Hypothesis:** Transforms create CUDA tensors in workers

**Tests:**
- Disable all transforms, test with raw data
- Re-enable transforms one-by-one
- Check for accidental `.cuda()` calls in transforms

#### 1.3 Model/Tokenizer in DataLoader
**Files:** `ocr/domains/recognition/data/tokenizer.py`, dataset `__init__`

**Hypothesis:** Tokenizer or model components initialized in workers

**Tests:**
- Check if tokenizer creates any CUDA state
- Verify no model weights loaded in dataset init
- Test with dummy labels (no tokenizer)

#### 1.4 Custom Collate Function
**File:** `ocr/domains/recognition/data/datamodule.py` or similar

**Hypothesis:** Collate function has CUDA operations

**Tests:**
- Use default collate function
- Check for `.to(device)` in collate

---

## Proposed Changes

### Strategy A: Fix Root Cause (Preferred)

Based on investigation, likely fixes:

#### A1: LMDB Environment Per-Worker
**File:** [`ocr/domains/recognition/data/lmdb_dataset.py`](file:///workspaces/ocr/domains/recognition/data/lmdb_dataset.py)

```python
def __init__(self, ...):
    # Store path, don't open LMDB here
    self.lmdb_path = lmdb_path
    self.env = None

def __getitem__(self, idx):
    # Lazy initialization per-worker
    if self.env is None:
        import lmdb
        self.env = lmdb.open(self.lmdb_path, readonly=True, readahead=False, lock=False)
    # ... rest of code
```

#### A2: Remove CUDA from Transforms
Audit all transform code for:
- `.cuda()` calls
- `torch.device("cuda")` device creation
- GPU-specific operations

Ensure all transforms stay CPU-only.

#### A3: Worker Init Function
**File:** Create/modify DataModule

```python
def worker_init_fn(worker_id):
    """Initialize worker-specific state"""
    import torch
    # Clear any inherited CUDA state
    torch.cuda.empty_cache()
    # Set different random seed per worker
    np.random.seed(torch.initial_seed() % 2**32)
```

Then in DataLoader:
```python
DataLoader(..., worker_init_fn=worker_init_fn)
```

---

### Strategy B: Enhanced Configuration (Fallback)

If root cause fix is complex, optimize `num_workers=0` setup:

#### B1: Increase Batch Size
**File:** [`configs/experiment/rec_baseline_official.yaml`](file:///workspaces/configs/experiment/rec_baseline_official.yaml)

```yaml
data:
  batch_size: 128  # Increase from 64 to amortize data loading
```

#### B2: Prefetch to GPU
Add async data prefetching in training loop to hide CPU data loading latency.

#### B3: Documentation
Update config with clear warning and performance notes.

---

### Strategy C: Multiprocessing Start Method (Quick Test)

**File:** `scripts/runners/train.py` or training entry point

```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Before any CUDA operations
```

Test if spawn vs fork resolves the issue.

---

## Verification Plan

### Automated Tests

#### Test 1: Component Isolation Tests
```bash
# Create test script: test_pipeline_components.py
# Test LMDB reading with workers
uv run python __DEBUG__/training_failures/scripts/test_pipeline_components.py --component lmdb --workers 2

# Test transforms with workers
uv run python __DEBUG__/training_failures/scripts/test_pipeline_components.py --component transforms --workers 2

# Test tokenizer with workers
uv run python __DEBUG__/training_failures/scripts/test_pipeline_components.py --component tokenizer --workers 2
```

**Expected:** Identifies which component causes failure

#### Test 2: Training with num_workers=0
```bash
timeout 60 uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_baseline_official \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  dataloaders.train_dataloader.num_workers=0 \
  dataloaders.val_dataloader.num_workers=0 \
  2>&1 | tee __DEBUG__/training_failures/logs/verify_workers0.log
```

**Expected:** Completes without errors (baseline confirmation)

#### Test 3: Training with num_workers=1 (Post-Fix)
```bash
timeout 60 uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_baseline_official \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  dataloaders.train_dataloader.num_workers=1 \
  dataloaders.val_dataloader.num_workers=1 \
  2>&1 | tee __DEBUG__/training_failures/logs/verify_workers1.log
```

**Expected:** Completes without errors (after fix applied)

#### Test 4: Training with num_workers=2 (Post-Fix)
```bash
timeout 60 uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_baseline_official \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  dataloaders.train_dataloader.num_workers=2 \
  dataloaders.val_dataloader.num_workers=2 \
  2>&1 | tee __DEBUG__/training_failures/logs/verify_workers2.log
```

**Expected:** Completes without errors (full verification)

### Manual Verification

**Test 5: User Validation**
1. Run full training (no limits) with num_workers=0 for 1 epoch
2. Check training completes and metrics look reasonable
3. Run same training with num_workers=2 (post-fix)
4. Compare training time and verify no crashes

---

## Success Criteria

1. **Minimum (Workaround):** Training works with `num_workers=0`, documented as limitation
2. **Target (Fix):** Training works with `num_workers >= 1`, no CUDA errors
3. **Optimal:** Multi-worker training is faster than single-worker

---

## Files to Investigate

Priority order:

1. **[CRITICAL]** [`ocr/domains/recognition/data/lmdb_dataset.py`](file:///workspaces/ocr/domains/recognition/data/lmdb_dataset.py) - LMDB handling
2. **[HIGH]** [`ocr/domains/recognition/data/datamodule.py`](file:///workspaces/ocr/domains/recognition/data/datamodule.py) - DataLoader config
3. **[HIGH]** [`ocr/domains/recognition/data/tokenizer.py`](file:///workspaces/ocr/domains/recognition/data/tokenizer.py) - Tokenizer state
4. **[MEDIUM]** [`ocr/domains/recognition/data/transforms.py`](file:///workspaces/ocr/domains/recognition/data/transforms.py) - Transform operations
5. **[LOW]** [`scripts/runners/train.py`](file:///workspaces/scripts/runners/train.py) - Multiprocessing start method

---

## Next Steps

1. ✅ Get user feedback on questions above
2. Inspect LMDB dataset implementation for worker compatibility
3. Create component isolation test script
4. Test most likely fix (LMDB lazy init)
5. Verify with automated tests
6. Document findings and update configuration
