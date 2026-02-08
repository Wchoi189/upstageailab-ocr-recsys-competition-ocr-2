# Consolidated Findings - Training Failures

## Executive Summary

Multiple distinct failure modes prevent training:
1. **Spawn method fails** - Pickle errors with Hydra Environment
2. **num_workers=0 doesn't work** - Workers spawn anyway and segfault
3. **Configuration override mystery** - CLI args not reaching DataLoaders

**Critical Issue:** Even basic training with `num_workers=0` fails unexpectedly.

---

## Problem 1: Spawn Method + Pickle Incompatibility

### Symptoms
- `TypeError: cannot pickle 'Environment' object`
- Occurs immediately when using `mp.set_start_method('spawn')`
- Blocks all multiprocessing attempts

### Root Cause
Hydra `Environment` objects stored somewhere in:
- Dataset instances
- Transform pipelines
- Tokenizer initialization
- DataModule configuration

### Why It Matters
- Spawn is **required** for CUDA + multiprocessing on Linux
- Fork + CUDA + multiprocessing = crashes (as seen in previous session)
- Can't use multiple workers without spawn

### Solution Path
1. Find where Environment objects are stored
2. Convert `DictConfig` → plain dict: `OmegaConf.to_container(cfg, resolve=True)`
3. Re-test with spawn method

### Priority
🟡 Medium - Blocks multiprocessing but not immediate issue

---

## Problem 2: Configuration Override Not Propagating 🔴 CRITICAL

### Symptoms
```bash
# Explicit override:
++data.num_workers=0

# Expected: No workers spawn
# Actual: Worker 20447 spawned and segfaulted
```

### Evidence
- CLI override: `++data.num_workers=0`
- PyTorch created worker process anyway
- Worker immediately segfaulted

### Hypotheses

#### Hypothesis A: Hydra Config Precedence Issue
```
CLI Override (++) → Experiment Config → Domain Config → Default
                   ↑ One of these might override the CLI
```

Possible culprits:
- `configs/experiment/rec_baseline_v1.yaml` sets `data.batch_size: 64` but nothing about workers
- `configs/data/runtime/performance/balanced.yaml` might set workers
- Multiple config sources merging incorrectly

#### Hypothesis B: Separate Train/Val Configs
DataModule might receive different configs for:
- `train_dataloader()` - Uses one config
- `val_dataloader()` - Uses another config

Evidence: Crash happened in **validation loop**, not training.

#### Hypothesis C: DataModule Ignores Config
`ocr/data/lightning_data.py` might hardcode values or use defaults.

### Impact
🔴 **CRITICAL** - Blocks all testing. Can't control DataLoader behavior.

### Investigation Required
1. Add debug logging to DataModule
2. Print resolved Hydra config
3. Check for hardcoded values

---

## Problem 3: Worker Segfault During Validation

### Symptoms
```
RuntimeError: DataLoader worker (pid 20447) is killed by signal: Segmentation fault.
```

### Location
```python
training_epoch_loop.py:406
  → val_loop.run()  # Validation phase
  → evaluation_loop.py:139
  → next(data_fetcher)  # First validation batch
```

### Why It Matters
- Training setup **succeeded** (model loaded, datasets created)
- Crash is validation-specific
- Suggests different config for val_dataloader

### Related to Problem 2
If validation dataloader uses different config, it might:
- Have `num_workers > 0` despite override
- Trigger the segfault seen in previous session

### Investigation Required
Compare train vs val dataloader creation in `lightning_data.py`

---

## Recommended Debugging Steps

### Step 1: Add DataModule Logging ⭐ HIGHEST PRIORITY

**File:** `ocr/data/lightning_data.py`

**Changes:**
```python
def train_dataloader(self) -> DataLoader:
    logger.info(f"🔍 Creating TRAIN DataLoader:")
    logger.info(f"  - num_workers: {self.num_workers}")
    logger.info(f"  - batch_size: {self.batch_size}")
    logger.info(f"  - pin_memory: {self.pin_memory}")
    return DataLoader(...)

def val_dataloader(self) -> DataLoader:
    logger.info(f"🔍 Creating VAL DataLoader:")
    logger.info(f"  - num_workers: {self.num_workers}")
    logger.info(f"  - batch_size: {self.batch_size}")
    return DataLoader(...)
```

**Expected Outcome:** Confirm actual values reaching DataLoader

---

### Step 2: Print Resolved Hydra Config

**File:** `ocr/pipelines/orchestrator.py`

**Changes:**
```python
def setup_modules(self):
    logger.info("📋 RESOLVED DATA CONFIG:")
    logger.info(OmegaConf.to_yaml(self.cfg.data))
    ...
```

**Expected Outcome:** See if `num_workers: 0` appears

---

### Step 3: Test Validation Skip Workaround

**Command:**
```bash
python scripts/runners/train.py experiment=rec_baseline_v1 \
  ++trainer.limit_val_batches=0 \
  ++data.num_workers=0 \
  trainer.limit_train_batches=10 \
  trainer.max_epochs=3
```

**Expected Outcome:**
- If succeeds → Problem is validation-specific
- If fails → Problem is deeper

---

### Step 4: Create Minimal Test

**File:** `__DEBUG__/training_failures/scripts/test_dataloader.py`

```python
#!/usr/bin/env python
"""Minimal DataLoader test - bypasses Hydra entirely."""
import torch
from torch.utils.data import DataLoader
from ocr.domains.recognition.data.datasets import AIHubLMDBDataset

# Hardcoded - no Hydra
dataset = AIHubLMDBDataset(
    lmdb_path="/workspaces/data/processed/recognition/aihub_lmdb_validation",
    max_label_length=25
)

loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=0,  # Hardcoded!
    shuffle=False
)

print(f"✅ DataLoader created with num_workers=0")
print(f"   Dataset size: {len(dataset)}")

for i, batch in enumerate(loader):
    print(f"✅ Batch {i} loaded: {batch['images'].shape}")
    if i >= 2:
        break

print("✅ Test completed successfully!")
```

**Expected Outcome:**
- If succeeds → Hydra config is the problem
- If fails → Dataset/LMDB corruption

---

## Files Requiring Investigation

| Priority | File | Purpose |
|----------|------|---------|
| 🔴 | `ocr/data/lightning_data.py` | DataModule - where DataLoaders created |
| 🔴 | `configs/data/datasets/recognition.yaml` | Dataset config source |
| 🟡 | `configs/data/runtime/performance/balanced.yaml` | Runtime config defaults |
| 🟡 | `ocr/pipelines/orchestrator.py` | Config resolution point |
| 🟢 | `ocr/domains/recognition/data/datasets.py` | Dataset implementation |

---

## Workarounds to Try

### Option A: Skip Validation Entirely
```bash
++trainer.limit_val_batches=0
```
**Pros:** Bypass validation crash
**Cons:** No validation metrics

### Option B: Use fast_dev_run
```bash
++trainer.fast_dev_run=5
```
**Pros:** Built-in Lightning test mode
**Cons:** Very limited

### Option C: Revert to Previous Environment
If configuration worked in previous session (c77f3ff0), check:
- What changed in environment?
- Different Hydra version?
- Different config files?

---

## Next Actions

1. ✅ **Created debug workspace** (`__DEBUG__/training_failures/`)
2. ⏭️ **Add DataModule logging** (Step 1)
3. ⏭️ **Test validation skip** (Step 3)
4. ⏭️ **Create minimal test** (Step 4)
5. ⏭️ **Fix pickle issue** (after config issue resolved)

### Solution Found (2026-02-07)
**Root Cause**: Vocabulary size mismatch (Tokenizer: 1027 vs Model Default: ~1000). The `OCRProjectOrchestrator` failed to inject the correct vocab size into the model config due to logic that ignored flat configuration structures used by PARSeq.

**Fix**:
Updated `ocr/pipelines/orchestrator.py` to robustly inject `vocab_size` and `out_features` variables into the model configuration, handling both legacy nested `params` and modern flat config structures.

**Verification**:
Training `rec_baseline_v1` is stable beyond 300 iterations with no CUDA assertion errors.

**Artifacts**:
- Bug Report: `/workspaces/docs/artifacts/bug_reports/2026-02-07_0419_bug_20260207_vocab-injection.md`
- Fix PR/File: `ocr/pipelines/orchestrator.py`
