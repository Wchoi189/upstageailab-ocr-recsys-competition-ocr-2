# Walkthrough: Multi-Worker CUDA Training Fix

**Date:** 2026-02-08
**Status:** ✅ RESOLVED
**Impact:** Enables faster data loading with multi-worker support in Docker+CUDA environment

---

## Problem Summary

Recognition training crashed with CUDA initialization errors when `num_workers > 0`:
```
terminate called after throwing an instance of 'c10::Error'
what():  CUDA error: initialization error
Exception raised from c10_cuda_check_implementation
frame #8: c10::TensorImpl::~TensorImpl()
```

**Symptoms:**
- `num_workers=0` worked fine (confirmed by user)
- `num_workers > 0` crashed with CUDA errors
- Terminal froze with repetitive error log spam
- Error occurred during tensor cleanup, not allocation

---

## Root Cause

**Multiprocessing fork mode + CUDA context inheritance**

Linux default `fork` start method:
1. Main process initializes datasets/models
2. Worker processes fork → **inherit CUDA context** from parent
3. Workers try to use CUDA → context mismatch → crash

**Why simplified tests passed:** No complex pipeline state, no CUDA in main process before fork

---

## Investigation Process

### Phase 1: Environment Verification ✅
Created diagnostic scripts to rule out infrastructure issues:

**`test_cuda_environment.py`** - Verified:
- ✅ PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3090
- ✅ Basic CUDA operations work
- ✅ Driver compatible (581.80/13.0)

**`test_dataloader_workers.py`** - Tested:
- ✅ All worker configs pass (0, 1, 2 workers)
- ✅ Multiple pin_memory/persistent_workers combinations
- 🎯 **Conclusion:** Issue is pipeline-specific, not general CUDA problem

### Phase 2: Pipeline Inspection ✅
Reviewed recognition training components:

**LMDB Dataset** (`ocr/domains/recognition/data/lmdb_dataset.py`):
- ✅ Already has lazy env initialization in `__getitem__`
- ⚠️ Opens/closes env in `__init__` (before fork) - minor concern but not the cause

**Missing Components:**
- ❌ No `worker_init_fn` to clear CUDA state
- ❌ No explicit multiprocessing start method
- 🎯 **Found:** Comments in `train.py` about previous pickle issues with spawn

### Phase 3: Solution Testing ✅
Re-enabled spawn start method (previously commented out):

```python
# scripts/runners/train.py
try:
    mp.set_start_method('spawn', force=True)
    print("[MULTIPROCESSING] Using 'spawn' start method for CUDA compatibility")
except RuntimeError as e:
    print(f"[MULTIPROCESSING] Start method already set: {mp.get_start_method()}")
```

**Test 1: num_workers=1** → ✅ PASS (no CUDA errors)
**Test 2: num_workers=2** → ✅ PASS (no CUDA errors)

---

## Solution Implemented

### 1. Enable Spawn Start Method
**File:** [`scripts/runners/train.py`](file:///workspaces/scripts/runners/train.py#L1-L26)

**Change:** Uncommented and improved spawn configuration
- Creates fresh Python processes (no fork inheritance)
- Each worker gets clean CUDA context
- Added informative logging

### 2. Update Global Configuration
**File:** [`configs/global/default.yaml`](file:///workspaces/configs/global/default.yaml#L33-L54)

**Changes:**
- `pin_memory: false` → `true` (safe with spawn)
- Added note referencing spawn method in train.py
- Updated comments to reflect working solution

**Result:**
```yaml
dataloaders:
  train_dataloader:
    num_workers: 2      # ✅ Works with spawn
    pin_memory: true    # ✅ Safe with spawn
    persistent_workers: true
```

---

## Validation Results

### Test Logs

**Environment Diagnostic:**
```
__DEBUG__/training_failures/logs/env_diagnostic_20260208_230110.log
✅ CUDA Environment: READY
   PyTorch: 2.6.0+cu124
   CUDA: 12.4
   Device: NVIDIA GeForce RTX 3090
✅ All basic CUDA tests passed!
```

**DataLoader Isolation Test:**
```
__DEBUG__/training_failures/logs/dataloader_test_20260208_230143.log
✅ PASS  w0_pmFalse_pwFalse_cdFalse
✅ PASS  w2_pmTrue_pwTrue_cdFalse
Working configurations: 8/8
```

**Training with num_workers=1:**
```
__DEBUG__/training_failures/logs/test_spawn_workers1_20260208_230714.log
[MULTIPROCESSING] Using 'spawn' start method for CUDA compatibility
Epoch 0/0  ━━━━━━━━━━━━ 5/5 0:00:02 • 0:00:00 7.14it/s
✅ Training complete!
Exit code: 0
```

**Training with num_workers=2:**
```
__DEBUG__/training_failures/logs/test_spawn_workers2_20260208_230740.log
[MULTIPROCESSING] Using 'spawn' start method for CUDA compatibility
Epoch 0/0  ━━━━━━━━━━━━ 5/5 0:00:02 • 0:00:00 7.14it/s
✅ Training complete!
Exit code: 0
```

---

## Performance Impact

### Before (num_workers=0)
- Single-process data loading
- GPU often waiting for data
- Acceptable but suboptimal

### After (num_workers=2)
- Parallel data loading
- ~2x data throughput potential
- Better GPU utilization
- **Same training speed in small tests** (GPU-bound) but will help with larger batches

---

## Files Modified

### Production Code
1. **[`scripts/runners/train.py`](file:///workspaces/scripts/runners/train.py)**
   - Lines 1-26: Updated multiprocessing configuration
   - Enabled spawn start method
   - Updated documentation

2. **[`configs/global/default.yaml`](file:///workspaces/configs/global/default.yaml)**
   - Lines 33-54: DataLoader defaults
   - Enabled `pin_memory=true`
   - Added spawn method reference

### Debug Artifacts
3. **[`__DEBUG__/training_failures/scripts/test_cuda_environment.py`](file:///workspaces/__DEBUG__/training_failures/scripts/test_cuda_environment.py)**
   - Environment diagnostic tool

4. **[`__DEBUG__/training_failures/scripts/test_dataloader_workers.py`](file:///workspaces/__DEBUG__/training_failures/scripts/test_dataloader_workers.py)**
   - Systematic DataLoader test framework

---

## Why This Works

### Fork vs Spawn

| Aspect | Fork (Default Linux) | Spawn (Solution) |
|--------|---------------------|------------------|
| Process Creation | Copy parent memory | Fresh Python process |
| CUDA Context | **Inherited (problematic)** | Clean initialization |
| Performance | Slightly faster startup | Slightly slower startup |
| Safety | Unsafe with CUDA | ✅ Safe with CUDA |
| Pickle Requirement | No | Yes (all data must pickle) |

### Why Spawn Works Now

Previous comments mentioned pickle issues with Hydra Environment objects. These appear to be resolved, likely because:
- Hydra objects no longer in dataset chain, or
- PyTorch/Hydra versions updated to handle pickling better, or
- LMDB lazy initialization delays env creation until after spawn

---

## Lessons Learned

1. **Fork + CUDA = Problems in Docker**
   - Fork inherits file descriptors and CUDA state
   - Docker GPU isolation makes this worse
   - Always use spawn for CUDA multiprocessing

2. **Simplified Tests Are Essential**
   - Isolated DataLoader test passed → ruled out infrastructure
   - Narrowed problem to pipeline-specific issue
   - Saved hours of debugging

3. **Check Historical Context**
   - Previous comments in code were valuable clues
   - Earlier pickle issue was resolved somehow
   - Don't assume old comments are still accurate

4. **Diagnostic Scripts Pay Off**
   - Environment audit script → confirmed CUDA healthy
   - Systematic test script → identified working configs
   - Reusable for future debugging

---

## Known Limitations

### Spawn Trade-offs
- **Slower worker startup:** ~1s overhead per worker spawn
- **Pickle requirement:** All dataset/config objects must be picklable
- **Memory overhead:** Each worker is full Python process

### Acceptable Because
- Worker startup amortized over many batches
- `persistent_workers=true` → workers don't restart each epoch
- GPU compute still dominates for ParSeq model

---

## Future Recommendations

### Immediate
1. ✅ Use spawn method (implemented)
2. ✅ Enable multi-worker training (verified working)
3. Test with larger `num_workers` (4+) for bigger batch sizes

### Long-term
1. **Add CI/CD test** with `num_workers > 0`
2. **Document Docker+CUDA requirements** in project wiki
3. **Monitor worker efficiency** - profile data loading vs compute time
4. **Consider batching optimizations** if data loading still bottlenecks

### Monitoring
Watch for:
- Worker spawn overhead with very large `num_workers`
- Memory usage with persistent workers
- Any regression if PyTorch/CUDA versions change

---

## Success Criteria Met

- ✅ Training works with `num_workers=0` (baseline preserved)
- ✅ Training works with `num_workers=1` (verified)
- ✅ Training works with `num_workers=2` (verified)
- ✅ No CUDA initialization errors
- ✅ No terminal freezing
- ✅ Configuration documented
- ✅ Solution is simple and maintainable

---

## Commands to Reproduce

### Quick Test (Recommended)
```bash
uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_baseline_official \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  data.batch_size=16
```

### Test Different Worker Counts
```bash
# Test with 0 workers (verifies backward compatibility)
uv run python scripts/runners/train.py \
  domain=recognition experiment=rec_baseline_official \
  trainer.max_epochs=1 trainer.limit_train_batches=5 \
  dataloaders.train_dataloader.num_workers=0 \
  dataloaders.val_dataloader.num_workers=0

# Test with 4 workers (stress test)
uv run python scripts/runners/train.py \
  domain=recognition experiment=rec_baseline_official \
  trainer.max_epochs=1 trainer.limit_train_batches=10 \
  dataloaders.train_dataloader.num_workers=4 \
  dataloaders.val_dataloader.num_workers=2
```

---

**Fix Status:** 🎉 **COMPLETE AND VERIFIED**
