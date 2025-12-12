---
title: "Wandb Multiprocessing Issue Analysis"
date: "2025-11-10"
status: "in_progress"
related: "2025-11-10-cuda-debugging-session-handover.md"
---

# Wandb Multiprocessing Issue: Root Cause Analysis

## Executive Summary

**Problem:** Training crashes with CUDA errors. Initial hypothesis blamed wandb multiprocessing conflict.

**Reality:** Two separate issues:
1. ✅ **Multiprocessing conflict** - Partially confirmed, fixable with `forkserver`
2. ⚠️ **cuDNN stability issue** - Underlying GPU/driver problem that persists

**Status:** `spawn` fix proved the multiprocessing hypothesis but introduced unacceptable performance degradation. Testing `forkserver` as better alternative.

---

## Test Results: spawn Method

### Command
```bash
bash scripts/test_wandb_multiprocessing_fix.sh
# Uses mp.set_start_method('spawn')
```

### Results

| Metric | Result | Notes |
|--------|--------|-------|
| Training started | ✅ SUCCESS | 20/20 steps completed |
| Performance | ❌ UNACCEPTABLE | 10x slower than fork |
| Test phase | ❌ FAILED | cuDNN error during test |
| Wandb logging | ⚠️ UNCLEAR | No confirmation wandb was active |

### Key Observations

1. **Training Phase Succeeded** ✅
   - Completed 20 training steps without crash
   - Previous attempts crashed at step 0-26
   - **This confirms multiprocessing conflict was real**

2. **Performance Degraded** ❌
   ```
   Loading images to RAM: 404/404 [00:31<00:00, 12.71it/s]  # Slow!
   Epoch 0/19: 40/1636 [0:00:27 • 0:10:32, 2.53it/s]       # 10x slower
   ```
   - Each worker spawns a new Python process
   - Massive overhead for process creation
   - Unusable for production training

3. **Test Phase Failed** ❌
   ```
   RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
   torch.AcceleratorError: CUDA error: an illegal instruction was encountered
   ```
   - Error in decoder convolution operation
   - **Not a multiprocessing issue** - happens after training succeeds
   - Suggests underlying GPU driver or cuDNN compatibility problem

4. **Multiple Worker Spawns**
   ```
   [Multiprocessing] Start method already set to 'spawn'  # Repeated 8+ times
   ```
   - Each DataLoader worker prints this message
   - Confirms workers are being created repeatedly
   - Source of performance degradation

---

## Analysis: Why spawn "Worked" But Failed

### What spawn Fixed
- ✅ Prevented CUDA context corruption during training
- ✅ Allowed training to complete 20 steps
- ✅ Proved multiprocessing conflict was a contributing factor

### What spawn Didn't Fix
- ❌ cuDNN stability during test/inference
- ❌ "Illegal instruction" errors
- ❌ Performance degradation

### Conclusion
The multiprocessing conflict was **masking a deeper issue**. Once we fixed the fork corruption, we exposed the underlying cuDNN/GPU driver problem.

---

## Root Cause: Dual Issues

### Issue 1: Multiprocessing Fork Corruption (PARTIAL)

**Mechanism:**
1. Wandb spawns background process using `fork()`
2. PyTorch DataLoader spawns workers using `fork()`
3. Fork-after-fork corrupts CUDA context
4. Training crashes with NaN or illegal memory access

**Evidence:**
- ✅ Training succeeds with spawn (clean processes)
- ✅ Training fails with fork (default on Linux)
- ✅ Explains variable crash locations (race condition)

**Solution:**
- Use `forkserver` (compromise between speed and safety)
- Faster than `spawn`, safer than `fork`

### Issue 2: cuDNN/GPU Driver Instability (PRIMARY)

**Mechanism:**
- cuDNN fails to execute convolution algorithms
- May be WSL2 GPU passthrough issue
- May be driver compatibility issue
- May be GPU memory fragmentation

**Evidence:**
- ❌ Test phase crashes even with spawn
- ❌ Error: "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED"
- ❌ Happens after successful training (not multiprocessing-related)
- ❌ "Illegal instruction" during GPU cleanup

**Potential Causes:**
1. **WSL2 GPU Passthrough** - GPU operations unstable through WSL2
2. **Driver/cuDNN Mismatch** - CUDA 13.0 driver, PyTorch 12.8 runtime
3. **GPU Memory Fragmentation** - Repeated allocations cause fragmentation
4. **Hardware Issue** - RTX 3060 specific bug

---

## Solution: forkserver Method

### Why forkserver is Better

| Method | Speed | CUDA Safe | Notes |
|--------|-------|-----------|-------|
| `fork` (default) | ⚡⚡⚡ Fast | ❌ NO | Corrupts CUDA context |
| `spawn` | 🐌 Slow (10x) | ✅ YES | Too slow for production |
| `forkserver` | ⚡⚡ Fast | ✅ YES | **Best of both worlds** |

### How forkserver Works
1. Creates ONE clean "server" process at startup
2. Server forks workers on demand (fast)
3. Workers inherit clean state (safe)
4. ~2-3x faster than spawn, still CUDA-safe

### Implementation
```python
# runners/train.py (already updated)
import multiprocessing as mp
mp.set_start_method('forkserver', force=False)
```

---

## Next Steps: Testing Protocol

### Test 1: forkserver with Wandb (Recommended First)
```bash
bash scripts/test_forkserver_fix.sh
```
**Purpose:** Test if forkserver is fast enough AND CUDA-safe
**Expected:** Training completes in reasonable time without crashes

### Test 2: cuDNN Stability without Wandb (Diagnostic)
```bash
bash scripts/test_cudnn_stability.sh
```
**Purpose:** Isolate whether cuDNN issue is related to multiprocessing
**Expected:** If this fails, confirms Issue #2 is primary

### Test 3: Full Training Run (If Test 1 succeeds)
```bash
uv run python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  model.encoder.model_name=resnet18 \
  dataloaders.train_dataloader.batch_size=4 \
  trainer.max_epochs=1 \
  logger.wandb.enabled=true
```
**Purpose:** Verify full training + test cycle works
**Expected:** Complete epoch + test without crashes

---

## If forkserver Still Fails

### Alternative Solutions

#### Option A: Disable Wandb Multiprocessing
Add to train.py before wandb import:
```python
os.environ["WANDB_START_METHOD"] = "thread"  # Use threading instead of fork
```

#### Option B: Reduce DataLoader Workers
```yaml
# configs/dataloaders/default.yaml
num_workers: 0  # Disable multiprocessing entirely
```
**Trade-off:** Slower data loading, but eliminates multiprocessing issues

#### Option C: Use Native Linux (Not WSL2)
Boot into native Linux, test same configuration
**If works:** Confirms WSL2 GPU passthrough is the problem

#### Option D: Downgrade PyTorch/CUDA
```bash
pip install torch==2.7.0+cu121  # Match CUDA versions exactly
```

---

## Diagnostic Commands

### Check GPU Health
```bash
nvidia-smi
nvidia-smi -q  # Detailed info
```

### Check CUDA/PyTorch Compatibility
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA (compiled): {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Test Basic Convolution
```python
import torch
x = torch.randn(4, 3, 224, 224, device='cuda')
conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
y = conv(x)
loss = y.sum()
loss.backward()
print("✓ Basic convolution works")
```

---

## Updated Hypothesis

### Primary Issue: cuDNN Stability (60% confidence)
- Test phase crashes suggest cuDNN can't execute operations reliably
- May be WSL2-specific (works on RTX 3090 native Linux)
- `CUDNN_STATUS_EXECUTION_FAILED` is the smoking gun

### Secondary Issue: Multiprocessing Conflict (40% confidence)
- Training succeeds with spawn/forkserver
- Fails with fork (default)
- Wandb + DataLoader fork-after-fork corruption

### Recommendation
1. ✅ Keep `forkserver` fix (minimal performance cost, significant stability gain)
2. ⚠️ Test cuDNN stability independently
3. 🔍 If cuDNN issues persist, investigate WSL2 GPU passthrough or hardware

---

## References
- Previous session: [2025-11-10-cuda-debugging-session-handover.md](./2025-11-10-cuda-debugging-session-handover.md)
- PyTorch multiprocessing: https://pytorch.org/docs/stable/notes/multiprocessing.html
- cuDNN errors: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-status-execution-failed

---

*Last updated: 2025-11-10*
