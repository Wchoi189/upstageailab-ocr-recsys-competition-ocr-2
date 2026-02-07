# Training Issue Diagnostic Plan

## Executive Summary

Based on the [previous walkthrough](file:///home/vscode/.gemini/antigravity/brain/c77f3ff0-af7c-46ab-8e30-c25d4d478fab/walkthrough.md.resolved), the training run exhibits **multiprocessing/CUDA initialization conflicts** when using `num_workers > 0`, resulting in SIGABRT crashes. With `num_workers=0`, training starts but becomes extremely slow or stalls.

**No orphaned processes detected.** System resources are adequate (8GB `/dev/shm`, RTX 3090 with 24GB VRAM).

---

## Root Cause Analysis

### Primary Issue: PyTorch Multiprocessing + CUDA Conflict

**Symptom:** SIGABRT/CUDA initialization crashes with `num_workers=4` (default)

**Root Cause:** PyTorch is using `fork` as the multiprocessing start method, which is **incompatible with CUDA** when:
- CUDA context is initialized in the parent process before forking
- Child processes inherit CUDA state, causing corruption

**Verification:**
```bash
python -c "import torch.multiprocessing as mp; print(mp.get_start_method())"
# Output: fork  ← THIS IS THE PROBLEM
```

> [!CAUTION]
> **Fork + CUDA = Crashes**
> 
> PyTorch documentation explicitly warns against using `fork` with CUDA. The default on Linux is `fork`, but CUDA requires `spawn` or `forkserver`.

### Secondary Issue: Slow Training with num_workers=0

**Symptom:** Training starts but becomes extremely slow (~0.20 it/s or stalls)

**Potential Causes:**
1. **CPU-bound data loading** - Single-threaded dataset access
2. **GPU transfer bottleneck** - Without async data loading, GPU waits for CPU
3. **Batch preparation overhead** - Synchronous collation and preprocessing

**Dataset verification shows I/O is NOT the bottleneck:**
- LMDB read latency: ~3ms per item
- Batch load time: 0.017s for 64 samples
- Disk I/O is blazing fast ✅

---

## Environment Assessment

### ✅ Resources Available
| Resource | Status | Details |
|----------|--------|---------|
| GPU | Available | RTX 3090, 24GB VRAM, CUDA 12.4 |
| `/dev/shm` | Adequate | 8GB, empty |
| Memory | Adequate | Unlimited ulimits |
| Processes | Clean | No orphaned training processes |

### ⚠️ Configuration Issues
| Component | Current | Problem |
|-----------|---------|---------|
| Multiprocessing | `fork` | Incompatible with CUDA |
| `num_workers` | 4 (default) | Triggers crashes |
| Start method | Not set | Defaults to `fork` on Linux |

---

## Proposed Debugging Strategy

### Phase 1: Fix Multiprocessing Start Method ⭐ **HIGH PRIORITY**

The most critical fix is to change PyTorch's multiprocessing start method to `spawn` or `forkserver`.

**Option A: Set in Training Script** (Recommended)
```python
# Add to scripts/runners/train.py, BEFORE any CUDA/torch imports
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

**Option B: Environment Variable**
```bash
export PYTORCH_MULTIPROCESSING_START_METHOD=spawn
```

**Option C: Hydra Config Override**
Add to experiment config:
```yaml
defaults:
  - override /runtime/multiprocessing: spawn
```

> [!IMPORTANT]
> **Why `spawn`?**
> 
> - `spawn`: Creates fresh child processes without inheriting parent state. **Most compatible with CUDA.**
> - `forkserver`: Uses a server process to spawn children. Faster than `spawn` but more complex.
> - `fork`: Copies parent memory (fast but breaks CUDA).

### Phase 2: Optimize DataLoader Configuration

After fixing the start method, optimize DataLoader settings:

**Recommended Settings for Testing:**
```yaml
data:
  num_workers: 2  # Start conservative, scale up if stable
  pin_memory: true  # Async GPU transfer
  persistent_workers: true  # Reuse workers across epochs
  prefetch_factor: 2  # Prefetch batches per worker
```

**Gradual Scaling:**
1. Start with `num_workers=1` to verify fix
2. Increase to `num_workers=2` 
3. Monitor for crashes, scale to 4-8 if stable

### Phase 3: Verify with Micro-Training Run

**Test Command:**
```bash
python scripts/runners/train.py experiment=rec_baseline_v1 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=2 \
  trainer.max_epochs=5 \
  ++trainer.log_every_n_steps=1 \
  ++data.num_workers=1 \
  ++trainer.num_sanity_val_steps=0
```

**Success Criteria:**
- ✅ No SIGABRT/CUDA crashes
- ✅ Training completes 5 epochs without stalling
- ✅ Loss decreases (model learns from micro-dataset)
- ✅ Predictions are not gibberish

### Phase 4: Full Training Run (If Micro-Run Succeeds)

Once micro-training is verified:
1. Scale up `num_workers` (2 → 4 → 8)
2. Remove batch limits
3. Run full training with monitoring

---

## Areas Requiring Investigation

### 1. **Multiprocessing Configuration** 🔴 **CRITICAL**
- **Action:** Change start method to `spawn`
- **Priority:** HIGHEST - This is likely the root cause
- **Effort:** Low (single line change)
- **Risk:** Low (well-documented fix)

### 2. **DataLoader Settings** 🟡 **MEDIUM**
- **Action:** Tune `num_workers`, `pin_memory`, `persistent_workers`
- **Priority:** MEDIUM - After fixing start method
- **Effort:** Medium (iterative testing)
- **Risk:** Low (safe to experiment)

### 3. **CUDA/PyTorch Environment** 🟢 **LOW**
- **Action:** Verify no CUDA initialization before fork
- **Priority:** LOW - Environment looks clean
- **Effort:** Low (code review)
- **Risk:** Low

### 4. **Model/Training Configuration** 🟢 **LOW**
- **Action:** Review for unnecessary early CUDA allocation
- **Priority:** LOW - Config looks standard
- **Effort:** Low
- **Risk:** Low

---

## Immediate Next Steps

1. **Implement multiprocessing fix** (Option A recommended)
   - Edit `scripts/runners/train.py`
   - Add `mp.set_start_method('spawn', force=True)` at top
   
2. **Run micro-training test**
   - Use `num_workers=1` initially
   - Verify no crashes
   
3. **If successful, scale up**
   - Increase `num_workers` to 2, then 4
   - Monitor stability
   
4. **Document results**
   - Record iteration times
   - Note any remaining issues
   - Update walkthrough

---

## Questions for User

1. **Preferred approach for multiprocessing fix?**
   - Option A (script modification) - cleanest
   - Option B (environment variable) - temporary
   - Option C (config-based) - requires creating config

2. **Risk tolerance for micro-run?**
   - Conservative: `num_workers=1`, very short run (5 epochs)
   - Moderate: `num_workers=2`, standard run (10-20 epochs)
   - Aggressive: `num_workers=4`, full micro-run

3. **Should we proceed with automatic fix?**
   - I can implement Option A immediately and run micro-training
   - Or would you prefer to review this plan first?

---

## References

- **Previous Micro-Training Walkthrough:** [walkthrough.md](file:///home/vscode/.gemini/antigravity/brain/c77f3ff0-af7c-46ab-8e30-c25d4d478fab/walkthrough.md.resolved)
- **Session Handover Document:** [session_handover.md](file:///workspaces/dev_tools/project_compass/history/ocr-domain-refactor/20260207_005500_dataset_regeneration_and_fix/session_handover.md)
- **Training Script:** [train.py](file:///workspaces/scripts/runners/train.py)
- **Experiment Config:** [rec_baseline_v1.yaml](file:///workspaces/configs/experiment/rec_baseline_v1.yaml)
- **PyTorch Multiprocessing Docs:** https://pytorch.org/docs/stable/notes/multiprocessing.html
