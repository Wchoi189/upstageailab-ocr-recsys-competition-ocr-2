# Investigation Log - Training Failures

## 2026-02-07 - Initial Debugging Session

### Background
Previous micro-training attempt (conversation c77f3ff0) showed:
- `num_workers=4` (default) → SIGABRT/CUDA crashes
- `num_workers=0` → Training started but extremely slow

### Test 1: Spawn Method Investigation
**Time:** 01:19:00
**Objective:** Fix CUDA+multiprocessing crash by using spawn method

**Configuration:**
- Modified `/workspaces/scripts/runners/train.py`
- Added: `mp.set_start_method('spawn', force=True)`
- Command: `python scripts/runners/train.py experiment=rec_baseline_v1 ++data.num_workers=1`

**Result:** ❌ FAILED
**Error:** `TypeError: cannot pickle 'Environment' object`

**Analysis:**
```
Traceback (most recent call last):
  File ".../multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
TypeError: cannot pickle 'Environment' object
```

- Spawn method requires pickling dataset/dataloader to send to workers
- Hydra `Environment` object embedded somewhere in the chain
- Cannot proceed with spawn until pickle issue resolved

**Action Taken:** Reverted spawn method, documented issue

---

### Test 2: Fork Method with num_workers=0 (Baseline)
**Time:** 01:20:20
**Objective:** Verify basic training works without multiprocessing

**Configuration:**
- Reverted to fork method (default)
- Command: `python scripts/runners/train.py experiment=rec_baseline_v1 ++data.num_workers=0`
- Expected: Training should work (based on previous session)

**Result:** ❌ UNEXPECTED SEGFAULT
**Error:** `RuntimeError: DataLoader worker (pid 20447) is killed by signal: Segmentation fault.`

**Key Observations:**
1. **Workers spawned despite num_workers=0** 🔴 CRITICAL
   - Worker process 20447 was created
   - Should NOT happen with `num_workers=0`

2. **Crash during validation, not training**
   ```python
   File ".../training_epoch_loop.py", line 406, in on_advance_end
     self.val_loop.run()  # ← Validation phase
   ```

3. **Training setup succeeded**
   - Model loaded: PARSeq (56.4M params)
   - Datasets created successfully
   - GPU detected and initialized

**Hypotheses:**
1. **Config override not propagating** - `++data.num_workers=0` not reaching DataLoader
2. **Separate val/train configs** - Validation dataloader uses different config source
3. **Multiple DataModules** - Different DataLoader instances with different configs

**Evidence:**
```
INFO: You are using a CUDA device ('NVIDIA GeForce RTX 3090')...
┏━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┓
│ 0 │ model   │ PARSeq        │ 56.4 M │
└───┴─────────┴───────────────┴────────┘
ERROR: Unexpected segmentation fault encountered in worker.
RuntimeError: DataLoader worker (pid 20447) is killed by signal: Segmentation fault.
```

**Log Saved:** `logs/test2_fork_num_workers_0_segfault.log`

---

### Findings Summary

**Problem 1: Pickle Incompatibility**
- **Blocker:** Spawn method (required for CUDA multiprocessing)
- **Root Cause:** Hydra Environment object in dataset chain
- **Priority:** Medium (needed for proper multiprocessing)
- **Fix Required:** Remove Environment refs, use `OmegaConf.to_container()`

**Problem 2: Configuration Mystery** 🔴 **CRITICAL**
- **Blocker:** CLI overrides not working
- **Root Cause:** Unknown - config resolution issue
- **Priority:** HIGHEST (blocks all testing)
- **Fix Required:** Investigate Hydra config chain + DataModule

**Problem 3: Validation-Specific Crash**
- **Blocker:** Worker segfault in validation loop only
- **Root Cause:** Likely related to Problem 2
- **Priority:** High (after fixing Problem 2)
- **Fix Required:** Understand val vs train dataloader differences

---

### Next Investigation Steps

#### Immediate (Priority 1)
1. **Verify actual num_workers values**
   - Add logging to DataModule `train_dataloader()` and `val_dataloader()`
   - Print: `logger.info(f"Creating DataLoader with num_workers={self.num_workers}")`
   - Expected outcome: Confirm if 0 reaches DataLoader

2. **Check Hydra config resolution**
   - Add to orchestrator: `logger.info(OmegaConf.to_yaml(self.cfg.data))`
   - Expected outcome: See if CLI override appears in resolved config

#### Short-term (Priority 2)
3. **Test validation skip workaround**
   - Command: `++trainer.limit_val_batches=0`
   - Bypass validation dataloader entirely
   - Goal: Isolate if training alone works

4. **Create minimal DataLoader test**
   - Bypass Hydra entirely
   - Hardcode `num_workers=0`
   - Direct dataset instantiation

#### Medium-term (Priority 3)
5. **Fix pickle issue**
   - Search for Environment object storage
   - Convert DictConfig to plain dict
   - Re-enable spawn method

---

### Environment Info
- **Python:** 3.11.14
- **PyTorch:** 2.6.0+cu124
- **CUDA:** 12.4
- **GPU:** RTX 3090 (24GB)
- **Default MP Method:** fork
- **Shared Memory:** 8GB available

---

### References
- Previous walkthrough: `/home/vscode/.gemini/antigravity/brain/c77f3ff0-af7c-46ab-8e30-c25d4d478fab/walkthrough.md.resolved`
- Diagnostic plan: `/home/vscode/.gemini/antigravity/brain/2ce746b0-a960-4e90-a8f2-4da4fc87a964/diagnostic_plan.md`
- Failure analysis: `/home/vscode/.gemini/antigravity/brain/2ce746b0-a960-4e90-a8f2-4da4fc87a964/failure_analysis.md`
