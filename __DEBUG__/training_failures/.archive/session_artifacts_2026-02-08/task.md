# Debug ParSeq Multi-Worker DataLoader CUDA Failure

## Status
- [x] Phase 1: Verify Configuration and Baseline ✅
- [x] Phase 2: Isolate Multi-Worker Failure Mechanism ✅
- [x] Phase 3: Test Solutions for Multi-Worker Support ✅
- [x] Phase 4: Implement and Validate Fix ✅

---

## Phase 1: Verify Configuration and Baseline ✅
- [x] Check current global configuration (num_workers=2, pin_memory=false, persistent=true)
- [x] Capture environment details (PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3090)
- [x] Test basic CUDA operations (ALL PASS ✅)
- [x] Test simplified DataLoader with workers (ALL CONFIGS PASS ✅)

## Phase 2: Isolate Multi-Worker Failure Mechanism ✅
- [x] Confirmed simplified tests work - issue is pipeline-specific
- [x] Inspected LMDB dataset - has lazy init (good) but opens in __init__ (minor concern)
- [x] Identified missing components: no worker_init_fn, no spawn start method
- [x] Root cause: Fork mode inheriting CUDA context from main process

### Key Finding
**Issue is NOT general CUDA+DataLoader** - isolated tests all pass. Problem is multiprocessing fork + CUDA context inheritance in recognition pipeline.

## Phase 3: Test Solutions ✅
- [x] Test spawn start method (highest priority - quick win) - **WORKS!**
- [x] Verify fix with num_workers=1 - **PASS ✅**
- [x] Verify fix with num_workers=2 - **PASS ✅**

## Phase 4: Implementation ✅
- [x] Apply spawn start method to scripts/runners/train.py
- [x] Update global config to enable workers and pin_memory
- [x] Update documentation and comments
- [x] Create walkthrough documenting solution
- [x] All tests passing, fix validated

---

## Solution Summary

**Root Cause:** Fork-based multiprocessing inheriting CUDA context from main process

**Fix:** Enable spawn start method in `scripts/runners/train.py`
- Creates fresh Python processes without state inheritance
- Each worker initializes clean CUDA context
- Compatible with pin_memory=true and persistent_workers=true

**Verification:**
- ✅ num_workers=0: Works (backward compatibility)
- ✅ num_workers=1: Works (single worker)
- ✅ num_workers=2: Works (multi-worker)
- ✅ No CUDA errors, no terminal freezing

**Files Modified:**
1. `scripts/runners/train.py` - Enabled spawn method
2. `configs/global/default.yaml` - Updated worker/pin_memory defaults
