# Historical Bug Analysis - Segfault Resolution

## Date: 2026-02-07
## Location: __DEBUG__/training_failures/

---

## Critical Discovery

User pointed to historical bug reports documenting **THE EXACT SAME ISSUE** we encountered.

### Bug ID: BUG-20260114-001
**Status:** Resolved (2026-01-15)
**Reports:**
- https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/blob/main/docs/artifacts/bug_reports/2026-01-15_1200_bug_report_segfault-resolution.md
- https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/blob/main/docs/artifacts/bug_reports/2026-01-14_2153_bug_001_segfault.md

---

## Identical Symptoms

| Our Tests (2026-02-07) | Historical Bug (2026-01-14) |
|------------------------|----------------------------|
| `RuntimeError: DataLoader worker (pid 20447) exited` | `RuntimeError: DataLoader worker (pid 739) exited` |
| Crash during validation loop | Crash during validation loop |
| "Unexpected segmentation fault in worker" | "Unexpected segmentation fault in worker" |
| Using `num_workers > 0` | Using `num_workers=4` |

**EXACT MATCH** ✅

---

## Root Causes Identified in Historical Reports

### Issue 1: pin_memory=True (Primary Trigger)
**Problem:**
- `pin_memory=True` spawns background `_pin_memory_loop` thread
- IPC mechanism for transferring pinned buffers between processes fails
- Function `rebuild_storage_fd` causes segfault
- Occurs even with `num_workers=0` in certain conditions

**Current Config Status:**
```yaml
# configs/global/default.yaml
dataloaders:
  train_dataloader:
    pin_memory: true  # ← PROBLEM!
  val_dataloader:
    pin_memory: true  # ← PROBLEM!
```

### Issue 2: Mask Type Mismatch (Secondary, Already Fixed)
**Problem:**
- `tgt_key_padding_mask` was `bool` while `tgt_mask` was `float`
- PyTorch 2.6 stricter SDPA kernel handling caused undefined behavior

**Current Code Status:**
```python
# ocr/domains/recognition/models/decoder.py:112-113
tgt_key_padding_mask = torch.zeros_like(targets, dtype=tgt_mask.dtype)  # ✅ FIXED
tgt_key_padding_mask.masked_fill_(targets == self.pad_token_id, float("-inf"))
```

**Verified:** The mask type fix IS already applied in the codebase.

---

## Why Our CLI Overrides Failed

The configuration hierarchy means:
```
Global Default (pin_memory: true)
  ↓
Hardware Config (rtx3090.yaml - pin_memory: true)
  ↓
Experiment Config (rec_baseline_v1.yaml - no dataloader settings)
  ↓
CLI Override (++data.num_workers=0) ← Only affects top-level, not dataloaders!
```

**The Problem:**
- `++data.num_workers=0` sets `config.data.num_workers = 0`
- But DataModule reads from `config.dataloaders.train_dataloader.num_workers`
- These are DIFFERENT paths in the Hydra config tree!
- `pin_memory: true` from global defaults was NEVER overridden

---

## Historical Solution (Verified Working)

Created `configs/trainer/debug_safe.yaml`:

```yaml
data:
  num_workers: 0

dataloaders:
  train_dataloader:
    num_workers: 0
    pin_memory: false    # ← Critical!
    persistent_workers: false
  val_dataloader:
    num_workers: 0
    pin_memory: false    # ← Critical!
    persistent_workers: false
```

**Verification Command (from bug report):**
```bash
python scripts/runners/train.py \
  experiment=rec_baseline_v1 \
  trainer=debug_safe
```

---

## Test Plan

### Test 1: Apply Historical Safe Config ⭐ HIGHEST PRIORITY

**Command:**
```bash
python scripts/runners/train.py \
  experiment=rec_baseline_v1 \
  trainer=debug_safe
```

**Expected Result:**
- ✅ Training completes without segfault
- ✅ Validation runs successfully
- ✅ Loss decreases (model learns)

**Success Criteria:**
Same as historical verification run `verify_fix_safe_v4` - complete training cycle without errors.

---

## Files Changed

| File | Change | Status |
|------|--------|--------|
| `configs/trainer/debug_safe.yaml` | Created safe config override | ✅ Created |
| `ocr/domains/recognition/models/decoder.py` | Mask type fix (lines 112-113) | ✅ Already exists |
| `scripts/runners/train.py` | Multiprocessing spawn attempt | ❌ Reverted (not needed) |

---

## Key Learnings

1. **Historical bugs are goldmines** - Exact same issue, documented solution
2. **Config hierarchy matters** - CLI overrides don't cascade to nested keys
3. **pin_memory is dangerous** - Not just num_workers, also background threads
4. **Both fixes needed**:
   - Code fix: Mask type casting (already done)
   - Config fix: disable pin_memory + multiprocessing (now done)

---

## Next Action

Run Test 1 with the safe configuration to verify training works.
