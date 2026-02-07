# Historical Debug Session Archive

**Source:** https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/tree/main/__DEBUG__/2026-01-14_cuda_segfault
**Archived:** 2026-02-07
**Reason:** Original files lost, preserving for reference

---

## Contents Preserved

1. **README.md** - Overview and hypothesis table
2. **01_initial_analysis.md** - Problem statement and key observations
3. **02_investigation.md** - Test plan and results
4. **03_findings.md** - Final resolution summary

See individual files in this directory for full content.

---

## Quick Reference - Key Findings

### Root Causes Identified

**Primary: Mask Type Mismatch**
- `tgt_key_padding_mask` was bool, `tgt_mask` was float
- PyTorch 2.6+ stricter SDPA kernel handling
- **Fix:** Cast to float: `(targets == self.pad_token_id).float()`

**Secondary: LMDB Fork Unsafety**
- LMDB environment handle inherited by worker processes
- Fork method (default Linux) caused corruption
- **Fix:** Lazy initialization in workers only

**Tertiary: pin_memory + IPC Failures**
- `pin_memory=True` creates `_pin_memory_loop` thread
- IPC mechanism `rebuild_storage_fd` fails in some environments
- **Fix:** Set `pin_memory=False`

### Solution Stack

```yaml
# Complete safe configuration
dataloaders:
  train_dataloader:
    num_workers: 0           # No multiprocessing
    pin_memory: false        # No background thread
    persistent_workers: false

# Code fix in decoder.py:105
tgt_key_padding_mask = (targets == self.pad_token_id).float()  # Not .bool()
```

---

## Test Results from Original Session

| Test | Config | Result | Notes |
|------|--------|--------|-------|
| Baseline | num_workers=4 | ❌ SEGFAULT | Crashed at step 38-40 |
| Mask fix | num_workers=2 | ✅ PASSED | 100+ batches stable |
| Safe config | num_workers=0, pin_memory=false | ✅ PASSED | Full epoch completed |

---

## Relevance to Current Session (2026-02-07)

Our investigation **independently rediscovered** the same issues:
1. ✅ Mask fix already applied in current codebase
2. ✅ pin_memory=false solution matches our safe config
3. ✅ num_workers=0 workaround identical

The historical documentation **validated** our diagnostic approach and confirmed the solution before we implemented it.

---

## Links to Preserved Files

- [01_initial_analysis.md](./01_initial_analysis.md)
- [02_investigation.md](./02_investigation.md)
- [03_findings.md](./03_findings.md)
- [README.md](./README.md)
