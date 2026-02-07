# Quick Reference: GPU Training Configuration

**Updated:** 2026-02-08
**Status:** ✅ Working

---

## TL;DR - Working Configuration

```yaml
# In configs/global/default.yaml
dataloaders:
  train_dataloader:
    num_workers: 0
    pin_memory: false
  val_dataloader:
    num_workers: 0
    pin_memory: false
```

**Why:** Docker CUDA context + multiprocessing + pin_memory = crashes

---

## Quick Start

### Test (15 batches)
```bash
uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_baseline_official \
  trainer.limit_train_batches=15
```

### Full Training
```bash
uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_baseline_official \
  trainer.max_epochs=20
```

---

## What Changed

| Setting | Old | New | Why |
|---------|-----|-----|-----|
| num_workers | 4 | 0 | Docker CUDA stability |
| pin_memory | true | false | IPC failures |

---

## Trade-offs

- **Lost:** ~2-3x data loading speed
- **Gained:** Stable GPU training
- **Worth it:** Yes - GPU compute dominates

---

## Don't Change
- ✅ num_workers=0
- ✅ pin_memory=false
- ✅ Official PARSeq adapter

## Safe to Change
- Batch size
- Learning rate
- Epochs

---

**Full docs:** [`session_handover_2026-02-08_0252.md`](./session_handover_2026-02-08_0252.md)
