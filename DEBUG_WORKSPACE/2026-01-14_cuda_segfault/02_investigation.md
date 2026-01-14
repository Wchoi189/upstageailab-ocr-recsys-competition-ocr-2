# Investigation Log

## Test Plan

### Test 1: Mask Type Fix
**Hypothesis**: H2 - Bool/float mask mismatch causes CUDA instability
**Change**: Cast `tgt_key_padding_mask` to float in `decoder.py`
```python
# Line 105-107
tgt_key_padding_mask = (targets == self.pad_token_id).float()  # Cast to float
```
**Command**:
```bash
uv run python runners/train.py domain=recognition \
  trainer=hardware_rtx3090_24gb_i5_16core \
  exp_name="test_mask_fix" +trainer.max_epochs=1
```

### Test 2: Single Worker Isolation
**Hypothesis**: Multi-worker interaction is the trigger
**Change**: Force `num_workers=1` (still uses multiprocessing, but isolates)
**Command**:
```bash
uv run python runners/train.py domain=recognition \
  trainer=hardware_rtx3090_24gb_i5_16core \
  exp_name="test_single_worker" +trainer.max_epochs=1 \
  +data.dataloaders.train_dataloader.num_workers=1
```

### Test 3: LMDB Worker Lock
**Hypothesis**: H3 - LMDB env re-init races
**Change**: Add thread lock to `_init_env()` or use `multiprocessing.Lock`

### Test 4: Memory Monitoring
**Command** (run in parallel terminal):
```bash
watch -n 1 "df -h /dev/shm; nvidia-smi --query-gpu=memory.used --format=csv"
```

---

## Results Table

| Test             | Config                  | Duration | Result      | Notes                                             |
| ---------------- | ----------------------- | -------- | ----------- | ------------------------------------------------- |
| Baseline         | num_workers=4           | ~4s      | ❌ SEGFAULT  | Crashes at step 38-40                             |
| test_h2_mask_fix | num_workers=0 (default) | 43m      | ✅ Completed | Epoch 0 finished; import error at checkpoint save |
| -                | -                       | -        | -           | Next: test with num_workers=2                     |
