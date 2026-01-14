# Initial Analysis

## Problem Statement
Training with `num_workers > 0` crashes at ~step 38-40 of epoch 0 with:
```
ERROR: Unexpected segmentation fault encountered in worker.
```

## Key Observations

### Environment Verification
- **CUDA**: PyTorch compiled for cu124, runtime 12.4, driver 581.80 → **Compatible** ✓
- **Shared Memory**: `/dev/shm` = 8GB → Adequate for typical workloads
- **Dataset**: 616,366 samples in LMDB at `data/processed/recognition/aihub_lmdb_validation`

### Warning Signal
```
UserWarning: Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.
```
**Location**: `decoder.py:102-107`
- `tgt_mask` = float tensor (from `generate_square_subsequent_mask`)
- `tgt_key_padding_mask` = bool tensor (from comparison)

### Timeline
1. Before path change: Training passed epoch 1
2. After path change (`data/processed/{data}` → `data/processed/recognition/{data}`): Crash at step ~38

## Priority Hypotheses

### H2: Mask Type Mismatch (HIGH)
PyTorch 2.6 introduced stricter handling. The bool/float mismatch may trigger undefined behavior in CUDA kernels during multi-worker batching.

### H3: LMDB Re-initialization Race (MEDIUM)
Workers re-open LMDB env lazily. If multiple workers hit `_init_env()` simultaneously, could cause corruption.

### H4: Shared Memory (MEDIUM)
`prefetch_factor=3` × `batch_size=8` × large images × workers may exceed shm limits.

## Recommended Tests

1. **Mask Fix Test**: Cast `tgt_key_padding_mask` to float before decoder call
2. **Worker Isolation**: Test with `num_workers=1` (single worker, still multiprocessing)
3. **LMDB Lock**: Add per-worker locking or `fork` method awareness
4. **Memory Monitor**: Run with `shm` monitoring during crash window
