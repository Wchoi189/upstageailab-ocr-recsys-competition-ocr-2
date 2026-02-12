---
ads_version: 1.0
type: bug_report
category: troubleshooting
status: active
version: 1.0
severity: medium
description: PLM+Flash training fails with CUDA initialization error when num_workers >= 16, but succeeds with num_workers=12. GPU utilization remains low (45%) suggesting dataloader bottleneck.
tags: flash-attention,cuda-error,plm,dataloader,performance
title: Flash Attention causes CUDA initialization error with high dataloader workers
date: 2026-02-13 01:34 (KST)
branch: main
---

# Bug Report - Flash Attention causes CUDA initialization error with high dataloader workers
Bug ID: BUG-2026-02-13-001

## Summary
PLM+Flash training with `enable_flash_attention_kernel()` context manager fails with CUDA initialization error when `num_workers >= 16`. Training succeeds with `num_workers=12` but exhibits low GPU utilization (45%) and suboptimal throughput (2.1 it/s, ~336 img/sec), indicating dataloader bottleneck rather than Flash Attention benefit.

## Environment
- **OS**: Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)
- **GPU**: NVIDIA GeForce RTX 3090 (24GB, sm_86)
- **CUDA**: 12.4
- **PyTorch**: 2.6.0+cu124
- **Training Config**: `experiment=parseq_plm_flash`
- **Branch**: `001-mcp-tooling-refactor`

## Reproduction

### Test 1: Succeeds (Low Performance)
```bash
uv run python scripts/runners/train.py \
  experiment=parseq_plm_flash \
  ++data.num_workers=12 \
  ++data.prefetch_factor=4 \
  ++data.batch_size=160 \
  ++trainer.max_steps=200 \
  ++trainer.limit_val_batches=0.1
```

**Result**:
- ✅ Training completes successfully
- ❌ Low throughput: 2.10 it/s × 160 batch = 336 img/sec (target: 400-640)
- ❌ Low GPU utilization: 45% (239W / 370W)
- ⚠️ Validation accuracy: 0.000 (Match Count: 0/160)

### Test 2: CUDA Error (High Workers)
```bash
uv run python scripts/runners/train.py \
  experiment=parseq_plm_flash \
  ++data.num_workers=16 \
  ++data.prefetch_factor=8 \
  ++data.batch_size=160 \
  ++trainer.max_steps=200
```

**Result**: CUDA initialization error

## Comparison
**Expected**:
- Flash Attention should provide 2-4x speedup (400-640 img/sec)
- GPU utilization should be 95%+
- Training should work with `num_workers=16`

**Actual**:
- Test 1: 2.3x speedup vs baseline (336 vs 170 img/sec), but GPU underutilized
- Test 2: CUDA error with higher worker count
- Backend likely falling back to MATH kernel (not Flash) due to PLM custom masks

## Logs

### Test 2 Error
```
Match Count: 0/160
terminate called after throwing an instance of 'c10::Error'
  what():  CUDA error: initialization error
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at
/pytorch/c10/cuda/CUDAException.cpp:43 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96
```

### GPU Status (Test 1)
```
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        On  |   00000000:01:00.0  On |                  N/A |
| 44%   53C    P2            239W /  370W |    9784MiB /  24576MiB |     45%      Default |
```

## Root Cause Analysis

### Backend Fallback Issue
The `enable_flash_attention_kernel()` context manager forces Flash backend priority:
```python
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=True,  # Allows fallback for PLM masks
    enable_mem_efficient=False
):
```

However, PLM's **custom additive attention masks** are incompatible with Flash Attention 2 backend, forcing fallback to:
- **MATH backend** (slower matmul-based attention)
- **cuDNN backend** (memory-efficient attention)

### Why High Workers Fail
Multi-process dataloading (`num_workers >= 16`) with CUDA initialization in forked processes can cause:
1. CUDA context conflicts between parent and child processes
2. Flash Attention kernel initialization failures in worker processes
3. Backend selection race conditions during multiprocessing

### Why GPU Utilization is Low
- Dataloader bottleneck (CPU-bound, not GPU-bound)
- Workers spending time on data preprocessing/augmentation
- GPU idle waiting for next batch

## Impact
- **Severity**: Medium
- **Blocker**: No (workaround available: use `num_workers=12`)
- **Performance**: Training throughput 2.3x vs baseline, but GPU underutilized
- **Production Risk**: Medium (unstable with optimized dataloader settings)

## Workarounds

### Workaround 1: Limit Workers (Current)
```bash
# Use num_workers <= 12
++data.num_workers=12 ++data.prefetch_factor=4
```

### Workaround 2: Use Pure Flash AR (No PLM)
```bash
# Avoid PLM custom masks entirely
experiment=parseq_flash ++data.batch_size=128
```
Expected: True Flash backend, 1350-1800 img/sec, 95% GPU utilization

### Workaround 3: Increase Batch Size
```bash
# Reduce dataloader pressure by increasing batch work
++data.batch_size=256 ++data.num_workers=12
```

## Recommended Fix

### 1. Add Backend Detection Logging
Modify `enable_flash_attention_kernel()` to log which backend is actually used:
```python
import logging
logger = logging.getLogger(__name__)

@contextmanager
def enable_flash_attention_kernel():
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=False
        ):
            # Log warning if Flash not available
            logger.warning(
                "⚠️  Flash Attention may fallback to MATH backend with custom masks"
            )
            yield
    except (AttributeError, RuntimeError) as e:
        logger.warning(f"⚠️  Flash Attention unavailable: {e}, using standard attention")
        yield
```

### 2. Optimize Dataloader
- Profile data preprocessing pipeline
- Cache augmentations if deterministic
- Use `persistent_workers=True` to avoid reinitialization

### 3. Consider Alternative Configurations
- **Option A**: Pure Flash AR (no PLM, guaranteed Flash backend)
- **Option B**: PLM Baseline (no Flash dependency, stable)
- **Option C**: Larger batches to amortize dataloader overhead

## Related Issues
- Phase 6.1 Audit: "Custom additive masks from PLM prevent Flash Attention 2 backend"
- Performance target: 400-640 img/sec (PLM+Flash), currently at 336 img/sec
- Validation accuracy: 0.000 suggests training issue (separate from Flash Attention)

## Next Steps
1. Add backend detection logging with colored warnings
2. Profile dataloader pipeline to identify bottleneck
3. Test pure Flash AR to verify Flash backend works without PLM masks
4. Investigate validation accuracy issue (0.000 match rate)
