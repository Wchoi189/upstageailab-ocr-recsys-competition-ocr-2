---
type: bug_report
id: 20260208_dataloader-cuda-multiworker
title: DataLoader Multi-Worker CUDA Initialization Failure
date: 2026-02-08 23:33 (KST)
category: troubleshooting
status: completed
version: 1.0
severity: high
---

## Description
Training crashed with CUDA initialization errors when using DataLoader `num_workers > 0`. Error occurred during tensor cleanup in `TensorImpl::~TensorImpl()`. Terminal froze with repetitive error log spam.

## Impact
- **Severity**: High
- **Effect**: Training impossible with multi-worker data loading. GPU underutilized.
- **Error**: `CUDA error: initialization error` from `c10_cuda_check_implementation`

## Reproduction
```bash
uv run python scripts/runners/train.py \
  domain=recognition experiment=rec_baseline_official \
  dataloaders.train_dataloader.num_workers=2
```
**Result**: Crash during first training batch

## Root Cause
Fork-based multiprocessing inherits CUDA context from main process. Workers cannot initialize their own CUDA contexts, causing context mismatch on tensor operations.

**Environment**: Docker + GPU passthrough, PyTorch 2.6.0+cu124, Linux default fork method

## Resolution
Enabled spawn multiprocessing start method in `scripts/runners/train.py`:
```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

Updated `configs/global/default.yaml`:
- `num_workers: 2` (now works)
- `pin_memory: true` (safe with spawn)

## Verification
| num_workers | Before | After |
|-------------|--------|-------|
| 0 | ✅ | ✅ |
| 1 | ❌ | ✅ |
| 2 | ❌ | ✅ |

Logs: `__DEBUG__/training_failures/logs/test_spawn_workers*.log`

## References
- **Spec**: `AgentQMS/specs/tier2-framework/core-infra.spec.md` (line 131-157)
- **Walkthrough**: `brain/4e7f4c2a-5a1b-4830-a69c-3e2869d06e9f/walkthrough_multiworker_cuda_fix.md`
