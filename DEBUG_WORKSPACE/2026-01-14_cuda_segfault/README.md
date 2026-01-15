# Debug Session: CUDA Segfault in DataLoader

**Date**: 2026-01-14 (Updated: 2026-01-15)
**Status**: ✅ Resolved
**Bug ID**: BUG-20260114-001 (Segfault)
## Quick Summary
Training crashes with "Unexpected segmentation fault in worker" when `num_workers > 0`.

## Environment
| Component    | Value                      |
| ------------ | -------------------------- |
| PyTorch      | 2.6.0+cu124                |
| CUDA Runtime | 12.4                       |
| Driver       | 581.80 (CUDA 13.0 capable) |
| GPU          | RTX 3090 24GB              |
| /dev/shm     | 8GB                        |

## Hypothesis Table

| ID  | Hypothesis                            | Status            | Priority | Evidence                                                   |
| --- | ------------------------------------- | ----------------- | -------- | ---------------------------------------------------------- |
| H1  | CUDA driver/runtime mismatch          | ❌ Ruled out       | -        | Driver 581.80 is forward-compatible with CUDA 12.4 runtime |
| H2  | Mask type mismatch (attn_mask)        | ✅ ROOT CAUSE      | High     | Cast to float fixed segfault, verified with num_workers=2  |
| H3  | LMDB handle fork unsafety             | ⚠️ Partially ruled | Medium   | `__getstate__` removes env, but re-init may race           |
| H4  | Shared memory exhaustion              | ❌ Ruled out       | Medium   | Passed 100+ batches after H2 fix                           |
| H5  | Dataset path update caused corruption | ❌ Ruled out       | Low      | Path was fine, mask mismatch was the issue                 |

## Documents
- [01_initial_analysis.md](./01_initial_analysis.md) - Problem statement
- [02_investigation.md](./02_investigation.md) - Test steps
- [03_findings.md](./03_findings.md) - Results
