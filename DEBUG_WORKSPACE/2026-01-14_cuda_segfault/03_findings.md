# Findings

*Updated 2026-01-15 03:30*

## Summary

**Status**: 🟡 PARTIAL ROOT CAUSE IDENTIFIED

The segfault is a multi-factor issue:

1. **H2 Mask Fix**: ✅ Applied and necessary (removed PyTorch deprecation warning)
2. **LMDB + DataLoader**: ✅ NOT the cause (minimal test passes with all configs)
3. **Root Cause**: Likely in Lightning's training loop interaction with validation

## Key Evidence

### Minimal Test Results

```
✅ LMDB + DataLoader (num_workers=0, pin_memory=false) → PASSED
✅ LMDB + DataLoader (num_workers=0, pin_memory=true) → PASSED
✅ LMDB + DataLoader (num_workers=2, pin_memory=false) → PASSED
✅ LMDB + DataLoader (num_workers=2, pin_memory=true) → PASSED
```

This proves the data loading pipeline is NOT the source of the segfault.

### Crash Pattern

| Test                        | Crash Point         | Notes                      |
| --------------------------- | ------------------- | -------------------------- |
| Full training               | Validation step 6-8 | Consistent crash location  |
| Minimal DataLoader test     | -                   | No crash                   |
| Detection domain (previous) | -                   | Works with multiprocessing |

## Confirmed Facts

1. **Driver Compatibility**: ✅ CUDA 13.0 driver forward-compatible with CUDA 12.4 runtime
2. **PyTorch Build**: ✅ torch 2.6.0+cu124 correctly compiled
3. **Shared Memory**: ✅ 8GB available at /dev/shm
4. **H2 Mask Fix Applied**: ✅ Cast `tgt_key_padding_mask` to float in `decoder.py:105`
5. **LMDB Fork Safety**: ✅ Confirmed working in minimal test

## Working Hypothesis

The crash occurs specifically when:
1. Training with Lightning + Hydra + WandB active
2. Transition from training to validation phase
3. With recognition-domain model (PARSeq)
4. Around step 6-8 of validation

Possible causes:
- Memory pressure during GPU context switching
- WandB logger interaction despite `wandb=false`
- Lightning's combined_loader behavior
- Container/devcontainer IPC limits

## Recommended Next Steps

1. **Test with `trainer=cpu`** - isolate GPU kernel issues
2. **Test without callbacks** - simpler training loop
3. **Check WandB background processes** - even with `wandb=false`, API key detection runs
4. **Try PyTorch 2.5** - downgrade to check for 2.6-specific regression

## Resolution

**Resolved**. See detailed report: [2026-01-14_segfault_resolution.md](../../docs/artifacts/bug_reports/2026-01-14_segfault_resolution.md).

The issue was fixed by enforcing `num_workers=0`, `pin_memory=False`, and disabling `wandb` image logging to bypass environment-specific IPC failures.
