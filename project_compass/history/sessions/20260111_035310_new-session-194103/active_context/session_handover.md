# Session Handover: Recognition Pipeline Fixes

## Summary
The "Recognition Pipeline Initialization" and debugging session concluded successfully. The persistent training crashes were resolved by systematically fixing configuration overrides, model architecture bugs, and runtime validation logic.

## Key Achievements
- **Training Stability**: PARSeq model now trains end-to-end (verified with 3 epochs + test set).
- **Configuration Hygiene**: Resolved conflicts between detection (DBNet) and recognition (PARSeq) overrides by enforcing strict domain separation in `configs/domain/recognition.yaml`.
- **Runtime Fixes**:
    - **CUDA Device-side Assert**: Fixed by injecting `vocab_size` (1027) into `PARSeqDecoder`.
    - **Feature Shape Mismatch**: Fixed `PARSeq.forward` to handle 3D `(B, S, C)` ViT features.
    - **DataLoader Crashes**: Identified `num_workers=4` as unstable; default reduced to 2 for safety.
    - **Deterministic Error**: Disabled strict determinism (`upsample_bicubic2d` is non-deterministic).

## Current State
- **Branch/Commit**: Testing on `master` (or current workspace state).
- **Configs**: `configs/trainer/default.yaml` has `deterministic: False`. `configs/base.yaml` has `batch_size: 32`.
- **Artifacts**: `walkthrough.md` details the full verification process.

## Next Steps (New Session)
1.  **Full-Scale Training**: Launch the long training run on RTX 3090.
    ```bash
    uv run python runners/train.py domain=recognition trainer=hardware_rtx3090_24gb_i5_16core exp_name="rec_parseq_v1_run1"
    ```
2.  **Experimentation**: Implement augmentations or architecture tweaks as per roadmap.
3.  **Metrics**: Re-enable detection-style metrics (CLEval) or migrate to pure text recognition metrics (already logging accuracy/CER in WandB).

## Known Issues
- `num_workers > 2` may cause shared memory crashes on this instance.
- Initial imports (scipy/transformers) are slow.

## Artifacts Created
- `walkthrough.md` (Verification Log)
- `task.md` (Checklist)
