# Research: High-Loss Image Audit Strategy for Recognition Training

**Feature**: `001-wandb-config-logging` (follow-up extension)
**Date**: 2026-02-17

## Executive Summary

Two continuation runs from the same checkpoint/epoch range (19-39) show persistent high train-loss volatility while producing materially different validation outcomes:

- Previous run (`lr=2e-4`): final/best `val/acc ~= 0.862`, `val_loss ~= 0.243`
- Current run (`lr=5e-5`): final/best `val/acc ~= 0.828`, `val_loss ~= 0.319`

The volatility signature remains in both runs, so LR reduction alone did not remove instability. The immediate highest-value next step is selective high-loss sample auditing (not blanket image logging) to identify whether spikes are dominated by label noise, OOV symbols, or genuinely hard samples (stamps/handwriting).

## Run Comparison (Objective 1)

### Observation

1. **Generalization gap exists in both runs**
    - Previous: train/loss `0.041` vs val_loss `0.243`
    - Current: train/loss `0.036` vs val_loss `0.319`
2. **Lower LR underperformed in this horizon**
    - Same final epoch/global step, but `val/acc` dropped by ~3.45 points (`0.862 -> 0.8275`)
3. **Spiky train/loss persists across LR settings**
    - Indicates data/batch heterogeneity or difficult-token dynamics, not just optimizer step size.

### Decision

**Decision**: Treat spikes as data-visibility problem first; instrument difficult samples before further LR/scheduler changes.

**Rationale**:
- Existing random validation image logging is informative but not targeted to failure regions.
- Per-batch full logging previously caused excessive upload volume and was disabled.
- High-loss-focused sampling gives direct signal with bounded overhead.

**Alternatives considered**:
- Lower LR further: rejected as first move (already reduced with worse quality).
- Re-enable unrestricted per-batch logging: rejected (known volume issue).
- Full offline dataset audit only: useful, but slower feedback loop than in-run validation auditing.

## Existing System Review (Objective 3)

### What already exists

1. **Safe WandB logger construction** in orchestrator
    - `ocr/pipelines/orchestrator.py` manually instantiates `WandbLogger` and strips unsafe fields.
2. **Recognition image logging path** (module-based)
    - `ocr/domains/recognition/module.py::_log_validation_images`
    - gated by `train.logger.wandb.log_recognition_images`
    - currently logs sampled images from first two validation batches.
3. **Detection-specific problematic batch logger/callback**
    - `ocr/core/lightning/loggers/wandb_loggers.py` and `ocr/domains/detection/callbacks/wandb_image_logging.py`
    - tuned for box metrics (`recall`, `precision`, `hmean`), not recognition token loss.
4. **Disabled high-volume toggle already present**
    - `configs/train/logger/wandb.yaml -> per_batch_image_logging.enabled: false`

### Design implication

**Decision**: Implement high-loss auditing in recognition path, do not repurpose detection callback directly.

**Rationale**:
- Recognition has different failure semantics (sequence/token loss, CER/edit mismatch).
- Existing recognition path already has decoded predictions and GT text.
- Minimal-risk integration point: `validation_step` output + `on_validation_epoch_end` summarization.

**Alternatives considered**:
- Unify detection/recognition logger now: rejected (larger refactor, mixed domain assumptions).
- Implement as standalone callback immediately: possible, but module-level integration is lower churn and can be callbackized later.

## Recommended Feasible Approach (Objective 2)

### Decision

Add a **bounded Top-K High-Loss Audit** for recognition validation.

### Proposed behavior

1. Compute `per_sample_loss` in recognition validation step (length-normalized sequence CE/NLL).
2. Keep only epoch Top-K worst samples (e.g., `K=16`) in memory using bounded structure.
3. Log once per epoch to WandB as:
    - Image panel (`wandb.Image`) with caption: `loss`, `gt`, `pred`, `batch_idx`, optional filename
    - Table (`wandb.Table`) with structured columns for sortable diagnosis
4. Keep existing random sample logger optional and separate.

### Minimal config extension

Add under `train.logger.wandb`:

```yaml
high_loss_audit:
  enabled: false
  top_k: 16
  min_global_step: 0
  log_every_n_epochs: 1
  max_image_side: 768
  include_table: true
  include_correct_but_high_loss: false
```

### Why this is feasible now

- Reuses existing WandB + recognition image rendering path.
- Avoids flood by strict top-k and epoch-only upload.
- Directly answers the spike question with visual evidence.

## Data-Centric Action Policy

After 2-3 audited runs, bucket high-loss samples:

1. **Label issue** (clean image, incorrect GT): fix/prune labels.
2. **Hard but valid** (handwriting/stamp/noise): keep, optionally reweight.
3. **Charset/OOV** (missing symbol coverage): update tokenizer charset or normalize labels.
4. **Pipeline issue** (crop/rotation artifact): fix preprocessing/metadata.

Do not delete all high-loss samples blindly; first classify by cause.

## Open Questions Resolved (Phase 0)

- Q: Is batch-level image logging needed?
  **A**: Yes, but only selective (Top-K worst), not full batch logging.
- Q: Are spikes substantial?
  **A**: Yes, amplitude is substantial and persistent across LR settings.
- Q: Should filtering be attempted?
  **A**: Yes, but only after high-loss sample triage (label noise vs hard-valid split).

## References

- `/workspaces/docs/reports/baseline_2026-02-16.md`
- `/workspaces/docs/reports/baseline_epoch_19_39_lr5e-5_2026-02-17.md`
- `/workspaces/ocr/domains/recognition/module.py`
- `/workspaces/ocr/domains/recognition/callbacks/wandb_logging.py`
- `/workspaces/configs/train/logger/wandb.yaml`
