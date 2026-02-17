# Contracts: WandB Config Safety + Recognition High-Loss Audit

**Feature**: `001-wandb-config-logging` (follow-up extension)
**Date**: 2026-02-17

## Scope

This contract governs:

1. Existing WandB config serialization safety (`log_config: false` by default)
2. New recognition high-loss audit behavior with bounded image/table logging

## Configuration Contract

### Input YAML

**File**: `/workspaces/configs/train/logger/wandb.yaml`

```yaml
type: object
properties:
  _target_:
    type: string
    const: lightning.pytorch.loggers.WandbLogger
  log_config:
    type: boolean
    default: false
  log_recognition_images:
    type: boolean
    default: false
  high_loss_audit:
    type: object
    properties:
      enabled: {type: boolean, default: false}
      top_k: {type: integer, minimum: 1, default: 16}
      log_every_n_epochs: {type: integer, minimum: 1, default: 1}
      min_global_step: {type: integer, minimum: 0, default: 0}
      max_image_side: {type: integer, minimum: 128, default: 768}
      include_table: {type: boolean, default: true}
      include_correct_but_high_loss: {type: boolean, default: false}
```

### Output Behavior

#### A) Config serialization safety

- `log_config: false` (default) must always initialize WandB logger safely.
- `log_config: true` is user override and may fail for Hydra configs with `_target_` fields.

#### B) High-loss audit behavior

When `high_loss_audit.enabled: true`:

1. System computes/consumes per-sample validation loss.
2. System retains only epoch top-K worst samples.
3. System logs once per configured epoch cadence:
   - image panel key: `audit/high_loss_samples`
   - optional table key: `audit/high_loss_table`

When disabled, no additional image/table logging occurs from this feature.

## Module Contract

### Producer contract (`RecognitionPLModule.validation_step`)

Must expose enough data for high-loss audit:

- `per_sample_loss` (`Tensor[B]` or equivalent list)
- decoded prediction text
- ground truth text
- images
- optional filename metadata

### Consumer contract (epoch-end logger)

Must:

- ignore NaN/inf losses
- apply top-K cap strictly
- avoid per-batch `wandb.log` flood
- gracefully no-op if WandB logger/run unavailable

## Non-Functional Contract

- Upload volume bounded by `top_k`
- Runtime overhead bounded by small in-memory top-K tracking
- No new external dependencies
- No changes to detection callback behavior

## Backward Compatibility

Guaranteed:

- Existing runs with `log_recognition_images` unchanged
- Existing config logging constraint unchanged
- Existing run naming and metric logging unchanged

Not guaranteed:

- Historical comparability of image panels if panel key names are changed by users

## Validation Contract

Pass criteria for this extension:

1. With `high_loss_audit.enabled=false`, no new image/table keys appear.
2. With `enabled=true`, image count per epoch never exceeds `top_k`.
3. Logged samples correspond to highest per-sample losses in that epoch.
4. Training remains stable (no serialization crashes or callback exceptions).

## Explicit Rejections

- No unrestricted per-batch image logging (known excessive volume).
- No auto-pruning/deleting samples from dataset during training.
- No forced unification of detection and recognition image logging in this iteration.
