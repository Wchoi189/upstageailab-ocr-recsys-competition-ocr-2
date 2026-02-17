# Data Model: Recognition High-Loss Audit (WandB)

**Feature**: `001-wandb-config-logging` (follow-up extension)
**Date**: 2026-02-17

## Overview

This extension adds bounded, per-epoch high-loss sample tracking for recognition validation. It introduces lightweight runtime entities and a config block; no DB or persistent schema migration is required.

## Entities

### 1) `HighLossAuditConfig`

**Location**: `train.logger.wandb.high_loss_audit`

**Fields**:

| Field | Type | Default | Notes |
|---|---|---:|---|
| `enabled` | bool | `false` | Master switch |
| `top_k` | int | `16` | Max samples logged per epoch |
| `log_every_n_epochs` | int | `1` | Logging cadence |
| `min_global_step` | int | `0` | Warmup before logging |
| `max_image_side` | int | `768` | Optional image resize cap |
| `include_table` | bool | `true` | Log structured `wandb.Table` |
| `include_correct_but_high_loss` | bool | `false` | Exclude numerically high-loss but exact-match predictions by default |

### 2) `HighLossSample`

Ephemeral record for one validation sample.

| Field | Type | Description |
|---|---|---|
| `epoch` | int | Validation epoch index |
| `global_step` | int | Trainer step for traceability |
| `batch_idx` | int | Source validation batch |
| `sample_idx` | int | Index inside batch |
| `loss` | float | Per-sample normalized loss |
| `gt_text` | str | Ground-truth text |
| `pred_text` | str | Decoded prediction |
| `filename` | str\|None | Optional source filename |
| `image_ref` | tensor/PIL | Image payload for visualization |

### 3) `HighLossEpochBuffer`

In-memory bounded container used during validation epoch.

| Field | Type | Description |
|---|---|---|
| `capacity` | int | Mirrors `top_k` |
| `items` | list[`HighLossSample`] | Maintains worst-K by `loss` |

## Relationships

1. `HighLossAuditConfig.enabled == true` activates sample collection in recognition validation path.
2. Each validation batch emits candidate `HighLossSample` objects using per-sample loss.
3. `HighLossEpochBuffer` retains only top-K worst samples.
4. Epoch end serializes retained samples to WandB image panel/table.

## Validation Rules

- `top_k > 0`
- `log_every_n_epochs >= 1`
- `min_global_step >= 0`
- `max_image_side >= 128` when provided
- `loss` must be finite (`not NaN/inf`) before candidate insertion

## State Transitions

1. **Epoch start**: clear `HighLossEpochBuffer`
2. **Validation batch end**: update buffer with high-loss candidates
3. **Epoch end**: emit WandB logs, then clear buffer

## Compatibility Notes

- Existing `log_config: false` safety remains unchanged.
- Existing random `log_recognition_images` behavior remains available.
- Detection-specific problematic batch logger is intentionally separate.

## Non-Goals

- No persistent storage of full per-sample losses across all epochs.
- No automatic pruning/deletion of samples during training.
- No detection/recognition logger unification in this iteration.
