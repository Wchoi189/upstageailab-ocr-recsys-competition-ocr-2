# Quickstart: Bounded High-Loss Image Audit on WandB

**Feature**: `001-wandb-config-logging` (follow-up extension)
**Audience**: OCR training engineers
**Time**: ~5 minutes setup

## Goal

Inspect unstable/jagged training behavior by logging only the worst validation samples per epoch (Top-K high-loss), without re-enabling excessive per-batch image logging.

## Prerequisites

1. Keep WandB config safety default:
   - `train.logger.wandb.log_config=false`
2. Ensure WandB logger is enabled for the run.
3. Ensure a Korean-capable font is available for recognition image rendering:
   - Preferred: `fonts-nanum` package
   - Or set `OCR_WANDB_FONT_PATH` to a valid Hangul-capable `.ttf/.otf/.ttc`

## Recommended Config

In `configs/train/logger/wandb.yaml`:

```yaml
log_recognition_images: false
recognition_patch_native_view: false  # regular val image logging: keep overlay captions
high_loss_audit:
  enabled: true
  top_k: 16
  log_every_n_epochs: 1
  min_global_step: 0
  max_image_side: 768
  include_table: true
  include_correct_but_high_loss: false
   patch_native_view: true  # high-loss patches: keep native image without extra canvas
```

Optional strict rendering override:

```bash
export OCR_WANDB_FONT_PATH=/usr/share/fonts/truetype/nanum/NanumGothic.ttf
```

## Run

```bash
cd /workspaces
uv run python scripts/runners/train.py \
  experiment=parseq_flash_plateau \
  trainer.max_epochs=40 \
  train.logger.wandb.enabled=true
```

## What to Check in WandB

1. `audit/high_loss_samples` image panel
   - Confirm images are truly difficult (stamps, handwriting, low contrast) vs mislabeled.
2. `audit/high_loss_table`
   - Sort by `loss`, inspect `gt_text`, `pred_text`, filename, batch index.
3. Compare with scalar curves
   - Correlate high-loss sample content with spike windows in `train/loss` and plateaus in `val/acc`.

## Triage Policy

Use this decision table after collecting 2-3 epochs of audits:

- Clean image + wrong GT → label fix/prune candidate.
- Hard but valid image → keep; consider targeted augmentation or weighting.
- OOV/symbol mismatch → charset/tokenizer update.
- Preprocessing artifact (crop/rotation) → pipeline fix.

## Guardrails

- Keep `top_k` small (`8-16`) to avoid upload bloat.
- Do not enable legacy/full per-batch image logging for long runs.
- Log once per epoch; avoid per-step `wandb.log` loops.
- Keep `max_image_side` bounded (recommended `<=768`) to avoid oversized media panels.

## Troubleshooting (Fail-Fast)

If strict mode fails, use the exact error text to resolve quickly:

1. **Missing WandB run for recognition images**
   - Error: `Recognition image logging is enabled, but no active WandB experiment is attached to the trainer.`
   - Fix: Attach a real `WandbLogger` run or disable `log_recognition_images`.

2. **No Korean-capable font**
   - Error: `No Korean-capable font found for WandB recognition image logging.`
   - Fix: Install `fonts-nanum` (or equivalent) and/or set `OCR_WANDB_FONT_PATH`.

3. **Invalid image size config**
   - Error: `train.logger.wandb.recognition_image_max_side must be an integer or null.`
   - Fix: Set integer value (e.g., `640`, `768`) or `null`.

4. **Invalid high-loss audit config values**
   - Errors include:
     - `train.logger.wandb.high_loss_audit.top_k must be > 0.`
     - `train.logger.wandb.high_loss_audit.log_every_n_epochs must be >= 1.`
     - `train.logger.wandb.high_loss_audit.min_global_step must be >= 0.`
   - Fix: Correct invalid values in `wandb.yaml` or Hydra overrides.

## Fast Compare Recipe (for your two-run scenario)

Use same checkpoint/start epoch and run A/B with only one variable changed:

1. Baseline LR (e.g., `2e-4`) + high-loss audit on
2. Lower LR (e.g., `5e-5`) + same audit settings

Then compare not only final `val/acc`, but overlap of top-K failure sample types.

## Expected Outcome

You should be able to answer:

1. Are spikes driven by a small repeated subset of hard/noisy samples?
2. Are failures mostly data quality, charset coverage, or preprocessing defects?
3. Whether next change should be data curation, augmentation, tokenizer updates, or scheduler tuning.

## Validation Evidence

- Online strict validation run (media upload verified):
   - https://wandb.ai/ocr-team2/receipt-text-recognition-ocr-project/runs/v4y4yrbt
- Local integration smoke with high-loss audit enabled (fast-dev):
   - Command: `uv run python scripts/runners/train.py experiment=parseq_flash_plateau_images +trainer.fast_dev_run=true train.logger.wandb.settings.offline=true train.logger.wandb.log_recognition_images=false +train.logger.wandb.high_loss_audit.enabled=true`
   - Result: Completed successfully; `val/acc` and `val/cer` emitted; high-loss audit path executed without runtime exceptions.
