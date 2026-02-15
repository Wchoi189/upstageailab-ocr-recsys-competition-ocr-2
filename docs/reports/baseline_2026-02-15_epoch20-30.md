# Performance Baseline Report

**Generated:** 2026-02-15 16:06:17
**WandB Run:** [wchoi189_bs160_SCORE_PLACEHOLDER](https://wandb.ai/runs/ykjt4lpg)
**Run ID:** `ykjt4lpg`
**Status:** finished

---

## Recognition Metrics Summary

**Note:** Performance profiler metrics were not logged; report focuses on recognition quality trends.
**Trend Source:** `local_output_log_regex`

| Metric | Best | Final | Delta |
|--------|------|-------|-------|
| **val/acc** | 0.8440 | 0.8090 | -0.0350 |
| **val/cer** | 0.0960 | 0.1190 | +0.0230 |
| **val_loss** | 0.3911 | 0.3911 | +0.0000 |

| Training Details | Value |
|------------------|-------|
| **Training Loss (final)** | 0.2441 |
| **Validation Loss (final)** | 0.3911 |
| **Validation Accuracy (final)** | 0.8093 |
| **Validation CER (final)** | 0.1192 |
| **Epoch** | 29 |
| **Global Step** | 130049 |
| **History Points (val/acc)** | 125 |

## Best/Final Snapshot Hints

- **Best val/acc point:** epoch 19/29, step 3000/8370
- **Final val/acc point:** epoch 29/29, step 8369/8370

## Regex Parsing Diagnostics

| Signal | Value |
|--------|-------|
| **Raw metric lines found** | 18412 |
| **Failed metric parses** | 0 |
| **Best debug match ratio** | 0.074 |
| **Final debug match ratio** | 0.070 |

## Run Insights (Learning Guide)

This run improved early, then regressed late. That usually means optimization overshot the best region rather than the model failing to learn.

### Illustrated Interpretation

- **Best quality reached:** val/acc 0.8440, val/cer 0.0960
- **End of run:** val/acc 0.8090, val/cer 0.1190
- **Regression size:** Δacc -0.0350, Δcer +0.0230

Simple mental model:
- Training = searching for a valley in error landscape.
- Best checkpoint = lowest spot reached so far.
- Late regression = optimizer steps moved away from that spot.

### Phase Trend

- **Early mean val/acc:** 0.7829
- **Middle mean val/acc:** 0.7863
- **Late mean val/acc:** 0.7855

## Hypothesis for Latest Regression

Most plausible explanation is late-stage optimization instability during continuation from a strong checkpoint.

- The model likely reached a good local optimum early in resumed training.
- Continued updates (and scheduler behavior) moved parameters away from that optimum.
- Validation noise from frequent in-epoch checks amplifies apparent fluctuations.

## Training vs Validation Comparison

- **Note:** Performance profiling not enabled - cannot compare training vs validation timing.

## Identified Issues

### 1. Validation regression after peak (HIGH)

Best val/acc (0.8440) fell to final (0.8090), drop=0.0350

### 2. Character error rate worsened (MEDIUM)

Best val/cer (0.0960) increased to final (0.1190), delta=0.0230

## Next Run Checklist (Auto-Gated)

### Pre-run Gates

| Gate | Status | Evidence | Action |
|------|--------|----------|--------|
| **Resume LR conservative (<=2e-4)** | ❌ FAIL | Δacc=-0.0350 from best to final | Lower LR by 5-10x for continuation runs. |
| **Validation cadence stable** | ❌ FAIL | Current run shows regression after interim peaks | Use `trainer.val_check_interval=1.0` for cleaner epoch-level signal. |

### In-run Gates

| Gate | Status | Evidence | Action |
|------|--------|----------|--------|
| **No significant accuracy backslide** | ❌ FAIL | best=0.8440, final=0.8090 | Early stop if `val/acc` drops >0.02 from run-best. |
| **CER remains near best** | ❌ FAIL | best=0.0960, final=0.1190 | Reduce LR / halt when CER rises persistently. |

### Post-run Gates

| Gate | Status | Evidence | Action |
|------|--------|----------|--------|
| **Inference checkpoint selection** | ❌ FAIL | Final underperformed best by 0.0350 acc | Publish best-acc checkpoint, not final-epoch checkpoint. |
| **Continue training decision** | ❌ FAIL | Regression indicates optimization instability | Continue only with reduced LR + conservative scheduler. |

## Recommendations

1. **Select Best Checkpoint for Inference**: Use best `val/acc` checkpoint instead of final epoch checkpoint.
2. **Lower Resume LR for Fine-tuning**: For continuation runs, reduce LR by 5-10x to prevent post-resume regression.
3. **Stabilize Validation Signal**: Validate at epoch end (`trainer.val_check_interval=1.0`) for clearer epoch-level comparisons.
4. **Track LR Curves in W&B**: Ensure one LR key is logged continuously to correlate LR spikes with quality drops.
5. **Enable early-stop style guardrail**: Stop when best metric has not improved for N validations.

Example continuation override:
```bash
uv run python scripts/runners/train.py mode=train experiment=parseq_flash_fast checkpoint_path=<best_ckpt> trainer.max_epochs=<target> train.optimizer.lr=2e-4 trainer.val_check_interval=1.0 train.logger.wandb.log_config=false
```

Suggested scheduler stabilization (optional):
```bash
uv run python scripts/runners/train.py mode=train experiment=parseq_flash_fast checkpoint_path=<best_ckpt> trainer.max_epochs=<target> train.optimizer.lr=2e-4 train.lr_scheduler._target_=torch.optim.lr_scheduler.ReduceLROnPlateau train.lr_scheduler.mode=max train.lr_scheduler.monitor=val/acc train.lr_scheduler.factor=0.5 train.lr_scheduler.patience=3 train.lr_scheduler.min_lr=1e-6 trainer.val_check_interval=1.0
```

## Raw Metrics Summary

### Configuration
```json
{}
```

### Summary Values
```json
{
  "_runtime": 28876,
  "_step": 1925,
  "_timestamp": 1771126773.960006,
  "_wandb": {
    "runtime": 28876
  },
  "checkpoint_dir": "/mnt/external_artifacts/outputs/checkpoints",
  "epoch": 29,
  "train/loss": 0.2441025823354721,
  "trainer/global_step": 130049,
  "val/acc": 0.8093437552452087,
  "val/cer": 0.11918019503355026,
  "val_loss": 0.3911342918872833
}
```
