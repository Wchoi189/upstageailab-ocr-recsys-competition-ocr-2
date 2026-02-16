# Performance Baseline Report

**Generated:** 2026-02-17 02:21:49
**WandB Run:** [wchoi189_bs160_SCORE_PLACEHOLDER](https://wandb.ai/runs/ee562loz)
**Run ID:** `ee562loz`
**Status:** finished

---

## Recognition Metrics Summary

**Note:** Performance profiler metrics were not logged; report focuses on recognition quality trends.
**Trend Source:** `wandb_history`

| Metric | Best | Final | Delta |
|--------|------|-------|-------|
| **val/acc** | 0.8275 | 0.8275 | +0.0000 |
| **val/cer** | 0.1029 | 0.1029 | +0.0000 |
| **val_loss** | 0.3190 | 0.3190 | +0.0000 |

| Training Details | Value |
|------------------|-------|
| **Training Loss (final)** | 0.0363 |
| **Validation Loss (final)** | 0.3190 |
| **Validation Accuracy (final)** | 0.8275 |
| **Validation CER (final)** | 0.1029 |
| **Epoch** | 39 |
| **Global Step** | 213768 |
| **History Points (val/acc)** | 0 |

## Best/Final Snapshot Hints

- **Best val/acc point:** N/A
- **Final val/acc point:** N/A

## Run Insights (Learning Guide)

This run improved early, then regressed late. That usually means optimization overshot the best region rather than the model failing to learn.

### Illustrated Interpretation

- **Best quality reached:** val/acc 0.8275, val/cer 0.1029
- **End of run:** val/acc 0.8275, val/cer 0.1029
- **Regression size:** Δacc +0.0000, Δcer +0.0000

Simple mental model:
- Training = searching for a valley in error landscape.
- Best checkpoint = lowest spot reached so far.
- Late regression = optimizer steps moved away from that spot.

## Hypothesis for Latest Regression

Most plausible explanation is late-stage optimization instability during continuation from a strong checkpoint.

- The model likely reached a good local optimum early in resumed training.
- Continued updates (and scheduler behavior) moved parameters away from that optimum.
- Validation noise from frequent in-epoch checks amplifies apparent fluctuations.

## Training vs Validation Comparison

- **Note:** Performance profiling not enabled - cannot compare training vs validation timing.

## Identified Issues

### 1. Overfitting detected (MEDIUM)

Validation loss (0.319) is significantly higher than training loss (0.036)

## Next Run Checklist (Auto-Gated)

### Pre-run Gates

| Gate | Status | Evidence | Action |
|------|--------|----------|--------|
| **Resume LR conservative (<=2e-4)** | ✅ PASS | Δacc=-0.0000 from best to final | Lower LR by 5-10x for continuation runs. |
| **Validation cadence stable** | ✅ PASS | Current run shows regression after interim peaks | Use `trainer.val_check_interval=1.0` for cleaner epoch-level signal. |

### In-run Gates

| Gate | Status | Evidence | Action |
|------|--------|----------|--------|
| **No significant accuracy backslide** | ✅ PASS | best=0.8275, final=0.8275 | Early stop if `val/acc` drops >0.02 from run-best. |
| **CER remains near best** | ✅ PASS | best=0.1029, final=0.1029 | Reduce LR / halt when CER rises persistently. |

### Post-run Gates

| Gate | Status | Evidence | Action |
|------|--------|----------|--------|
| **Inference checkpoint selection** | ✅ PASS | Final underperformed best by 0.0000 acc | Publish best-acc checkpoint, not final-epoch checkpoint. |
| **Continue training decision** | ✅ PASS | Regression indicates optimization instability | Continue only with reduced LR + conservative scheduler. |

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
  "_runtime": 31174,
  "_step": 8767,
  "_timestamp": 1771260866.8624196,
  "_wandb": {
    "runtime": 31174
  },
  "checkpoint_dir": "/mnt/external_artifacts/outputs/checkpoints",
  "epoch": 39,
  "train/loss": 0.036274198442697525,
  "trainer/global_step": 213768,
  "val/acc": 0.8274999856948853,
  "val/cer": 0.10294077545404434,
  "val_loss": 0.3190430700778961
}
```
