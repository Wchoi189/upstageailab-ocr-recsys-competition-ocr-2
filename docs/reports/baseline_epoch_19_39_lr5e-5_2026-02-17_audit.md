# Performance Baseline Report

**Generated:** 2026-02-17 19:39:34
**WandB Run:** [wchoi189_bs160_SCORE_PLACEHOLDER](https://wandb.ai/runs/3tk6nz1n)
**Run ID:** `3tk6nz1n`
**Status:** finished

---

## Recognition Metrics Summary

**Note:** Performance profiler metrics were not logged; report focuses on recognition quality trends.
**Trend Source:** `wandb_history`

| Metric | Best | Final | Delta |
|--------|------|-------|-------|
| **val/acc** | 0.8656 | 0.8656 | +0.0000 |
| **val/cer** | 0.0768 | 0.0768 | +0.0000 |
| **val_loss** | 0.2336 | 0.2336 | +0.0000 |

| Training Details | Value |
|------------------|-------|
| **Training Loss (final)** | 0.0269 |
| **Validation Loss (final)** | 0.2336 |
| **Validation Accuracy (final)** | 0.8656 |
| **Validation CER (final)** | 0.0768 |
| **Epoch** | 49 |
| **Global Step** | 297468 |
| **History Points (val/acc)** | 0 |

## Best/Final Snapshot Hints

- **Best val/acc point:** N/A
- **Final val/acc point:** N/A

## Run Insights (Learning Guide)

This run improved early, then regressed late. That usually means optimization overshot the best region rather than the model failing to learn.

### Illustrated Interpretation

- **Best quality reached:** val/acc 0.8656, val/cer 0.0768
- **End of run:** val/acc 0.8656, val/cer 0.0768
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

Validation loss (0.234) is significantly higher than training loss (0.027)

## Next Run Checklist (Auto-Gated)

### Pre-run Gates

| Gate | Status | Evidence | Action |
|------|--------|----------|--------|
| **Resume LR conservative (<=2e-4)** | ✅ PASS | Δacc=-0.0000 from best to final | Lower LR by 5-10x for continuation runs. |
| **Validation cadence stable** | ✅ PASS | Current run shows regression after interim peaks | Use `trainer.val_check_interval=1.0` for cleaner epoch-level signal. |

### In-run Gates

| Gate | Status | Evidence | Action |
|------|--------|----------|--------|
| **No significant accuracy backslide** | ✅ PASS | best=0.8656, final=0.8656 | Early stop if `val/acc` drops >0.02 from run-best. |
| **CER remains near best** | ✅ PASS | best=0.0768, final=0.0768 | Reduce LR / halt when CER rises persistently. |

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
  "_runtime": 17387,
  "_step": 4654,
  "_timestamp": 1771292842.4287038,
  "_wandb": {
    "runtime": 17387
  },
  "audit/high_loss_samples": {
    "_type": "images/separated",
    "captions": [
      "Epoch 49 | None | GT: \uc7a5 | Pr: 23",
      "Epoch 49 | None | GT: T623-870 | Pr: 623-870",
      "Epoch 49 | None | GT: \ubc88\uc9c0dhl | Pr: \ubc88\uc9c0\uc758",
      "Epoch 49 | None | GT: -1 | Pr: 4",
      "Epoch 49 | None | GT: KTD\ub300\ub3552\ud3c9\uc0dd\uad50\uc721\uc6d0\uc7a5, | Pr: \uc2dcT1C9\ub355 2 \uc0dd\ud589:\uad50\uc721\uc6d0\uc7a5",
      "Epoch 49 | None | GT: \uc900,\uacf5\uc2dc | Pr: \uc900\uacf5\uc2dc",
      "Epoch 49 | None | GT: \ub2ec\ub9ac\ubd84\ub958\ub418\uc9c0\uc54a\uc740\uae30\ud0c0\ub098\ubb34\ubc0f\ucf5c\ud06c\uc81c\uc870\uc5c5 | Pr: \uc0dd\uc77c\uc774 \uc0ac\uc5c5\uc7744 2\ub85c 1\uc5c51\uc5c5 \uc9c0     ",
      "Epoch 49 | None | GT: 6\uc545 | Pr: 605",
      "Epoch 49 | None | GT: \ud2cd \ubcf4 | Pr: \ud1b5\ubcf4",
      "Epoch 49 | None | GT: \ub2e4 | Pr: 2",
      "Epoch 49 | None | GT: \ub18d\uc9c0 \uc77c\uc2dc\uc804\uc6a9\ud5c8\uac00\uc2e0\uccad\uc5d0 \ub530\ub978 | Pr: \ubd09\uc5c5\uc67820666372(\uc6d0333-000003",
      "Epoch 49 | None | GT: \uc774\uc6a9\uc5d0\uad00\ud55c | Pr: 0.\uc6a9\uc5d0.\ud55c\ud55c"
    ],
    "count": 12,
    "filenames": [
      "media/images/audit/high_loss_samples_4650_9b3cbf526135d7c5cb61.png",
      "media/images/audit/high_loss_samples_4650_7d149d7c32833df14327.png",
      "media/images/audit/high_loss_samples_4650_3e4e1e63325d45e48635.png",
      "media/images/audit/high_loss_samples_4650_40edf4549ec93cafaa60.png",
      "media/images/audit/high_loss_samples_4650_8669993cb4449b1321e7.png",
      "media/images/audit/high_loss_samples_4650_df553f15d4bff2fbccb5.png",
      "media/images/audit/high_loss_samples_4650_3e631e743968550c6537.png",
      "media/images/audit/high_loss_samples_4650_36335ab6533c08d00aac.png",
      "media/images/audit/high_loss_samples_4650_bc6f21294068d59bea52.png",
      "media/images/audit/high_loss_samples_4650_8af5d96f281f1189761a.png",
      "media/images/audit/high_loss_samples_4650_97441026b8af0568fc62.png",
      "media/images/audit/high_loss_samples_4650_71ba3743026e71bc4bba.png"
    ],
    "format": "png",
    "height": 112,
    "width": 128
  },
  "audit/high_loss_table": {
    "_latest_artifact_path": "wandb-client-artifact://5fqbq9ho5xz0b651hava7m9ycuutfs2bbbc28ewzszbwen36nfvgkq9jhv52hb3egl8i00gye2x3ctttcfw6wq8tucdtdinf9zl9qfvlv5e4rw1ny5ondturw3rfgv5m:latest/audit/high_loss_table.table.json",
    "_type": "table-file",
    "artifact_path": "wandb-client-artifact://6jny1mm0rqqy7lj8rw49iye7apyijfh0mimf7egbt0fdsp3adf9oxybci2eo91rqitq8yflojpffyq0jhoqmoslt1hodh537p1l3a5d59n4qs0dp7c16qqzaxux7vv3i/audit/high_loss_table.table.json",
    "log_mode": "IMMUTABLE",
    "ncols": 9,
    "nrows": 12,
    "path": "media/table/audit/high_loss_table_4651_b77c34d63e177d5cd201.table.json",
    "sha256": "b77c34d63e177d5cd201c045be53dc7a4cc3980ae13bdb5033f20f14120dc5ee",
    "size": 1485
  },
  "checkpoint_dir": "/mnt/external_artifacts/outputs/checkpoints",
  "epoch": 49,
  "train/loss": 0.0268910713493824,
  "trainer/global_step": 297468,
  "val/acc": 0.8655624985694885,
  "val/cer": 0.07677727192640305,
  "val_loss": 0.23356182873249057
}
```


# Sample Data
```csv
"epoch","global_step","batch_idx","sample_idx","loss","gt_text","pred_text","is_exact_match"
"49","297470","152","20","9.619140625","장","23","false"
"49","297470","43","68","9.325770378112793","T623-870","623-870","false"
"49","297470","21","66","9.299538612365723","번지dhl","번지의","false"
"49","297470","10","79","8.64371395111084","-1","4","false"
"49","297470","4","92","8.184186935424805","KTD대덕2평생교육원장,","시T1C9덕 2 생행:교육원장","false"
"49","297470","174","132","7.987441062927246","준,공시","준공시","false"
"49","297470","79","34","7.7521586418151855","달리분류되지않은기타나무및콜크제조업","생일이 사업이4 2로 1업1업 지     ","false"
"49","297470","100","137","7.256076335906982","6악","605","false"
"49","297470","76","41","7.115131855010986","틍 보","통보","false"
"49","297470","118","71","6.9141845703125","다","2","false"
"49","297470","118","61","6.8726806640625","농지 일시전용허가신청에 따른","봉업외20666372(원333-000003","false"
"49","297470","11","117","6.861979007720947","이용에관한","0.용에.한한","false"
```
