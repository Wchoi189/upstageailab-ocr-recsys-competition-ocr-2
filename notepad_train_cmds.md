**Notepad for train cmd's**

## Fast Iterative Training Commands (Detection Domain)
```bash
# Debug run (Detection) - 1 epoch, minimal batches
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  domain=detection \
  data=canonical \
  data/performance_preset=minimal \
  batch_size=4 \
  data.train_num_samples=1024 \
  data.val_num_samples=256 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=0.15 \
  trainer.limit_val_batches=0.5 \
  seed=123
```

## Fast Iterative Training Commands (Recognition Domain)
```bash
# Debug run (Recognition)
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  domain=recognition \
  data=canonical \
  batch_size=8 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=100 \
  seed=123
```

## Standard Training (Detection)
```bash
# Standard run
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  domain=detection \
  data=canonical \
  batch_size=8 \
  trainer.max_epochs=2 \
  dataloaders.train_dataloader.num_workers=8
```

## Quick Verification (No Validation)
```bash
# Fast check without validation loop
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  domain=detection \
  batch_size=4 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=0 \
  logger.wandb.enabled=false
```


##  Recognition
```bash
# Debug run (Recognition)
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  domain=recognition \
  data=canonical \
  batch_size=8 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=100 \
  seed=123
```

## Standard Training (Recognition)
```bash
# Standard run
PYTHONUNBUFFERED=1 /workspaces/upstageailab-ocr-recsys-competition-ocr-2/.venv/bin/python runners/train.py \
  domain=recognition \
  batch_size=8 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=4 \
  logger.wandb.enabled=true \
  ++logger.wandb.job_type="debug_fix_final_v3" \
  ++logger.wandb.project="ocr-debug"
```

## Improve GPU Utilization
```
Currently:

Utilization: 72% (Low for training)
Memory: 5.3GB / 24GB (Very Low - ~22%)
Bottleneck: Validation is slow (2.54 it/s) because batch size is small (128), making the sequential decoding inefficient.
Optimization Plan (Ready-to-Run):

Enable Mixed Precision: trainer.precision=16-mixed (Faster math, less memory).
Increase Batch Size: batch_size=448 (Utilize closer to 20GB VRAM).
Speedup: This will make validation 3-4x faster (parallelizing the slow decoding) and training 2x faster.
```
### Recommended run:
```bash
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  experiment.name=optimized_v1 \
  domain=recognition \
  batch_size=448 \
  trainer.max_epochs=10 \
  trainer.check_val_every_n_epoch=1 \
  trainer.precision=16-mixed \
  logger.wandb.enabled=true \
  ++logger.wandb.project="ocr-recognition-opt"
```
### Debug run (Recognition)
```bash
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  experiment.name=diagnose_collapse_v3_opt \
  domain=recognition \
  batch_size=448 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=100 \
  trainer.limit_val_batches=4 \
  trainer.limit_test_batches=0 \
  trainer.check_val_every_n_epoch=1 \
  trainer.precision=16-mixed \
  logger.wandb.enabled=true \
  ++logger.wandb.project="ocr-debug"
```
### Slower learning
```
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  experiment.name=fix_pos_scale_overfit \
  domain=recognition \
  batch_size=32 \
  trainer.max_epochs=100 \
  +trainer.overfit_batches=1 \
  trainer.limit_val_batches=4 \
  trainer.limit_test_batches=0 \
  trainer.check_val_every_n_epoch=10 \
  +optimizer.lr=1e-4 \
  trainer.precision=32 \
  logger.wandb.enabled=true \
  ++logger.wandb.project="ocr-debug"
```

### Faster learning
```

UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  experiment.name=fix_all_high_lr_overfit \
  domain=recognition \
  batch_size=32 \
  trainer.max_epochs=200 \
  +trainer.overfit_batches=1 \
  trainer.limit_val_batches=4 \
  +skip_test=true \
  trainer.check_val_every_n_epoch=20 \
  +optimizer.lr=1e-3 \
  trainer.precision=32 \
  logger.wandb.enabled=true \
  ++logger.wandb.project="ocr-debug"
```

### Xavier Initialization
```
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  experiment.name=fix_init_overfit \
  domain=recognition \
  batch_size=32 \
  trainer.max_epochs=200 \
  +trainer.overfit_batches=1 \
  trainer.limit_val_batches=4 \
  +skip_test=true \
  trainer.check_val_every_n_epoch=20 \
  +optimizer.lr=1e-3 \
  trainer.precision=32 \
  logger.wandb.enabled=true \
  ++logger.wandb.project="ocr-debug"
```
