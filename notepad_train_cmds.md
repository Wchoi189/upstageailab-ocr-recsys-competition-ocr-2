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


