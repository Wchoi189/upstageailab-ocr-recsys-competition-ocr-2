**Notepad for train cmd's**

## Fast Iterative Training Commands for Debugging
```bash

```



## Fast Non Zero metrics Training Commands for Debugging
To stay under ~5 min while still driving the metrics upward, you can keep the dataset canonical but trim how much of it each epoch sees and cut the number of epochs. The command below typically finishes in 3–6 minutes on the RTX 3060 (batch 4, only 15 % of the train set each epoch, 3 epochs total).

```bash
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  data=canonical \
  data/performance_preset=minimal \
  batch_size=4 \
  data.train_num_samples=1024 \
  data.val_num_samples=256 \
  data.test_num_samples=256 \
  dataloaders.train_dataloader.num_workers=2 \
  dataloaders.val_dataloader.num_workers=2 \
  dataloaders.test_dataloader.num_workers=2 \
  trainer.max_epochs=3 \
  trainer.limit_train_batches=0.15 \
  trainer.limit_val_batches=0.5 \
  trainer.limit_test_batches=0.5 \
  trainer.accumulate_grad_batches=1 \
  trainer.log_every_n_steps=10 \
  seed=123
```

## Batch size 16
```
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  data=canonical \
  batch_size=16 \
  trainer.max_epochs=16 \
  dataloaders.train_dataloader.num_workers=8 \
  dataloaders.val_dataloader.num_workers=8
```

## Batch size 8
```
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  data=canonical \
  batch_size=8 \
  trainer.max_epochs=16 \
  dataloaders.train_dataloader.num_workers=8 \
  dataloaders.val_dataloader.num_workers=8
```
