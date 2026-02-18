# WandB Run Name Generation

## Overview

This project uses intelligent run name generation to create descriptive, searchable WandB run names instead of random names like "smooth-music-123".

## Example Generated Name

```
user_parseq-timm-backbone-parseq-decoder-parseq-head-cross-entropy-bs64-lr2e-4_acc0.8267
```

This includes:
- **User prefix**: `user` (from `WANDB_USER` env var or config)
- **Architecture**: `parseq`
- **Components**: `timm-backbone`, `parseq-decoder`, `parseq-head`
- **Loss**: `cross-entropy`
- **Hyperparameters**: `bs64` (batch size), `lr2e-4` (learning rate)
- **Final metric**: `acc0.8267` (added at training end)

## How to Enable/Disable

### Method 1: In Experiment Config (Recommended)

```yaml
# configs/experiment/your_experiment.yaml
train:
  logger:
    wandb:
      standardize_name: true  # Enable descriptive names
```

### Method 2: CLI Override

```bash
# Enable
uv run python scripts/runners/train.py ... train.logger.wandb.standardize_name=true

# Disable (use random WandB names)
uv run python scripts/runners/train.py ... train.logger.wandb.standardize_name=false
```

### Method 3: Base Logger Config

Edit [configs/train/logger/wandb.yaml](../configs/train/logger/wandb.yaml):

```yaml
standardize_name: true  # Default for all runs
```

## Name Components

The name generation extracts info from your config:

| Component | Config Path | Example |
|-----------|-------------|---------|
| User | `$WANDB_USER` env var | `user` |
| Tag | `wandb.experiment_tag` | `baseline` |
| Architecture | `model.architecture_name` | `parseq` |
| Encoder | `model.encoder.name` | `timm-backbone` |
| Decoder | `model.decoder.name` | `parseq-decoder` |
| Head | `model.head.name` | `parseq-head` |
| Loss | `model.loss.name` | `cross-entropy` |
| Batch Size | `data.batch_size` | `bs64` |
| Learning Rate | `train.optimizer.lr` | `lr2e-4` |

## Name Length Handling

Names are automatically truncated if they exceed 120 characters:
1. First, less important components are removed (in order: loss, head, lr, batch, decoder, encoder)
2. If still too long, a hash digest is added to maintain uniqueness

## Final Metric Injection

The `_SCORE_PLACEHOLDER` suffix is replaced with actual metrics after training:

```python
from ocr.core.utils.wandb_base import finalize_run

# At end of training
finalize_run({"val/acc": 0.8267})
# Changes: user_parseq_SCORE_PLACEHOLDER → user_parseq_acc0.8267
```

## Source Code

- Name generation: [ocr/core/utils/wandb_base.py](../ocr/core/utils/wandb_base.py#L312)
- Orchestrator integration: [ocr/pipelines/orchestrator.py](../ocr/pipelines/orchestrator.py#L206-L210)

## Examples

**With standardize_name enabled:**
```
user_parseq-flash-timm-backbone-bs64-lr2e-4_acc0.8521
```

**With standardize_name disabled:**
```
smooth-music-123  # Random WandB generated name
```

## Troubleshooting

### Names are still random

Check your experiment config includes:
```yaml
train:
  logger:
    wandb:
      standardize_name: true
```

### Name is missing components

The name generator only includes components it can find in the config. If your model doesn't define `encoder.name`, that component will be skipped.

### Want to add custom tags

Set in your config:
```yaml
wandb:
  experiment_tag: "my-experiment"
```

This will prefix the name: `user_my-experiment_parseq-...`
