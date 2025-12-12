# Training Guide

## Quick Start

```bash
# Basic training with default config
uv run python runners/train.py model/presets=model_example

# With custom parameters
uv run python runners/train.py \
    model/presets=model_example \
    trainer.max_epochs=20 \
    data.batch_size=8
```

## Data Preparation

### Directory Structure

```
data/datasets/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── jsons/
    ├── train.json
    ├── val.json
    └── test.json
```

### Annotation Format

JSON files should contain:
```json
{
  "images": [...],
  "annotations": [
    {
      "image_id": 1,
      "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "text": "receipt text"
    }
  ]
}
```

## Offline Preprocessing (Recommended)

Pre-compute probability and threshold maps for 5-8x faster training:

```bash
# Preprocess entire dataset
uv run python scripts/preprocess_maps.py

# Test with limited samples
uv run python scripts/preprocess_maps.py \
    data.train_num_samples=100 \
    data.val_num_samples=20
```

## Training Configurations

### Model Presets

Choose from available model presets:
- `model_example` - DBNet with ResNet18 encoder
- `craft` - CRAFT architecture
- `dbnetpp` - DBNet++ with advanced features

```bash
uv run python runners/train.py model/presets=dbnetpp
```

### Common Overrides

```bash
# Adjust batch size and epochs
uv run python runners/train.py \
    data.batch_size=16 \
    trainer.max_epochs=50

# Use specific GPU
uv run python runners/train.py trainer.devices=[0]

# Enable gradient accumulation
uv run python runners/train.py \
    data.batch_size=4 \
    trainer.accumulate_grad_batches=4

# Limit training samples (for testing)
uv run python runners/train.py \
    data.train_num_samples=100 \
    data.val_num_samples=20
```

## Monitoring Training

### Weights & Biases

Training automatically logs to W&B if configured:
- Metrics: loss, precision, recall, H-mean
- Artifacts: checkpoints, predictions
- System: GPU usage, memory

### Local Logs

Outputs are saved to:
```
outputs/experiments/train/ocr/ocr_training_b/<run_id>/
├── checkpoints/
│   ├── best.ckpt
│   └── latest.ckpt
├── logs/
└── predictions/
```

## Testing and Evaluation

```bash
# Test trained model
uv run python runners/test.py \
    model/presets=model_example \
    checkpoint_path="outputs/experiments/train/ocr/ocr_training_b/<run_id>/checkpoints/best.ckpt"

# Generate predictions
uv run python runners/predict.py \
    model/presets=model_example \
    checkpoint_path="outputs/experiments/train/ocr/ocr_training_b/<run_id>/checkpoints/best.ckpt"
```

## Performance Optimization

### Memory Optimization
- Reduce batch size
- Enable gradient accumulation
- Use mixed precision training
- Enable gradient checkpointing

### Speed Optimization
- Use offline preprocessing
- Increase num_workers in dataloaders
- Use performance presets: `data/performance_preset=balanced`

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
uv run python runners/train.py data.batch_size=4

# Or use gradient accumulation
uv run python runners/train.py \
    data.batch_size=4 \
    trainer.accumulate_grad_batches=4
```

### Slow Training
```bash
# Run preprocessing first
uv run python scripts/preprocess_maps.py

# Use performance preset
uv run python runners/train.py data/performance_preset=balanced
```

## Next Steps

- [Configuration Guide](../architecture/CONFIG_ARCHITECTURE.md) - Deep dive into configs
- [Evaluation Guide](evaluation.md) - Analyze model performance
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
