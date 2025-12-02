# Architecture Reference

**Purpose:** Key facts about system architecture. For detailed context, see `docs/maintainers/architecture/`.

## System Overview

**Framework:** Modular, plug-and-play architecture
- Components: Encoders, Decoders, Heads, Losses
- Registry: Central catalog of components
- Configuration: Hydra-based declarative configs

## Directory Structure

```
src/ocr_framework/
├── architectures/     # Architecture implementations (dbnet, east)
├── core/             # Abstract base classes
├── models/           # Model factory and composite model
├── datasets/         # Data loading
├── training/         # Training logic
├── evaluation/       # Evaluation metrics
└── utils/            # Utility functions
```

## Component Registry

**Base Classes:**
- `BaseEncoder` → Encoder registry
- `BaseDecoder` → Decoder registry
- `BaseHead` → Head registry
- `BaseLoss` → Loss registry

**Model Factory:**
- Assembles models from registered components
- Uses Hydra configs for component selection

## Hydra Configuration

**Config Structure:**
```
configs/
├── data/
├── model/          # encoder/, decoder/, head/, loss/
├── trainer/
├── logger/         # wandb.yaml, csv.yaml, default.yaml
└── train.yaml      # Main config
```

**Usage:**
```bash
# Basic training
uv run python runners/train.py preset=<name>

# Override parameters
uv run python runners/train.py model.optimizer.lr=0.0005 data.batch_size=16

# Switch architectures
uv run python runners/train.py model.architecture=east
```

**Instantiation:**
```python
from hydra.utils import instantiate

config = {
    '_target_': 'ocr_framework.architectures.dbnet.encoder.TimmBackbone',
    'backbone': 'resnet50',
    'pretrained': True
}
encoder = instantiate(config)
```

## Data Flow

**Training:**
1. Input Image → OCRTransforms
2. ValidatedOCRDataset → DataLoader
3. OCRLightningModule → OCRModel
4. Encoder → Decoder → Head → Loss
5. Backward pass → Optimizer update

## Key Components

**Encoders:** TimmBackbone (ResNet, EfficientNet, etc.)
**Decoders:** DBNet decoder, EAST decoder
**Heads:** DBNet head, EAST head
**Losses:** DBNet loss, EAST loss

## Compatibility Rules

- Output channels of encoder must match input channels of decoder
- Some models require specific data formats (e.g., CRAFT needs character-level annotations)
- Check shape compatibility when adding new components
