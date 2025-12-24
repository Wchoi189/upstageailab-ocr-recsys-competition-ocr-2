# Hydra Configuration Structure Reference

## Overview

The OCR project uses **Hydra** for configuration management with a refactored architecture.

## Foundation: `configs/_base/`

The **new default configuration system** uses `configs/_base/` as the building foundation for all other configurations.

### Base Configuration Files

Located in `configs/_base/`:
- `core.yaml` - Core configuration
- `data.yaml` - Data configuration
- `logging.yaml` - Logging configuration
- `model.yaml` - Model configuration
- `preprocessing.yaml` - Preprocessing configuration
- `trainer.yaml` - Trainer configuration

## Model Configuration Structure

**Location**: `configs/model/` (not `configs/models/`)

### Subdirectories:
- `architectures/` - Model architectures (craft.yaml, dbnet.yaml, dbnetpp.yaml)
- `decoder/` - Decoder configurations (craft_decoder.yaml, dbpp_decoder.yaml, fpn.yaml, pan.yaml, unet.yaml)
- `encoder/` - Encoder configurations (craft_vgg.yaml, timm_backbone.yaml)
- `head/` - Head configurations (craft_head.yaml, db_head.yaml, dbpp_head.yaml)
- `loss/` - Loss configurations (craft_loss.yaml, db_loss.yaml)
- `lightning_modules/` - Lightning module configurations (base.yaml)
- `presets/` - Model presets (craft.yaml, dbnetpp.yaml, model_example.yaml)
- `optimizers/` - Optimizer configurations (adam.yaml, adamw.yaml)
- `default.yaml` - Default model configuration

## Architecture Note

**Two architectures exist**:
1. **Old architecture** - Legacy configuration structure
2. **New architecture** - Uses `configs/_base/` as foundation (current default)

The new architecture consolidates configurations and reduces duplication.

## Configuration Hierarchy

```
configs/
├── _base/              # Foundation for all configs (NEW DEFAULT)
│   ├── core.yaml
│   ├── data.yaml
│   ├── logging.yaml
│   ├── model.yaml
│   ├── preprocessing.yaml
│   └── trainer.yaml
├── model/              # Model configurations (NOT models/)
│   ├── architectures/
│   ├── decoder/
│   ├── encoder/
│   ├── head/
│   ├── loss/
│   ├── lightning_modules/
│   ├── presets/
│   └── optimizers/
├── data/               # Data configurations
├── trainer/            # Trainer configurations
├── callbacks/          # Callback configurations
└── base.yaml           # Main base configuration
```

## Usage in AI Helpers

When referencing configurations in AI helpers:
- ✅ Use: `configs/_base/` (foundation)
- ✅ Use: `configs/model/` (model configs)
- ❌ Don't use: `configs/models/` (doesn't exist)

## References
- See: `.ai-instructions/INDEX.yaml` for AI instructions
- See: `.ai-instructions/tier1-sst/system-architecture.yaml` for system architecture

