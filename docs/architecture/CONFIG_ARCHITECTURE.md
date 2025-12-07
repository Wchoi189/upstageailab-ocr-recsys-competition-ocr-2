# Configuration Architecture Reference

## Overview
- Total active groups: 17
- Total active files: 67
- Typical merge: 28-30 files per training run
- Cognitive Load Score: 7.1/10 (Phase 0 baseline)

## Configuration Groups (17 total)

### 1. **Core Training** (8 files)
- `train.yaml` - Main entry point
- `base.yaml` - Root defaults
- `preset/` - Preset configurations (2 files)

### 2. **Data** (12 files)
- `data/default.yaml` - Base dataset config
- `data/synthetic.yaml` - Synthetic data settings
- `dataloaders/` - DataLoader configs (3 files)
- `data/` subdirs (6 more files)

### 3. **Model Architecture** (22 files)
- `model/default.yaml` - Base model config
- `model/architectures/` - Specific architectures (5 files)
- `preset/models/encoder/` - Encoder options (4 files)
- `preset/models/decoder/` - Decoder options (4 files)
- `preset/models/head/` - Head options (3 files)
- `preset/models/loss/` - Loss options (3 files)

### 4. **Training** (8 files)
- `trainer/default.yaml` - Lightning Trainer config
- `trainer/debug/` - Debug overrides
- `callbacks/default.yaml` - Callback configuration
- `callbacks/metadata.yaml` - Metadata callback config

### 5. **Optimization** (7 files)
- `model/optimizers/` - Optimizer configs (3 files)
- `model/schedulers/` - LR scheduler configs (4 files)

### 6. **Logging & Monitoring** (4 files)
- `logger/default.yaml` - Base logger config
- `logger/wandb.yaml` - Weights & Biases config
- `logger/csv.yaml` - CSV logger config
- `callbacks/` - Additional logging callbacks

### 7. **Paths & Hydra** (3 files)
- `paths/default.yaml` - File paths
- `hydra/job_logging` - Hydra job logging
- `hydra/runtime_choices` - Hydra runtime choices

### 8-17. **Specialized Groups** (3 files total)
- `metrics/` - Evaluation metrics (1 file)
- `transforms/` - Data transforms (1 file)
- `extras/` - Extra configs (1 file)

## Training Run Merge Trace

When you run `python train.py`, here's the order configs are merged:

```
1. train.yaml (includes defaults list)
   ├── 2. base.yaml (@package _global_)
   │   ├── 3. data/default.yaml (@package data)
   │   ├── 4. model/default.yaml (@package model)
   │   │   ├── 5. model/architectures/dbnet.yaml (@package model)
   │   │   ├── 6. preset/models/encoder/timm_backbone.yaml (@package model.encoder)
   │   │   ├── 7. preset/models/decoder/ppm.yaml (@package model.decoder)
   │   │   ├── 8. preset/models/head/dbnet_head.yaml (@package model.head)
   │   │   └── 9. preset/models/loss/dbnet_loss.yaml (@package model.loss)
   │   ├── 10. model/optimizers/adamw.yaml (@package model.optimizer)
   │   ├── 11. model/schedulers/warmup_cosine.yaml (@package model.scheduler)
   │   ├── 12. trainer/default.yaml (@package trainer)
   │   ├── 13. callbacks/default.yaml (@package callbacks)
   │   ├── 14. callbacks/metadata.yaml (@package callbacks)
   │   ├── 15. logger/default.yaml (@package logger)
   │   ├── 16. logger/wandb.yaml (@package logger.wandb)
   │   ├── 17. paths/default.yaml (@package paths)
   │   └── ... (15+ more files)
   └── 18-28+. (Additional overrides and nested defaults)
```

## Variable Interpolation Map

Key variables defined in base configs:

```yaml
# base.yaml
dataset_path: ${paths.data_dir}/synthetic
dataset_module: ocr.datasets.doctr_synthetic
encoder_path: ocr.models.encoders.TimmBackbone
decoder_path: ocr.models.decoders.PPMDecoder
head_path: ocr.models.heads.DBNetHead
loss_path: ocr.losses.DBNetLoss

# These are used throughout to enable CLI overrides:
# python train.py encoder_path=ocr.models.encoders.CraftVGG
```

## @package Directive Reference

All 11 @package targets currently in use:

| Target | Usage | Files | Purpose |
|--------|-------|-------|---------|
| `@package _global_` | Root merging | 18 configs | Merge at root level (silent) |
| `@package model` | Model nesting | 8 configs | Nest under `model` key |
| `@package model.encoder` | Component nesting | 1 config | Nest under `model.encoder` |
| `@package model.decoder` | Component nesting | 1 config | Nest under `model.decoder` |
| `@package model.head` | Component nesting | 1 config | Nest under `model.head` |
| `@package model.loss` | Component nesting | 1 config | Nest under `model.loss` |
| `@package model.optimizer` | Component nesting | 1 config | Nest under `model.optimizer` |
| `@package model.scheduler` | Component nesting | 1 config | Nest under `model.scheduler` |
| `@package callbacks` | Callback nesting | 3 configs | Nest under `callbacks` |
| `@package logger` | Logger nesting | 2 configs | Nest under `logger` |
| `@package logger.wandb` | Logger sub-nesting | 1 config | Nest under `logger.wandb` |

**Key Finding**: `@package _global_` is overused (18 configs). Phase 1 will reduce this.

## Config Loading in Different Contexts

### Training (`python train.py`)
- Entry point: `train.yaml`
- Files merged: 28-30
- Config resolution: Uses Hydra decorator (@hydra.main)

### Inference (UI & API)
- Entry point: `train.yaml` (loaded from checkpoint metadata)
- Files merged: 28-30 (same as training)
- Config resolution: Uses `ui/utils/inference/config_loader.py`
- **Note**: Uses different loading mechanism than training

### Evaluation
- Entry point: `configs/test.yaml`
- Files merged: Similar to train (26-28 files)
- Config resolution: Uses Hydra decorator

## Known Issues & Workarounds

### Issue 1: Double-Nested Model Keys
**Symptom**: `cfg.model.model` instead of `cfg.model`
**Cause**: Both model/default.yaml and model/architectures/dbnet.yaml have `@package model`
**Workaround**: `load_config()` detects and unwraps this automatically
**Phase 1 Fix**: Will eliminate this by reducing @package targets

### Issue 2: Missing Variable Errors
**Symptom**: `${encoder_path}` undefined when loading certain configs
**Cause**: Variable defined in base.yaml but loaded config doesn't include base.yaml
**Workaround**: Always load base.yaml first
**Phase 1 Fix**: Will consolidate variable definitions

### Issue 3: Implicit Global Merging
**Symptom**: `@package _global_` causes silent merging without explicit imports
**Cause**: Hydra design; configs merge without being referenced
**Workaround**: Check for conflicts manually
**Phase 1 Fix**: Replace many `@package _global_` with explicit nesting

## Phase 0-2 Roadmap

| Phase | Duration | Target | Benefit |
|-------|----------|--------|---------|
| **Current** | - | 80 files / 17 groups / 7.1 cognitive load | Baseline |
| **Phase 0** | 2h | 67 files / 16 groups / 7.0 cognitive load | Removes clutter |
| **Phase 1** | 8h | 60 files / 14 groups / 5.5 cognitive load | Fewer @package targets |
| **Phase 2** | 16h | 35 files / 9 groups / 4.2 cognitive load | Flatter structure |

## How to Use This Document

1. **Understanding the system?** Start with "Overview" and "Configuration Groups"
2. **Debugging config issues?** Check "Known Issues & Workarounds"
3. **Adding new configs?** Reference "@package Directive Reference"
4. **Migrating to new structure?** Follow Phase 0-2 Roadmap

---

**Last Updated**: 2025-12-04
**Next Review**: After Phase 0 completion
