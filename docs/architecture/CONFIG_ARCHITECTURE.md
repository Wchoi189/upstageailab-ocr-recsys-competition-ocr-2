# Configuration Architecture Reference

## Overview (Updated: 2025-12-11)
- **Total active files**: 89 YAML files
- **Active directories**: 17 subdirectories
- **Typical merge**: 20-25 files per training run
- **Cognitive Load Score**: 4.0/10 (Phases 5-8 complete)
- **Improvement**: 43% reduction from baseline (7.0 → 4.0)

## Major Improvements (Phases 5-8)

### Phase 5: Low-Hanging Fruit
- Deleted `.deprecated/`, `metrics/`, `extras/` directories
- Moved `ablation/` configs to `docs/research/`
- Moved `schemas/` to `docs/schemas/`
- Consolidated `hardware/` into `trainer/`

### Phase 6: Data Config Consolidation
- Created unified `data/` hierarchy
- Moved `dataloaders/` → `data/dataloaders/`
- Moved `transforms/` → `data/transforms/`
- Moved `preset/datasets/` → `data/datasets/`

### Phase 7: Preset/Models Elimination
- Created `model/encoder/`, `model/decoder/`, `model/head/`, `model/loss/`, `model/presets/`
- Eliminated `preset/models/` directory
- Single source of truth for all model configs

### Phase 8: Final Consolidation
- Moved `lightning_modules/` → `model/lightning_modules/`
- Eliminated entire `preset/` directory
- Relocated tool configs to `.vscode/`

## Current Directory Structure

```
configs/
├── _base/                    # Base configuration templates
├── base.yaml                 # Root configuration
├── benchmark/                # Benchmarking configs
├── callbacks/                # Training callbacks
├── data/                     # ALL data-related configs
│   ├── dataloaders/          # DataLoader configurations
│   ├── datasets/             # Dataset preprocessing configs
│   ├── performance_preset/   # Performance optimization presets
│   └── transforms/           # Data transformation configs
├── debug/                    # Debug configurations
├── evaluation/               # Evaluation metrics
├── hydra/                    # Hydra framework configs
├── logger/                   # Logging configurations
├── model/                    # ALL model-related configs
│   ├── architectures/        # Full architecture definitions
│   ├── decoder/              # Decoder component configs
│   ├── encoder/              # Encoder component configs
│   ├── head/                 # Head component configs
│   ├── lightning_modules/    # PyTorch Lightning modules
│   ├── loss/                 # Loss function configs
│   ├── optimizers/           # Optimizer configurations
│   └── presets/              # Full model composition presets
├── paths/                    # Path configurations
├── trainer/                  # PyTorch Lightning Trainer configs
├── ui/                       # UI-specific configs
├── ui_meta/                  # UI metadata configs
└── [entry-point configs]     # train.yaml, test.yaml, predict.yaml, etc.
```

## Configuration Groups

### 1. **Core Training** (4 files)
- `train.yaml` - Main training entry point
- `train_v2.yaml` - Alternative training config
- `base.yaml` - Root defaults
- `_base/` - Base configuration templates

### 2. **Data** (20 files in unified hierarchy)
- `data/` - All data-related configurations
  - `dataloaders/` - DataLoader configs (2 files)
  - `datasets/` - Dataset preprocessing (4 files)
  - `transforms/` - Data transformations (3 files)
  - `performance_preset/` - Performance optimization (6 files)
  - Top-level data configs (5 files)

### 3. **Model Architecture** (22 files in unified hierarchy)
- `model/` - All model-related configurations
  - `architectures/` - Full architectures (3 files)
  - `encoder/` - Encoder components (2 files)
  - `decoder/` - Decoder components (5 files)
  - `head/` - Head components (3 files)
  - `loss/` - Loss functions (2 files)
  - `presets/` - Full model compositions (3 files)
  - `lightning_modules/` - PyTorch Lightning modules (1 file)
  - `optimizers/` - Optimizers (2 files)
  - Top-level model config (1 file)

### 4. **Training & Callbacks** (12 files)
- `trainer/` - PyTorch Lightning Trainer configs (4 files)
- `callbacks/` - Training callbacks (8 files)

### 5. **Logging & Monitoring** (4 files)
- `logger/` - Logger configurations
  - Wandb, CSV, and consolidated loggers

### 6. **Paths & Hydra** (3 files)
- `paths/default.yaml` - File paths
- `hydra/` - Hydra framework configs (2 files)

### 7. **Specialized Configs** (24 files)
- `ui/` - UI-specific configs (5 files)
- `ui_meta/` - UI metadata (6 files)
- `evaluation/` - Evaluation metrics (1 file)
- `debug/` - Debug configs (1 file)
- `benchmark/` - Benchmarking (1 file)
- Entry-point configs (10 files)

## Training Run Merge Trace

When you run `python train.py`, configs are merged in this order:

```
1. train.yaml (includes defaults list)
   ├── 2. base.yaml (@package _global_)
   │   ├── 3. data/default.yaml
   │   │   ├── 4. data/transforms/base.yaml (@package data.transforms)
   │   │   └── 5. data/dataloaders/default.yaml (@package data.dataloaders)
   │   ├── 6. model/default.yaml
   │   │   ├── 7. model/architectures/dbnet.yaml (@package model)
   │   │   ├── 8. model/optimizers/adamw.yaml (@package model.optimizer)
   │   ├── 9. model/presets/model_example.yaml
   │   │   ├── 10. model/encoder/timm_backbone.yaml (@package model.encoder)
   │   │   ├── 11. model/decoder/unet.yaml (@package model.decoder)
   │   │   ├── 12. model/head/db_head.yaml (@package model.head)
   │   │   └── 13. model/loss/db_loss.yaml (@package model.loss)
   │   ├── 14. model/lightning_modules/base.yaml (@package _global_)
   │   ├── 15. trainer/default.yaml (@package trainer)
   │   ├── 16. callbacks/default.yaml (@package callbacks)
   │   ├── 17. logger/consolidated.yaml (@package logger)
   │   ├── 18. paths/default.yaml (@package paths)
   │   └── 19. evaluation/metrics.yaml (@package _global_)
   └── 20-25. (Additional overrides and nested defaults)
```

## @package Directive Reference

Current @package targets in use:

| Target | Usage | Files | Purpose |
|--------|-------|-------|------------|
| `@package _global_` | Root merging | ~10 configs | Merge at root level |
| `@package model` | Model nesting | 3 configs | Nest under `model` key |
| `@package model.encoder` | Component nesting | 2 configs | Nest under `model.encoder` |
| `@package model.decoder` | Component nesting | 5 configs | Nest under `model.decoder` |
| `@package model.head` | Component nesting | 3 configs | Nest under `model.head` |
| `@package model.loss` | Component nesting | 2 configs | Nest under `model.loss` |
| `@package data` | Data nesting | 5 configs | Nest under `data` key |
| `@package callbacks` | Callback nesting | 8 configs | Nest under `callbacks` |
| `@package logger` | Logger nesting | 4 configs | Nest under `logger` |
| `@package trainer` | Trainer nesting | 4 configs | Nest under `trainer` |

**Improvement**: Reduced `@package _global_` usage from 18 to ~10 configs.

## Key Organizational Principles

### 1. Single Source of Truth
- **All data configs** → `configs/data/`
- **All model configs** → `configs/model/`
- **All tool configs** → `.vscode/`

### 2. Clear Hierarchy
- Component configs nested under parent directories
- No duplicate or scattered configs
- Logical grouping by functionality

### 3. Separation of Concerns
- Hydra training configs in `configs/`
- IDE/tool configs in `.vscode/`
- Documentation schemas in `docs/schemas/`
- Research configs in `docs/research/`

## Config Loading in Different Contexts

### Training (`python train.py`)
- Entry point: `train.yaml`
- Files merged: 20-25 (reduced from 28-30)
- Config resolution: Uses Hydra decorator (@hydra.main)

### Inference (UI & API)
- Entry point: `train.yaml` (loaded from checkpoint metadata)
- Files merged: 20-25 (same as training)
- Config resolution: Uses checkpoint metadata + config loader
- Schema validation: Uses `docs/schemas/ui_inference_compat.yaml`

### Evaluation
- Entry point: `configs/test.yaml`
- Files merged: Similar to train (18-22 files)
- Config resolution: Uses Hydra decorator

## Migration Guide

### From Old Structure to New

**Data Configs:**
- `configs/dataloaders/` → `configs/data/dataloaders/`
- `configs/transforms/` → `configs/data/transforms/`
- `configs/preset/datasets/` → `configs/data/datasets/`

**Model Configs:**
- `configs/preset/models/encoder/` → `configs/model/encoder/`
- `configs/preset/models/decoder/` → `configs/model/decoder/`
- `configs/preset/models/head/` → `configs/model/head/`
- `configs/preset/models/loss/` → `configs/model/loss/`
- `configs/preset/models/*.yaml` → `configs/model/presets/`
- `configs/preset/lightning_modules/` → `configs/model/lightning_modules/`

**Tool Configs:**
- `configs/tools/*.json` → `.vscode/*.json`

**Deleted Directories:**
- `configs/.deprecated/` - Removed
- `configs/metrics/` - Consolidated into `evaluation/`
- `configs/extras/` - Inlined into `base.yaml`
- `configs/hardware/` - Moved to `trainer/`
- `configs/preset/` - Completely eliminated
- `configs/tools/` - Moved to `.vscode/`

## Phases 5-8 Summary

| Phase | Duration | Files | Dirs | Cognitive Load | Key Achievement |
|-------|----------|-------|------|----------------|-----------------|
| **Phase 5** | 2-4h | 102 → 90 | -4 | 7.0 → 6.5 | Removed cruft |
| **Phase 6** | 4-6h | 90 → 90 | -3 | 6.5 → 5.8 | Unified data configs |
| **Phase 7** | 8-12h | 90 → 90 | -1 | 5.8 → 4.5 | Unified model configs |
| **Phase 8** | 2-3h | 90 → 89 | -2 | 4.5 → 4.0 | Final cleanup |
| **Total** | ~20h | **102 → 89** | **-10** | **7.0 → 4.0** | **43% improvement** |

## How to Use This Document

1. **Understanding the system?** Start with "Overview" and "Current Directory Structure"
2. **Adding new configs?** Reference "Key Organizational Principles"
3. **Migrating old code?** Check "Migration Guide"
4. **Understanding config composition?** See "Training Run Merge Trace"

---

**Last Updated**: 2025-12-11
**Status**: Phases 5-8 Complete
**Next Review**: As needed for future optimizations
