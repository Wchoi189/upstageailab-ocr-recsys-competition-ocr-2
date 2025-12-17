---
type: architecture
component: configuration
status: current
version: "2.0"
last_updated: "2025-12-15"
---

# Configuration Architecture

**Purpose**: Hydra-based configuration system with unified data/model hierarchies; 89 YAML files, 17 directories, 20-25 file merge per training run.

---

## Metrics

| Metric | Current | Baseline | Improvement |
|--------|---------|----------|-------------|
| **Total Files** | 89 | 102 | 13% reduction |
| **Active Directories** | 17 | 27 | 37% reduction |
| **Files/Training Merge** | 20-25 | 28-30 | 17% reduction |
| **Cognitive Load Score** | 4.0/10 | 7.0/10 | 43% reduction |

---

## Directory Structure

```
configs/
├── _base/                    # Base templates
├── base.yaml                 # Root config
├── data/                     # ALL data configs (20 files)
│   ├── dataloaders/          # DataLoader configs (2)
│   ├── datasets/             # Dataset preprocessing (4)
│   ├── transforms/           # Transformations (3)
│   └── performance_preset/   # Performance opts (6)
├── model/                    # ALL model configs (22 files)
│   ├── architectures/        # Full architectures (3)
│   ├── encoder/              # Encoders (2)
│   ├── decoder/              # Decoders (5)
│   ├── head/                 # Heads (3)
│   ├── loss/                 # Losses (2)
│   ├── presets/              # Compositions (3)
│   ├── lightning_modules/    # Lightning modules (1)
│   └── optimizers/           # Optimizers (2)
├── trainer/                  # Trainer configs (4)
├── callbacks/                # Callbacks (8)
├── logger/                   # Loggers (4)
├── paths/                    # Paths (1)
├── hydra/                    # Hydra framework (2)
├── evaluation/               # Metrics (1)
├── benchmark/                # Benchmarking (1)
├── debug/                    # Debug configs (1)
├── ui/                       # UI configs (5)
├── ui_meta/                  # UI metadata (6)
└── [entry configs]           # train.yaml, test.yaml, predict.yaml, etc. (10)
```

---

## Config Merge Order (Training)

| Step | File | Package Target | Purpose |
|------|------|----------------|---------|
| 1 | `train.yaml` | root | Entry point |
| 2 | `base.yaml` | `@package _global_` | Root defaults |
| 3 | `data/default.yaml` | root | Data defaults |
| 4 | `data/transforms/base.yaml` | `@package data.transforms` | Transforms |
| 5 | `data/dataloaders/default.yaml` | `@package data.dataloaders` | DataLoader |
| 6 | `model/default.yaml` | root | Model defaults |
| 7 | `model/architectures/dbnet.yaml` | `@package model` | Architecture |
| 8 | `model/optimizers/adamw.yaml` | `@package model.optimizer` | Optimizer |
| 9 | `model/presets/model_example.yaml` | root | Model preset |
| 10-13 | encoder/decoder/head/loss | `@package model.*` | Components |
| 14 | `model/lightning_modules/base.yaml` | `@package _global_` | Lightning module |
| 15 | `trainer/default.yaml` | `@package trainer` | Trainer |
| 16 | `callbacks/default.yaml` | `@package callbacks` | Callbacks |
| 17 | `logger/consolidated.yaml` | `@package logger` | Logger |
| 18 | `paths/default.yaml` | `@package paths` | Paths |
| 19 | `evaluation/metrics.yaml` | `@package _global_` | Metrics |
| 20-25 | Nested defaults | Various | Additional overrides |

**Total Merge**: 20-25 files (reduced from 28-30)

---

## Package Directive Reference

| Target | Files | Purpose |
|--------|-------|---------|
| `@package _global_` | ~10 | Root merging (reduced from 18) |
| `@package model` | 3 | Nest under `model` |
| `@package model.encoder` | 2 | Nest under `model.encoder` |
| `@package model.decoder` | 5 | Nest under `model.decoder` |
| `@package model.head` | 3 | Nest under `model.head` |
| `@package model.loss` | 2 | Nest under `model.loss` |
| `@package data` | 5 | Nest under `data` |
| `@package callbacks` | 8 | Nest under `callbacks` |
| `@package logger` | 4 | Nest under `logger` |
| `@package trainer` | 4 | Nest under `trainer` |

---

## Usage Patterns

### Training
```bash
# Basic training
uv run python runners/train.py preset=<name>

# Override parameters
uv run python runners/train.py model.optimizer.lr=0.0005 data.batch_size=16

# Switch architectures
uv run python runners/train.py model.architecture=east
```

### Inference (UI/API)
- Entry: `train.yaml` (from checkpoint metadata)
- Merge: 20-25 files (same as training)
- Validation: `docs/schemas/ui_inference_compat.yaml`

### Evaluation
- Entry: `configs/test.yaml`
- Merge: 18-22 files

---

## Dependencies

| Component | Imports | Config Files |
|-----------|---------|--------------|
| **Training** | PyTorch, Hydra | 20-25 merged |
| **Inference** | Checkpoint metadata | 20-25 from checkpoint |
| **Evaluation** | PyTorch, Hydra | 18-22 merged |

---

## Constraints

- **Single Source of Truth**: All data configs in `data/`; all model configs in `model/`
- **Clear Hierarchy**: Component configs nested under parent directories
- **Separation of Concerns**: Hydra configs in `configs/`; tool configs in `.vscode/`; schemas in `docs/schemas/`; research in `docs/research/`

---

## Backward Compatibility

**Status**: Maintained for config structure and package targets

**Breaking Changes** (Phases 5-8):
- Removed: `configs/.deprecated/`, `metrics/`, `extras/`, `hardware/`, `preset/`, `tools/`
- Moved: `dataloaders/` → `data/dataloaders/`; `transforms/` → `data/transforms/`
- Consolidated: `preset/models/` → `model/encoder|decoder|head|loss/`

**Migration Path**:

| Old Path | New Path |
|----------|----------|
| `configs/dataloaders/` | `configs/data/dataloaders/` |
| `configs/transforms/` | `configs/data/transforms/` |
| `configs/preset/datasets/` | `configs/data/datasets/` |
| `configs/preset/models/encoder/` | `configs/model/encoder/` |
| `configs/preset/models/decoder/` | `configs/model/decoder/` |
| `configs/preset/models/head/` | `configs/model/head/` |
| `configs/preset/models/loss/` | `configs/model/loss/` |
| `configs/preset/models/*.yaml` | `configs/model/presets/` |
| `configs/preset/lightning_modules/` | `configs/model/lightning_modules/` |
| `configs/tools/*.json` | `.vscode/*.json` |

**Compatibility Matrix**:

| Interface | v1.x | v2.0 | Notes |
|-----------|------|------|-------|
| Config structure | ✅ | ✅ | Directory hierarchy stable |
| Package targets | ✅ | ✅ | No changes to `@package` directives |
| Entry points | ✅ | ✅ | `train.yaml`, `test.yaml` unchanged |

---

## Change History (Phases 5-8)

| Phase | Files | Dirs | Cognitive Load | Achievement |
|-------|-------|------|----------------|-------------|
| Phase 5 | 102 → 90 | -4 | 7.0 → 6.5 | Removed deprecated dirs |
| Phase 6 | 90 → 90 | -3 | 6.5 → 5.8 | Unified data configs |
| Phase 7 | 90 → 90 | -1 | 5.8 → 4.5 | Unified model configs |
| Phase 8 | 90 → 89 | -2 | 4.5 → 4.0 | Final cleanup |
| **Total** | **-13** | **-10** | **-43%** | **43% improvement** |

---

## References

- [System Architecture](system-architecture.md)
- [Backward Compatibility](backward-compatibility.md)
