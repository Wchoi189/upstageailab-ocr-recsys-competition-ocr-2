# Hydra Configuration Guide

> [!WARNING]
> **Migration In Progress**: The configuration system is being restructured.
> - `_base/` → `_foundation/` (renamed)
> - `base.yaml` → `_foundation/defaults.yaml` (relocated)
> - Multiple KIE configs → unified `domain/kie.yaml` (consolidated)

This directory contains Hydra configuration files for the OCR project. This guide explains the configuration architecture, override patterns, and how to use configs effectively.

---

## Table of Contents

- [Configuration Architecture](#configuration-architecture)
- [Override Patterns](#override-patterns)
- [Quick Reference](#quick-reference)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)
- [File Organization](#file-organization)

---

## New Structure (Post-Migration)

```
configs/
├── train.yaml              # Universal training entry
├── eval.yaml               # Testing/evaluation
├── predict.yaml            # Inference
├── _foundation/            # Core composition fragments
│   ├── defaults.yaml       # Base defaults (replaces base.yaml)
│   ├── core.yaml          # Core experiment settings
│   ├── data.yaml          # Data composition defaults
│   ├── model.yaml         # Model composition defaults
│   ├── trainer.yaml       # Lightning trainer settings
│   ├── logging.yaml       # Logger configuration
│   └── preprocessing.yaml # Preprocessing settings
├── domain/                 # Multi-domain configs (detection, recognition, kie, layout)
├── model/                  # Architecture components
├── data/                   # Dataset configs
├── training/               # Training infrastructure (callbacks, logger, profiling)
└── __EXTENDED__/           # Edge cases, experiments
```

---

## Configuration Architecture

The project uses a **two-tier configuration system**:

### New Architecture (Primary)

**Foundation Layer**: `configs/_foundation/`
- `core.yaml` - Core experiment settings
- `data.yaml` - Data composition defaults
- `model.yaml` - Model composition defaults
- `trainer.yaml` - Lightning trainer settings
- `logging.yaml` - Logger configuration
- `preprocessing.yaml` - Preprocessing settings

**Entry Points**: Top-level configs that compose the foundation
- `train.yaml` - Training workflows
- `test.yaml` - Testing workflows
- `predict.yaml` - Prediction workflows
- `synthetic.yaml` - Synthetic data generation

**Base Defaults**: `configs/base.yaml` defines which config groups are active by default:
```yaml
defaults:
  - model: default          # Active by default
  - data: default           # Active by default
  - logger: consolidated    # Active by default
  - trainer: default        # Active by default
  # ... etc
```

### Legacy Architecture (Maintenance Mode)

Older standalone configs located in `configs/__LEGACY__/`:
- Superseded by new architecture
- Preserved for reference and backward compatibility
- Not recommended for new projects

---

## Override Patterns

### The Golden Rule

**Is the config in `base.yaml` defaults?**
- ✅ **YES** → Override **WITHOUT** `+` prefix
- ❌ **NO** → Override **WITH** `+` prefix

### Examples

#### Configs in defaults (use WITHOUT `+`):

```bash
# These are in base.yaml defaults, so NO + needed
uv run python runners/train.py logger=wandb                    # ✅ Correct
uv run python runners/train.py data=canonical                  # ✅ Correct
uv run python runners/train.py model=default                   # ✅ Correct
uv run python runners/train.py trainer=fp16_safe              # ✅ Correct
uv run python runners/train.py callbacks=default              # ✅ Correct

# Using + will cause errors:
uv run python runners/train.py +logger=wandb                   # ❌ Error: multiple values
uv run python runners/train.py +data=canonical                 # ❌ Error: multiple values
```

#### Configs NOT in defaults (use WITH `+`):

```bash
# These are NOT in base.yaml defaults, so + IS needed
uv run python runners/train.py +ablation=model_comparison      # ✅ Correct
uv run python runners/train.py +hardware=rtx3060_12gb          # ✅ Correct
uv run python runners/train.py +model/presets=craft            # ✅ Correct
uv run python runners/train.py +data/performance_preset=balanced  # ✅ Correct

# Without + will cause errors:
uv run python runners/train.py ablation=model_comparison       # ❌ Error: ablation not found
uv run python runners/train.py hardware=rtx3060_12gb           # ❌ Error: hardware not found
```

### Nested Config Overrides

For nested configs (e.g., `model/architectures`):
```bash
# If parent (model) is in defaults, nested configs don't need +
uv run python runners/train.py model/architectures=dbnetpp     # ✅ Correct
uv run python runners/train.py model/optimizers=adamw          # ✅ Correct

# Adding new nested groups needs +
uv run python runners/train.py +model/new_feature=value        # ✅ Correct
```

### Parameter Overrides

Direct parameter overrides never need `+`:
```bash
uv run python runners/train.py trainer.max_epochs=10           # ✅ Correct
uv run python runners/train.py seed=123                        # ✅ Correct
uv run python runners/train.py batch_size=16                   # ✅ Correct
```

---

## Quick Reference

### Configs in `base.yaml` defaults (NO + needed):

| Config Group | Override Example | Alternative Options |
|-------------|------------------|---------------------|
| `model` | `model=default` | - |
| `data` | `data=canonical`, `data=craft` | `data=preprocessing` |
| `logger` | `logger=wandb`, `logger=csv` | `logger=default` |
| `trainer` | `trainer=fp16_safe`, `trainer=rtx3060_12gb` | `trainer=hardware_rtx3060_12gb_i5_16core` |
| `callbacks` | `callbacks=default` | - |
| `evaluation` | `evaluation=metrics` | - |
| `paths` | `paths=default` | - |
| `debug` | `debug=default` | - |

### Configs NOT in defaults (+ needed):

| Config Group | Override Example | Purpose |
|-------------|------------------|---------|
| `ablation` | `+ablation=model_comparison` | Ablation studies |
| `hardware` | `+hardware=rtx3060_12gb_i5_16core` | Hardware-specific settings |
| `model/presets` | `+model/presets=craft` | Complete model configurations |
| `data/performance_preset` | `+data/performance_preset=balanced` | Performance optimizations |
| `extraction` | `+extraction=default` | Receipt data extraction |
| `layout` | `+layout=default` | Layout analysis |
| `recognition` | `+recognition=default` | Text recognition |

---

## Common Use Cases

### 1. Training with Different Data Configs

```bash
# Use canonical (rotation-corrected) images
uv run python runners/train.py data=canonical

# Use CRAFT-specific data config
uv run python runners/train.py data=craft

# Use preprocessing data config (legacy)
uv run python runners/train.py data=preprocessing
```

### 2. Changing Model Architecture

```bash
# Use DBNet architecture (default)
uv run python runners/train.py model/architectures=dbnet

# Use DBNet++ architecture
uv run python runners/train.py model/architectures=dbnetpp

# Use CRAFT architecture
uv run python runners/train.py model/architectures=craft

# Use complete CRAFT preset (architecture + optimizer + all components)
uv run python runners/train.py +model/presets=craft
```

### 3. Switching Loggers

```bash
# Use consolidated logger (W&B + CSV) - default
uv run python runners/train.py logger=consolidated

# Use W&B only
uv run python runners/train.py logger=wandb

# Use CSV only
uv run python runners/train.py logger=csv
```

### 4. Hardware-Specific Training

```bash
# Use FP16 safe trainer
uv run python runners/train.py trainer=fp16_safe

# Use RTX 3060 optimized settings
uv run python runners/train.py trainer=rtx3060_12gb

# Use complete hardware preset
uv run python runners/train.py +hardware=rtx3060_12gb_i5_16core
```

### 5. Performance Optimization

```bash
# Enable balanced performance preset
uv run python runners/train.py +data/performance_preset=balanced

# Enable validation-optimized caching (2.5-3x speedup)
uv run python runners/train.py +data/performance_preset=validation_optimized

# Enable memory-efficient mode
uv run python runners/train.py +data/performance_preset=memory_efficient
```

### 6. Combining Multiple Overrides

```bash
# Train with canonical data, CRAFT architecture, W&B logging, FP16
uv run python runners/train.py \
  data=canonical \
  model/architectures=craft \
  logger=wandb \
  trainer=fp16_safe \
  trainer.max_epochs=50 \
  seed=123
```

### 7. Running Multirun Sweeps

```bash
# Sweep over multiple architectures
uv run python runners/train.py -m model/architectures=dbnet,dbnetpp,craft

# Sweep over hyperparameters
uv run python runners/train.py -m trainer.max_epochs=5,10,15 seed=42,123,456

# Sweep with ablation studies
uv run python runners/train.py -m +ablation=model_comparison,batch_size,learning_rate
```

---

## Troubleshooting

### Error: "Multiple values for logger"

**Symptom**:
```
hydra.errors.ConfigCompositionException: Multiple values for logger
```

**Cause**: Using `+` prefix on a config already in defaults

**Fix**: Remove the `+` prefix
```bash
# Wrong:
uv run python runners/train.py +logger=wandb

# Correct:
uv run python runners/train.py logger=wandb
```

### Error: "Could not find ablation"

**Symptom**:
```
hydra.errors.MissingConfigException: Could not find 'ablation'
```

**Cause**: Missing `+` prefix on a config NOT in defaults

**Fix**: Add the `+` prefix
```bash
# Wrong:
uv run python runners/train.py ablation=model_comparison

# Correct:
uv run python runners/train.py +ablation=model_comparison
```

### Error: "Config file not found"

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'configs/data/myconfig.yaml'
```

**Cause**: Config file doesn't exist or path is incorrect

**Fix**: Check available configs
```bash
# List available data configs
ls configs/data/*.yaml

# List available model architectures
ls configs/model/architectures/*.yaml

# List all config groups
ls configs/
```

### How to Check Current Config

View resolved config without running:
```bash
# Show job config
uv run python runners/train.py --cfg job

# Show Hydra config
uv run python runners/train.py --cfg hydra

# Show help
uv run python runners/train.py --help
```

### How to Test Override Patterns

Run the override test suite:
```bash
uv run python tests/unit/test_hydra_overrides.py
```

---

## File Organization

### Directory Structure

```
configs/
├── README.md (this file)
├── base.yaml                    # Base defaults composition
├── train.yaml                   # Training entry point
├── test.yaml                    # Testing entry point
├── predict.yaml                 # Prediction entry point
├── synthetic.yaml               # Synthetic data generation
│
├── _base/                       # Foundation configs
│   ├── core.yaml
│   ├── data.yaml
│   ├── model.yaml
│   ├── trainer.yaml
│   ├── logging.yaml
│   └── preprocessing.yaml
│
├── model/                       # Model configurations
│   ├── default.yaml             # Default model config
│   ├── architectures/           # Architecture definitions
│   │   ├── dbnet.yaml
│   │   ├── dbnetpp.yaml
│   │   └── craft.yaml
│   ├── optimizers/              # Optimizer configs
│   │   ├── adam.yaml
│   │   └── adamw.yaml
│   ├── presets/                 # Complete model presets
│   │   ├── craft.yaml
│   │   └── dbnetpp.yaml
│   ├── encoder/                 # Encoder components
│   ├── decoder/                 # Decoder components
│   ├── head/                    # Head components
│   └── loss/                    # Loss components
│
├── data/                        # Data configurations
│   ├── default.yaml             # Default data config (composition)
│   ├── base.yaml                # Foundation data config
│   ├── canonical.yaml           # Canonical (rotation-corrected) data
│   ├── craft.yaml               # CRAFT-specific data
│   ├── transforms/              # Transform configurations
│   │   ├── base.yaml
│   │   ├── background_removal.yaml
│   │   └── with_background_removal.yaml
│   ├── dataloaders/             # Dataloader configurations
│   │   ├── default.yaml
│   │   └── rtx3060_16core.yaml
│   └── performance_preset/      # Performance optimization presets
│       ├── balanced.yaml
│       ├── memory_efficient.yaml
│       └── validation_optimized.yaml
│
├── logger/                      # Logger configurations
│   ├── consolidated.yaml        # W&B + CSV (default)
│   ├── wandb.yaml               # W&B only
│   ├── csv.yaml                 # CSV only
│   └── default.yaml             # Composition-based logger
│
├── trainer/                     # Trainer configurations
│   ├── default.yaml             # Default trainer settings
│   ├── fp16_safe.yaml           # FP16 safe mode
│   └── rtx3060_12gb.yaml        # Hardware-specific settings
│
├── callbacks/                   # Callback configurations
│   ├── default.yaml             # Default callbacks
│   ├── early_stopping.yaml
│   ├── model_checkpoint.yaml
│   └── wandb_image_logging.yaml
│
├── evaluation/                  # Evaluation configurations
│   └── metrics.yaml
│
├── paths/                       # Path configurations
│   └── default.yaml
│
├── debug/                       # Debug configurations
│   └── default.yaml
│
├── hydra/                       # Hydra-specific configurations
│   ├── default.yaml
│   └── disabled.yaml
│
├── ui/                          # UI configurations (standalone)
│   ├── inference.yaml
│   └── unified_app.yaml
│
├── extraction/                  # Receipt extraction configs
│   └── default.yaml
│
├── layout/                      # Layout analysis configs
│   └── default.yaml
│
├── recognition/                 # Text recognition configs
│   └── default.yaml
│
└── __LEGACY__/                  # Legacy configurations (deprecated)
    ├── README.md                # Legacy migration guide
    ├── model/
    │   └── optimizer.yaml       # Superseded by optimizers/adam.yaml
    └── data/
        └── preprocessing.yaml   # Superseded by new architecture
```

### Config Naming Conventions

- **Group directories**: `configs/{group}/` (e.g., `configs/model/`)
- **Default configs**: `{group}/default.yaml` (e.g., `model/default.yaml`)
- **Variant configs**: `{group}/{variant}.yaml` (e.g., `logger/wandb.yaml`)
- **Nested configs**: `{group}/{subgroup}/{variant}.yaml` (e.g., `model/architectures/dbnet.yaml`)
- **Foundation configs**: `_base/{component}.yaml` (e.g., `_base/model.yaml`)

---

## Additional Resources

### Documentation
- **Override Patterns Reference**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_OVERRIDE_PATTERNS.md`
- **Configuration Audit**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_CONFIG_AUDIT_ASSESSMENT.md`
- **Legacy Migration Guide**: `configs/__LEGACY__/README.md`

### Test Suite
- **Override Tests**: `tests/unit/test_hydra_overrides.py`

### External Resources
- **Hydra Documentation**: https://hydra.cc/docs/intro/
- **Override Grammar**: https://hydra.cc/docs/advanced/override_grammar/basic/
- **Composition Patterns**: https://hydra.cc/docs/advanced/defaults_list/

---

## Configuration Best Practices

### 1. Start with defaults, override as needed
```bash
# Use defaults for most settings
uv run python runners/train.py

# Override only what you need to change
uv run python runners/train.py trainer.max_epochs=50 seed=123
```

### 2. Use presets for common configurations
```bash
# Instead of overriding many individual settings:
uv run python runners/train.py \
  model/architectures=craft \
  model/optimizers=adamw \
  model/encoder=craft_vgg \
  model/decoder=craft_decoder \
  model/head=craft_head \
  model/loss=craft_loss

# Use a preset:
uv run python runners/train.py +model/presets=craft
```

### 3. Leverage performance presets
```bash
# Quick validation runs
uv run python runners/train.py +data/performance_preset=validation_optimized

# Production training
uv run python runners/train.py +data/performance_preset=balanced
```

### 4. Document custom configs
If you create new configs, add comments explaining:
- Purpose of the config
- When to use it
- How it differs from defaults
- Any dependencies or requirements

### 5. Test before committing
Always test new configs with:
```bash
# Dry run to check composition
uv run python runners/train.py --cfg job

# Run override tests
uv run python tests/unit/test_hydra_overrides.py
```

---

## Getting Help

- **Check this README** for common patterns and troubleshooting
- **Run tests**: `uv run python tests/unit/test_hydra_overrides.py`
- **View config**: `uv run python runners/train.py --cfg job`
- **Review audit**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/`
- **Ask team**: Post in project discussions

---

**Last Updated**: 2025-12-24
**Configuration Version**: v2.0 (New Architecture)
