# Implementation Plan: Hydra Configuration System Overhaul

**Status**: Ready for Implementation
**Created**: 2025-12-04 21:00 UTC
**Scope**: Full Overhaul (Phase 0 + 1 + 2)
**Estimated Duration**: 26+ hours
**Priority**: P0 (Blocks inference functionality & AI agent productivity)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Assessment Reference](#current-assessment-reference)
3. [Immediate Fix (Blocker)](#immediate-fix-blocker)
4. [Phase 0: Archive & Document](#phase-0-archive--document)
5. [Phase 1: Reduce @package Usage](#phase-1-reduce-package-usage)
6. [Phase 2: Flatten & Restructure](#phase-2-flatten--restructure)
7. [Integration & Validation](#integration--validation)
8. [Rollback Plan](#rollback-plan)
9. [Success Metrics](#success-metrics)

---

## Executive Summary

### Problem Statement

1. **Immediate Blocker**: Inference engine fails with "Configuration missing direct 'model' section" because `ui/utils/inference/config_loader.py` uses broken Hydra compose API that silently fails to resolve defaults.

2. **Long-term Issue**: Current config system has cognitive load score of **7.1/10 (HIGH)** with:
   - 80 YAML files across 17 groups
   - 28-30 files merged per training run
   - 11 different @package targets
   - 6 levels of nesting
   - 10+ interdependent variables
   - **Result**: AI agents cannot maintain context; developers struggle with debugging

### Proposed Solution

**Three-phase aggressive restructuring:**

- **Immediate (Fix)**: Replace broken Hydra compose with working `load_config()` wrapper (1-2 hours)
- **Phase 0**: Archive unused configs, create CONFIG_ARCHITECTURE.md, add validation (2 hours)
- **Phase 1**: Reduce @package targets from 11 → 5, consolidate single-file groups (8 hours)
- **Phase 2**: Flatten 80 files/17 groups → 35 files/9 groups, reduce nesting 6 → 3 levels (16 hours)

**Expected Outcomes**:
- ✅ Inference engine works immediately
- ✅ Cognitive load reduces from 7.1 → 4.2/10
- ✅ Config files reduce from 80 → 35
- ✅ Defaults chain reduces from 28-30 → 12-15 files
- ✅ @package targets reduce from 11 → 3-4
- ✅ Single source of truth documentation

### Assessment Reference

For detailed complexity analysis, see: `.cursor/plans/2025-12-04_1900_assessments_Hydra Configuration Architecture Complexity.md`

Key metrics from assessment:
- Cognitive Load: **7.1/10 (HIGH)** → Target: 4.2/10
- Config Files: **80** → Target: 35
- Config Groups: **17** → Target: 9
- @package Targets: **11** → Target: 3-4
- Max Nesting: **6 levels** → Target: 3 levels
- Files per Merge: **28-30** → Target: 12-15

---

## Immediate Fix (Blocker)

### Issue Summary

**Error**: "Configuration missing direct 'model' section, trying root level extraction"

**Root Cause**: `ui/utils/inference/config_loader.py` lines 106-129 attempt to resolve Hydra defaults using compose API, which fails silently, leaving config unresolved.

**Impact**:
- Inference endpoints return 500 errors
- Frontend cannot load checkpoints
- UI completely broken

### Solution Overview

Replace Hydra compose with wrapper around `ocr/utils/config_utils.py::load_config()` which:
- Uses manual OmegaConf merging (proven to work)
- Properly handles @package directives
- Resolves variable interpolation
- Returns correctly structured config with `model` section

### Implementation

**File**: `ui/utils/inference/config_loader.py`

**Changes**:
1. Remove broken Hydra compose code (lines 106-129)
2. Add import: `from ocr.utils.config_utils import load_config as load_hydra_config`
3. Create wrapper function `load_config_from_path()` that:
   - Accepts full file path (e.g., `/path/to/train.yaml`)
   - Extracts config name (e.g., `train`)
   - Calls `load_hydra_config(config_name)`
   - Returns properly resolved config

**Expected Result**:
- `bundle.raw_config` will have `model` section with encoder/decoder/head/loss
- Inference endpoints work
- No more 500 errors

**Code sketch**:
```python
def load_config_from_path(config_path: str | Path) -> DictConfig:
    """
    Load Hydra config from full path by extracting config name.

    Uses ocr/utils/config_utils.py::load_config() which properly
    resolves defaults and @package directives.
    """
    path = Path(config_path).resolve()

    # Extract config name (filename without .yaml)
    config_name = path.stem

    # Use proven load_config() function
    try:
        from ocr.utils.config_utils import load_config as load_hydra_config
        cfg = load_hydra_config(config_name=config_name)
        return cfg
    except Exception as e:
        LOGGER.error(f"Failed to load config from {config_path}: {e}")
        raise
```

**Testing**:
```bash
# Restart backend
make backend-stop
make backend-start

# Verify inference works
curl http://localhost:8000/api/inference/checkpoints?limit=10

# Should return 200 with checkpoint list, not 500 errors
```

---

## Phase 0: Archive & Document

**Duration**: ~2 hours
**Risk**: Low (non-breaking changes)
**Benefit**: High (establishes baseline for Phase 1-2)

### Objectives

1. Archive unused/legacy configs to reduce visible complexity
2. Create authoritative CONFIG_ARCHITECTURE.md as single source of truth
3. Add validation script to detect config issues early

### 0.1: Archive Legacy Configs

**Current State**:
```
configs/
├── schemas/          # 4 files - appear to be old snapshots
├── benchmark/        # 1 file - unused
├── tools/           # empty directory
├── ... (other 14 live groups)
```

**Action**: Move to `.deprecated/` with explanatory README

```bash
mkdir -p configs/.deprecated/schemas
mkdir -p configs/.deprecated/benchmark
mkdir -p configs/.deprecated/tools

# Move files
mv configs/schemas/*.yaml configs/.deprecated/schemas/
mv configs/benchmark/*.yaml configs/.deprecated/benchmark/
mv configs/tools/* configs/.deprecated/tools/ 2>/dev/null || true

# Create README explaining why
cat > configs/.deprecated/README.md << 'EOF'
# Deprecated Configurations

These configurations have been archived and are no longer used in the main training pipeline.

## schemas/ (4 files)
- **Reason**: Appear to be snapshot copies of config structure, not referenced anywhere
- **Benefit of removal**: Eliminates confusion about which schema is authoritative
- **Recovery**: All configs still functional without these files

## benchmark/ (1 file)
- **Reason**: Unused in current training/evaluation pipeline
- **Benefit of removal**: Clarifies what configs are actively used
- **Recovery**: Can be restored if needed for benchmarking work

## tools/ (empty)
- **Reason**: Placeholder directory, no configs
- **Benefit of removal**: Reduces clutter for new users

## Restoring Archived Configs

If you need a config from this directory:
1. Copy it back to the appropriate location in configs/
2. Update train.yaml defaults to include it
3. Test with a training run

EOF
```

**Impact**: Reduces visible config count from 80 → 67 files (13 files removed)

### 0.2: Create CONFIG_ARCHITECTURE.md

**File**: `docs/CONFIG_ARCHITECTURE.md`

**Purpose**: Single authoritative source documenting the entire config system

**Content Structure**:

```markdown
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
```

**Effort**: ~2 hours to create and validate document

### 0.3: Add Config Validation Script

**File**: `scripts/validate_config.py` (NEW)

**Purpose**: Detect config issues early (missing variables, broken @package, etc.)

**Features**:
1. Load all config files and check for syntax errors
2. Verify all `@package` targets are valid
3. Check for undefined variable references
4. Detect orphaned configs (not referenced anywhere)
5. Report @package collision risks

**Code sketch**:
```python
#!/usr/bin/env python3
"""Validate Hydra configuration system for common issues."""

import sys
from pathlib import Path
from omegaconf import OmegaConf
import yaml

CONFIG_ROOT = Path('configs')

def validate_syntax():
    """Check all YAML files are valid."""
    errors = []
    for yaml_file in CONFIG_ROOT.rglob('*.yaml'):
        try:
            with open(yaml_file) as f:
                yaml.safe_load(f)
        except Exception as e:
            errors.append(f"Syntax error in {yaml_file}: {e}")
    return errors

def validate_packages():
    """Check all @package directives are valid."""
    valid_packages = {
        '_global_', 'model', 'model.encoder', 'model.decoder',
        'model.head', 'model.loss', 'model.optimizer', 'model.scheduler',
        'callbacks', 'logger', 'logger.wandb', 'data', 'trainer'
    }
    errors = []
    for yaml_file in CONFIG_ROOT.rglob('*.yaml'):
        with open(yaml_file) as f:
            for i, line in enumerate(f, 1):
                if '# @package' in line:
                    pkg = line.split('@package')[-1].strip()
                    if pkg not in valid_packages:
                        errors.append(f"{yaml_file}:{i} - Unknown @package: {pkg}")
    return errors

def validate_variables():
    """Check for undefined variable references."""
    # Load base config to get defined variables
    from ocr.utils.config_utils import load_config
    try:
        cfg = load_config('train')
        # If this succeeds, variables are resolved
        return []
    except Exception as e:
        return [f"Variable resolution failed: {e}"]

def main():
    print("Validating Hydra configuration system...\n")

    all_errors = []
    all_errors.extend(validate_syntax())
    all_errors.extend(validate_packages())
    all_errors.extend(validate_variables())

    if all_errors:
        print("❌ Validation failed:")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print("✅ All config validations passed")
    return 0

if __name__ == '__main__':
    sys.exit(main())
```

**Usage**:
```bash
python scripts/validate_config.py

# Can be added to CI/CD:
# scripts/validate_config.py || exit 1
```

**Benefit**: Catches config issues before they break training/inference

### 0.4: Update Makefile with Phase 0 Commands

**New targets**:
```makefile
.PHONY: config-validate
config-validate:  ## Validate configuration system for errors
	python scripts/validate_config.py

.PHONY: config-archive
config-archive:   ## Archive legacy configs to .deprecated/
	mkdir -p configs/.deprecated/schemas configs/.deprecated/benchmark configs/.deprecated/tools
	mv configs/schemas/*.yaml configs/.deprecated/schemas/ 2>/dev/null || true
	mv configs/benchmark/*.yaml configs/.deprecated/benchmark/ 2>/dev/null || true
	mv configs/tools/* configs/.deprecated/tools/ 2>/dev/null || true
	@echo "✅ Legacy configs archived to configs/.deprecated/"

.PHONY: config-show-structure
config-show-structure:  ## Show current config structure
	find configs -name "*.yaml" -type f | wc -l | xargs echo "Total config files:"
	find configs -mindepth 1 -maxdepth 1 -type d | wc -l | xargs echo "Config groups:"
	grep -r "@package" configs --include="*.yaml" | cut -d: -f2 | sort | uniq -c | sort -rn
```

**Phase 0 Success Criteria**:
- ✅ Legacy configs archived (67 → 80 files)
- ✅ CONFIG_ARCHITECTURE.md created and complete
- ✅ `config-validate` script works without errors
- ✅ All training runs still work with new structure
- ✅ Inference loads checkpoints without errors

---

## Phase 1: Reduce @package Usage

**Duration**: ~8 hours
**Risk**: Medium (requires careful defaults testing)
**Benefit**: Reduces @package targets from 11 → 5, clearer config structure

### Objectives

1. Consolidate logger configs (wandb + csv → single config)
2. Reduce @package _global_ usage by using explicit nesting
3. Inline single-file configuration groups
4. Target: 60 files / 14 groups / 5 @package targets

### 1.1: Consolidate Logger Configurations

**Current State**:
```
configs/logger/
├── default.yaml      (@package _global_)
├── wandb.yaml        (@package logger.wandb)
└── csv.yaml          (@package logger.csv)
```

**Issue**: Scattered across 3 files with unclear relationships

**Action**: Merge into unified structure

```yaml
# New: configs/logger/default.yaml
# @package logger

console:
  level: INFO

csv:
  save_dir: ${paths.output_dir}/logs
  filename: "${now:%Y-%m-%d}/${now:%H-%M-%S}.log"

wandb:
  enabled: true
  project: ocr-training
  entity: upstage
  id: ${logger.wandb.id}  # From metadata callback
  mode: online
  save_code: true
  log_model: false

# ... other logger configs
```

**Benefit**: Single source of truth for all logging

### 1.2: Reduce @package _global_ Usage

**Current**: 18 configs use `@package _global_` (implicit global merging)

**Plan**: Convert most to explicit nesting

**Example**:
```yaml
# Before: base.yaml (@package _global_)
# Merges at root: cfg = {..., dataset_path: ..., model: {...}}

# After: base.yaml (@package _global_) [keep for compatibility]
# But only include essential root-level configs

# Sub-configs use explicit nesting:
# configs/model/defaults/base.yaml (@package model)
# configs/data/defaults/base.yaml (@package data)
```

**Benefit**: Reduces silent merging, makes config flow explicit

### 1.3: Inline Single-File Groups

**Current**: Several groups have only 1 file

```
configs/metrics/         → 1 file
configs/transforms/      → 1 file
configs/extras/          → 1 file
configs/hardware/        → 1 file
```

**Action**: Move single files into parent groups

```
configs/metrics/default.yaml          → configs/model/metrics.yaml
configs/transforms/default.yaml       → configs/data/transforms.yaml
configs/extras/default.yaml           → configs/base_extras.yaml
configs/hardware/default.yaml         → configs/trainer/hardware.yaml
```

**Benefit**: Reduces groups from 17 → 12

### 1.4: Plan File Reorganization

Create detailed migration mapping:

```markdown
# Phase 1 File Migration Plan

## Logger Consolidation
- Delete: logger/csv.yaml, logger/wandb.yaml
- Create: logger/default.yaml (merged config with _global_ → logger)
- Update: base.yaml includes logger/default.yaml

## Single-File Group Inlining
- configs/metrics/default.yaml → configs/model/metrics.yaml
- configs/transforms/default.yaml → configs/data/transforms.yaml
- configs/extras/default.yaml → configs/base/extras.yaml
- configs/hardware/default.yaml → configs/trainer/hardware.yaml
- Delete: metrics/, transforms/, extras/, hardware/ directories

## @package _global_ Reduction
- Review: All 18 @package _global_ configs
- Decision: Keep only 3-4 essential (base.yaml, paths.yaml)
- Convert: Others to explicit @package model, @package data, etc.

## Expected Results
- Files: 67 → 60
- Groups: 16 → 12
- @package targets: 11 → 6
```

### 1.5: Execution Steps

1. **Create new configs** with consolidated structure
2. **Update train.yaml defaults** to reference new structure
3. **Run validation**: `make config-validate`
4. **Test training run**: `python train.py +exp=test trainer.max_epochs=1`
5. **Verify inference**: Ensure checkpoint loading still works
6. **Delete old configs** once verified
7. **Update documentation**: CONFIG_ARCHITECTURE.md

**Phase 1 Success Criteria**:
- ✅ 60 files / 12 groups (down from 67 / 16)
- ✅ @package targets reduced from 11 → 6
- ✅ All validation checks pass
- ✅ Training runs successfully with new structure
- ✅ Inference loads checkpoints correctly
- ✅ CONFIG_ARCHITECTURE.md updated

---

## Phase 2: Flatten & Restructure

**Duration**: ~16 hours
**Risk**: High (major restructuring)
**Benefit**: Major cognitive load reduction, target: 35 files / 9 groups / 4.2 cognitive load

### Objectives

1. Eliminate deeply nested structures (6 → 3 levels)
2. Consolidate related configs into unified groups
3. Simplify variable interpolation
4. Target: 35 files / 9 groups / 3-4 @package targets

### 2.1: New Architecture Design

**Current Structure** (Hierarchical):
```
configs/
├── train.yaml → base.yaml → model/default.yaml → model/architectures/dbnet.yaml
├── preset/ → multiple architecture variants
├── model/optimizers/
├── model/schedulers/
└── ... (scattered across 17 groups)
```

**Proposed Structure** (Flattened):
```
configs/
├── _base_/              # Core configuration
│   ├── default.yaml     # All @package _global_ settings (paths, etc.)
│   ├── model.yaml       # All model base config
│   ├── data.yaml        # All data config
│   ├── trainer.yaml     # All trainer config
│   └── logging.yaml     # All logging config
├── architectures/       # Model architectures
│   ├── dbnet.yaml
│   ├── craft.yaml
│   └── ...
├── backbones/           # Encoder options
│   ├── timm_backbone.yaml
│   ├── craft_vgg.yaml
│   └── ...
├── decoders/            # Decoder options
│   ├── ppm.yaml
│   └── ...
├── heads/               # Head options
│   ├── dbnet_head.yaml
│   └── ...
├── losses/              # Loss function options
│   ├── dbnet_loss.yaml
│   └── ...
├── optimizers/          # Optimizer options
│   ├── adamw.yaml
│   └── ...
├── schedulers/          # LR scheduler options
│   ├── warmup_cosine.yaml
│   └── ...
└── presets/             # Combined presets
    ├── dbnet_base.yaml
    ├── craft_baseline.yaml
    └── ...
```

**Key Changes**:
- Max nesting depth: 6 → 2 levels
- Groups reduced: 17 → 9
- Files reduced: 67 → 35
- @package targets: 6 → 3-4

### 2.2: Core Configuration Groups (_base_/)

**File**: `configs/_base_/default.yaml`
```yaml
# @package _global_

# All root-level paths and settings
paths:
  project_root: ${oc.env:PWD}
  data_dir: ${paths.project_root}/data
  output_dir: ${paths.project_root}/outputs
  checkpoint_dir: ${paths.output_dir}/checkpoints

exp_name: ocr_training
seed: 42

# Include architecture-specific configs
defaults:
  - _self_
  - architectures/dbnet
  - backbones/timm_backbone
  - decoders/ppm
  - heads/dbnet_head
  - losses/dbnet_loss
  - optimizers/adamw
  - schedulers/warmup_cosine
```

**File**: `configs/_base_/model.yaml`
```yaml
# @package model

architecture_name: dbnet
encoder: ${backbones.timm_backbone}
decoder: ${decoders.ppm}
head: ${heads.dbnet_head}
loss: ${losses.dbnet_loss}
optimizer: ${optimizers.adamw}
scheduler: ${schedulers.warmup_cosine}
```

**File**: `configs/_base_/data.yaml`
```yaml
# @package data

dataset:
  name: doctr_synthetic
  path: ${paths.data_dir}/synthetic

train:
  batch_size: 32
  num_workers: 4

val:
  batch_size: 64
  num_workers: 4
```

**File**: `configs/_base_/trainer.yaml`
```yaml
# @package trainer

max_epochs: 100
val_check_interval: 0.1
check_val_every_n_epoch: 1

# Callbacks, logging, etc. all in one place
```

**File**: `configs/_base_/logging.yaml`
```yaml
# @package logger

level: INFO

wandb:
  enabled: true
  project: ocr

csv:
  enabled: true
```

### 2.3: Component Groups (Architecture Options)

**File**: `configs/architectures/dbnet.yaml`
```yaml
# @package model.architecture

name: dbnet
version: 1.0
pretrained: false

# All architecture-specific settings
```

**File**: `configs/backbones/timm_backbone.yaml`
```yaml
# @package model.encoder

_target_: ocr.models.encoders.TimmBackbone
model_name: resnet50
pretrained: true
```

**File**: `configs/optimizers/adamw.yaml`
```yaml
# @package model.optimizer

_target_: torch.optim.AdamW
lr: 0.001
betas: [0.9, 0.999]
```

### 2.4: Presets for Common Configurations

**File**: `configs/presets/dbnet_base.yaml`
```yaml
# Preset: DBNet with ResNet50 backbone

defaults:
  - /architectures/dbnet
  - /backbones/timm_backbone
  - /decoders/ppm
  - /heads/dbnet_head
  - /losses/dbnet_loss
  - /optimizers/adamw
  - /schedulers/warmup_cosine
```

**Usage**: `python train.py preset=dbnet_base` (simpler than current nested defaults)

### 2.5: Migration Strategy

**Phase 2 consists of these sub-steps**:

1. **Design new structure** (2 hours)
   - Create all new config files with proper nesting
   - Map old variables to new structure
   - Plan variable interpolation

2. **Create new configs** (4 hours)
   - Build `_base_/` directory with 5 files
   - Create 9 component groups (architectures, backbones, decoders, etc.)
   - Create presets for common configurations
   - Validate all new configs load without errors

3. **Migrate training code** (2 hours)
   - Update `@hydra.main` to load from new structure
   - Ensure model instantiation still works
   - Verify checkpoint saving/loading compatibility

4. **Create migration guide** (1 hour)
   - Document old → new config mapping
   - Provide examples for CLI overrides
   - Update training documentation

5. **Test migration** (4 hours)
   - Run training with new configs (1 epoch)
   - Verify checkpoint loading
   - Test all CLI override patterns
   - Validate inference still works

6. **Archive old structure** (1 hour)
   - Move old configs to `.deprecated_phase1/`
   - Keep as reference during transition
   - Can be deleted after stabilization period

7. **Update documentation** (2 hours)
   - Update CONFIG_ARCHITECTURE.md with new structure
   - Update CONTRIBUTING.md for new config patterns
   - Update README with simplified examples

### 2.6: Variable Interpolation Simplification

**Current** (complex):
```yaml
dataset_path: ${paths.data_dir}/synthetic
dataset_module: ocr.datasets.doctr_synthetic
encoder_path: ${...}  # Deeply nested
```

**New** (direct references):
```yaml
# In _base_/model.yaml
encoder: ${backbones.timm_backbone}  # Direct reference to config

# In _base_/default.yaml
defaults:
  - backbones/timm_backbone  # Loaded directly
```

**Benefit**: Fewer indirect references, easier to understand

### 2.7: Backward Compatibility

**Requirement**: Old training runs must still work with checkpoints

**Approach**:
- Create compatibility layer that maps old config locations to new
- Old `train.yaml` still works (loads with deprecation warning)
- New `train.yaml` uses new structure
- Config saving in checkpoint is unaffected

**Code sketch**:
```python
# In runners/train.py
if old_config_detected:
    LOGGER.warning("Using legacy config structure. Please migrate to new structure.")
    # Apply compatibility transformations
    cfg = transform_legacy_config(cfg)
```

**Phase 2 Success Criteria**:
- ✅ 35 files / 9 groups (down from 60 / 12)
- ✅ Max nesting depth: 3 (down from 6)
- ✅ @package targets: 3-4 (down from 6)
- ✅ Cognitive load: 4.2/10 (down from 7.1/10)
- ✅ All validation checks pass
- ✅ Training runs successfully with new structure
- ✅ Checkpoints load correctly
- ✅ Inference works
- ✅ Old configs still functional with deprecation warnings
- ✅ CONFIG_ARCHITECTURE.md fully updated

---

## Integration & Validation

**Duration**: ~4 hours
**Risk**: Medium (full system testing required)

### Objectives

1. Verify all phases work correctly together
2. Test backward compatibility
3. Validate performance metrics
4. Document any breaking changes

### Validation Checklist

- [ ] **Config Validation**
  ```bash
  make config-validate
  ```

- [ ] **Training Test (Phase 2)**
  ```bash
  python train.py \
    trainer.max_epochs=1 \
    data.train.batch_size=2
  ```

- [ ] **Checkpoint Loading**
  ```bash
  # Test inference with checkpoint from new configs
  curl http://localhost:8000/api/inference/predict \
    -F "image=@test.jpg" \
    -F "checkpoint=best.ckpt"
  ```

- [ ] **CLI Override Patterns**
  ```bash
  # Old pattern (should work with compatibility layer)
  python train.py model.architecture=craft

  # New pattern (should work with phase 2 structure)
  python train.py architectures/craft
  ```

- [ ] **Metrics Verification**
  ```
  - Files: 67 → 35 (52% reduction)
  - Groups: 16 → 9 (44% reduction)
  - Max depth: 6 → 3 (50% reduction)
  - @package targets: 11 → 3-4 (64-73% reduction)
  - Cognitive load: 7.1 → 4.2 (41% reduction)
  ```

### Test Scenarios

#### Scenario 1: Fresh Training with New Configs
```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
python train.py \
  preset=dbnet_base \
  trainer.max_epochs=1 \
  data.train.batch_size=4 \
  exp_name=test_phase2_new_configs

# Verify:
# - Training completes
# - Checkpoint saved
# - Metadata generated
```

#### Scenario 2: Load Old Checkpoint with New Configs
```bash
# Load checkpoint trained with old configs using new structure
python -c "
from ocr.utils.config_utils import load_config
from ocr.lightning_modules import get_pl_modules_by_cfg

cfg = load_config('train')
print('✅ Config loaded with new structure')
print('Model sections:', list(cfg.model.keys()))
"
```

#### Scenario 3: Inference API
```bash
# Start backend with new configs
make backend-start

# Load checkpoint
curl http://localhost:8000/api/inference/checkpoints?limit=10

# Run inference
curl http://localhost:8000/api/inference/predict \
  -F "image=@test_image.jpg" \
  -F "checkpoint_path=outputs/test_phase2_new_configs/checkpoints/best.ckpt"
```

#### Scenario 4: Legacy Config Compatibility
```bash
# Old configs should still work with warnings
python train.py \
  +exp=legacy_test \
  trainer.max_epochs=1

# Should see deprecation warning in logs
```

### Rollback Plan

**If Phase 2 breaks something**:

1. **Immediate rollback** (5 minutes)
   ```bash
   git checkout feature/outputs-reorg -- configs/
   make config-validate
   ```

2. **Partial rollback** (30 minutes)
   - Keep Phase 0 changes (archiving)
   - Revert Phase 1 + 2
   - Identify specific breaking change
   - Create targeted fix

3. **Gradual rollout** (alternative)
   - Keep both old and new configs in parallel
   - Add env var to select structure: `USE_NEW_CONFIG_STRUCTURE=true`
   - Run both structures in parallel testing
   - Switch when confident

### Performance Validation

**Before Phase 2**:
```bash
time python train.py trainer.max_epochs=1 --help
# Should be <2 seconds
```

**After Phase 2**:
```bash
time python train.py trainer.max_epochs=1 --help
# Should still be <2 seconds (no regression)
```

**Cognitive Load Measurement** (before/after Phase 2):
- Files per merge: 28-30 → 12-15
- Config groups: 16 → 9
- Max nesting: 6 → 3
- @package targets: 6 → 3-4

---

## Rollback Plan

### Level 1: Quick Revert (< 5 minutes)

If inference completely breaks:

```bash
# Revert all config changes
git checkout feature/outputs-reorg -- configs/

# Clear Hydra cache
rm -rf outputs/.hydra
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Restart backend
make backend-restart

# Test inference
curl http://localhost:8000/api/inference/checkpoints?limit=1
```

### Level 2: Partial Rollback (< 30 minutes)

If specific phase breaks:

```bash
# Keep Phase 0 (archiving), revert Phase 1
git checkout feature/outputs-reorg~N -- configs/  # where N is phase boundary

# Identify breaking change
git diff feature/outputs-reorg~N feature/outputs-reorg -- configs/

# Fix specific file
vim configs/[problematic_file].yaml

# Re-test
make config-validate
python train.py trainer.max_epochs=1
```

### Level 3: Parallel Structures (< 1 hour)

If need to maintain both old and new:

```bash
# Rename new structure
mv configs configs_new

# Restore old structure
git checkout main -- configs

# Create env-based loader
# In ocr/utils/config_utils.py:
import os
if os.getenv('USE_NEW_CONFIG_STRUCTURE'):
    CONFIG_ROOT = Path('configs_new')
else:
    CONFIG_ROOT = Path('configs')
```

### Blocked Signals (When to Rollback)

Automatic rollback triggered if:

- [ ] Inference returns 500 errors (> 5% of requests)
- [ ] Training fails during first epoch
- [ ] Config validation fails (> 1 error)
- [ ] Checkpoint loading fails
- [ ] Performance degrades > 20%

---

## Success Metrics

### Quantitative Metrics (from Assessment)

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Config files | 80 | 35 | Phase 2 |
| Config groups | 17 | 9 | Phase 2 |
| @package targets | 11 | 3-4 | Phase 2 |
| Max nesting depth | 6 | 3 | Phase 2 |
| Files per merge | 28-30 | 12-15 | Phase 2 |
| Variables dependencies | 10+ | 3-4 | Phase 2 |
| **Cognitive Load Score** | **7.1/10** | **4.2/10** | **Phase 2** |

### Qualitative Metrics

- ✅ Single source of truth (CONFIG_ARCHITECTURE.md)
- ✅ Clear variable interpolation path
- ✅ Reduced silent merging (@package _global_)
- ✅ Explicit component selection
- ✅ Easier CLI overrides
- ✅ Improved maintainability
- ✅ Better AI agent comprehension

### Operational Metrics

- ✅ Config validation: 0 errors
- ✅ Training time: <5% regression
- ✅ Inference latency: <5% regression
- ✅ Checkpoint compatibility: 100% (old and new)
- ✅ Test coverage: >80% of config paths

### AI Agent Productivity Metrics

- ✅ Context window requirement: 28 files → 12 files (57% reduction)
- ✅ Tracing complexity: O(6) nesting → O(3) nesting (50% reduction)
- ✅ @package confusion: 11 targets → 3-4 targets (64% reduction)
- ✅ Error diagnosis time: Reduced significantly

---

## Timeline & Dependencies

### Phase Dependencies

```
Immediate Fix (1-2h)
    ↓ (unblocks inference)
Phase 0: Archive & Document (2h)
    ↓ (establishes baseline)
Phase 1: Reduce @package (8h)
    ↓ (simplifies structure)
Phase 2: Flatten & Restructure (16h)
    ↓ (major overhaul)
Integration & Validation (4h)
    ↓
Complete Overhaul (Total: 26+ hours)
```

### Parallel Work Possible

- Phase 0 and Phase 1 can be done sequentially or overlapped
- Phase 2 requires Phase 1 to be mostly complete
- Integration testing happens after Phase 2
- Documentation can be updated throughout

---

## References

- **Assessment Document**: `.cursor/plans/2025-12-04_1900_assessments_Hydra Configuration Architecture Complexity.md`
- **Current Implementation**: `ocr/utils/config_utils.py`
- **Inference Config Loader**: `ui/utils/inference/config_loader.py`
- **Training Entry Point**: `runners/train.py`

---

## Next Steps

1. **Review & Approve**: Review this plan with stakeholders
2. **Start Immediate Fix**: Fix inference engine config loading (1-2 hours)
3. **Execute Phase 0**: Archive configs and create documentation (2 hours)
4. **Decision Point**: Proceed with Phase 1-2 or defer based on priorities
5. **Phase 1 Execution**: Reduce @package targets (8 hours)
6. **Phase 2 Execution**: Flatten & restructure (16 hours)
7. **Validation**: Full integration testing (4 hours)

---

**Status**: Ready for Implementation
**Created**: 2025-12-04 21:00 UTC
**Author**: GitHub Copilot
**Version**: 1.0
