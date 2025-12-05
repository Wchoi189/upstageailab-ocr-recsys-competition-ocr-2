# Implementation Plan: Hydra Configuration System Overhaul

**Status**: Phase 2 Complete - Validation In Progress
**Created**: 2025-12-04 21:00 UTC
**Updated**: 2025-12-05 00:40 UTC
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

# Progress Tracker

## Completed Phases

| Phase/Task | Status | Completion Date |
|------------|--------|-----------------|
| Immediate Fix: Inference Config Loader | ✅ Completed | 2025-12-04 |
| Phase 0.1: Archive Legacy Configs | ✅ Completed | 2025-12-04 |
| Phase 0.2: Create CONFIG_ARCHITECTURE.md | ✅ Completed | 2025-12-04 |
| Phase 0.3: Add Config Validation Script | ✅ Completed | 2025-12-04 |
| Phase 0.4: Update Makefile | ✅ Completed | 2025-12-04 |
| Phase 1.1: Consolidate Logger Configs | ✅ Completed | 2025-12-04 |
| Phase 1.2: Inline Single-File Groups | ✅ Completed | 2025-12-04 |
| Phase 1.3: Analyze @package Usage | ✅ Completed | 2025-12-04 |
| Phase 1.4: Update References | ✅ Completed | 2025-12-04 |
| Phase 2.1: Create _base/ Directory | ✅ Completed | 2025-12-05 |
| Phase 2.2: Create train_v2.yaml | ✅ Completed | 2025-12-05 |
| Phase 2.3: Migrate train.yaml | ✅ Completed | 2025-12-05 |
| Phase 2.4: Fix Hydra Config Issues | ✅ Completed | 2025-12-05 |

## Current Status

**Phase**: Integration & Testing
**Status**: 🔄 In Progress
**Blocker**: Training validation test interrupted during PyTorch initialization

## What Was Accomplished (Phase 2)

### Files Created
- `configs/_base/core.yaml` - Core experiment configuration (no hydra conflicts)
- `configs/_base/model.yaml` - Model defaults (@package model)
- `configs/_base/data.yaml` - Data consolidation (@package _global_)
- `configs/_base/trainer.yaml` - Lightning trainer settings (@package trainer)
- `configs/_base/logging.yaml` - Logger configuration (@package logger)
- `configs/train_v2.yaml` - Parallel entry point demonstrating new structure

### Files Modified
- `configs/train.yaml` - Updated to use _base/ structure with explicit hydra include
- `configs/hydra/default.yaml` - Added `mode: RUN` field to fix Hydra errors
- `ocr/utils/config_utils.py` - Fixed nest_at_path() for @package _global_ (Phase 1)
- `ui/utils/inference/config_loader.py` - Fixed inference loader (Immediate Fix)

### Critical Fixes
1. **_base/core.yaml**: Removed hydra section that conflicted with /hydra: default
2. **_base/data.yaml**: Changed @package from `data` → `_global_`
3. **hydra/default.yaml**: Added missing `mode: RUN` field
4. **train.yaml**: Simplified defaults list, explicitly includes /hydra: default

### Validation Status
✅ Config syntax validation passes (scripts/validate_config.py)
✅ Custom load_config() loads all sections correctly
✅ Inference config loader works with new structure
✅ Hydra initialization succeeds (progresses to PyTorch loading)
⏳ Full training run pending (interrupted during setup)

## What's Next (Immediate Actions)

### 1. Complete Training Validation Test
**Goal**: Verify training can complete 1 epoch with new config structure
**Command**:
```bash
python runners/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=0.25 \
  trainer.limit_val_batches=0.25 \
  exp_name=phase2_validation
```
**Expected Result**: Training completes without config errors, checkpoint saved

### 2. Update Documentation
**Files to update**:
- `docs/CONFIG_ARCHITECTURE.md` - Add Phase 2 section with new _base/ structure
- `docs/PHASE_2_COMPLETION_SUMMARY.md` - Already created, needs minor updates
- `README.md` - Update config examples if needed

### 3. Optional: Archive Old Structure (Phase 3)
**When ready**: Move old nested configs to `configs/.deprecated_phase2/`
**Benefit**: Reduce from 102 → ~70 files, cleaner structure
**Risk**: Low (both structures work in parallel currently)

## Metrics Progress

| Metric | Before | After Phase 2 | Target (Phase 3) |
|--------|--------|---------------|------------------|
| Config files | 96 | 102* | 60-70 |
| Config groups | 21 | 22* | 15-17 |
| Defaults chain | 28-30 files | ~20-25 files | 12-15 files |
| @package targets | 11 | 6-7 | 3-4 |
| Cognitive load | 7.1/10 | ~5.5/10 | 4.2/10 |

*Temporary increase due to both old and new structures present during transition

---## Executive Summary

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
