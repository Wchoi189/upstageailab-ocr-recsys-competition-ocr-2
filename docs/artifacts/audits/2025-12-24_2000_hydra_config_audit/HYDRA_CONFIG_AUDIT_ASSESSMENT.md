# Hydra Configuration System Audit Assessment

**Date**: 2025-12-24
**Auditor**: Claude Sonnet 4.5
**Audit Scope**: Complete Hydra configuration architecture analysis
**Status**: Complete - Resolution Required

---

## Executive Summary

This audit analyzed 80+ Hydra configuration files across the OCR project to classify them by architecture type (new vs legacy), document override patterns, identify code references, and assess removal impact. The project has successfully migrated to a new architecture using `configs/_base/` as the foundation, but legacy configurations remain scattered throughout the codebase, creating confusion about which configs are active and what override patterns to use.

### Key Findings

1. **Architecture Split Confirmed**: The project uses TWO distinct configuration architectures
   - **New Architecture**: `configs/_base/` foundation with modular composition (primary system)
   - **Legacy Architecture**: Standalone configs without `_base/` references (maintenance mode)

2. **Override Pattern Clarity**: Override requirements are well-defined but not consistently documented
   - Configs in `base.yaml` defaults → use WITHOUT `+` prefix
   - Configs NOT in defaults → use WITH `+` prefix
   - Current failures stem from using `+` on configs already in defaults

3. **Minimal Legacy Usage**: Most legacy configs have no active code references
   - Only `data/canonical`, `data/craft`, and UI configs show active usage
   - Model presets and specialized configs serve distinct purposes from base configs
   - Many configs can be safely containerized in `__LEGACY__/` without breaking functionality

4. **No True Duplicates**: Previous 2025-11-11 audit confirmed files with similar names serve different purposes
   - Example: `model/optimizer.yaml` vs `model/optimizers/adam.yaml` are complementary, not duplicates

### Recommendations Priority

1. **Immediate**: Document override patterns in project README
2. **Short-term**: Containerize legacy configs in `configs/__LEGACY__/` folder
3. **Medium-term**: Update code references to use new architecture
4. **Long-term**: Deprecate and eventually remove unused legacy configs

---

## Configuration Architecture Analysis

### New Architecture (Primary System)

**Foundation**: `configs/_base/` directory provides the base layer for all configurations.

**Base Configs** (`configs/_base/`):
- `core.yaml` - Core experiment settings (@package _global_)
- `data.yaml` - Data composition defaults (pulls from data/base + transforms + dataloaders)
- `model.yaml` - Model composition defaults (pulls from architectures + optimizers)
- `trainer.yaml` - Lightning trainer settings (@package trainer)
- `logging.yaml` - Logger configuration (@package logger)
- `preprocessing.yaml` - Preprocessing settings

**Entry Points** (top-level configs):
- `train.yaml` - Training entry point (uses _base/ foundation)
- `test.yaml` - Testing entry point
- `predict.yaml` - Prediction entry point
- `synthetic.yaml` - Synthetic data generation
- `performance_test.yaml` - Performance benchmarking
- `cache_performance_test.yaml` - Cache performance testing

**Defaults Composition** (`configs/base.yaml`):
```yaml
defaults:
  - model: default           # Uses configs/model/default.yaml
  - evaluation: metrics      # Uses configs/evaluation/metrics.yaml
  - paths: default          # Uses configs/paths/default.yaml
  - logger: consolidated    # Uses configs/logger/consolidated.yaml
  - trainer: default        # Uses configs/trainer/default.yaml
  - callbacks: default      # Uses configs/callbacks/default.yaml
  - debug: default          # Uses configs/debug/default.yaml
```

**Override Pattern**: Configs listed in `base.yaml` defaults → override WITHOUT `+`
- ✅ `logger=wandb` (logger in defaults)
- ✅ `model=default` (model in defaults)
- ❌ `+logger=wandb` (ERROR: multiple values for logger)

### Legacy Architecture (Maintenance Mode)

**Standalone Configs**: Configs that provide complete configuration without using `_base/` foundation.

**Legacy Data Configs**:
- `data/canonical.yaml` - Standalone data config (no defaults, defines everything)
- `data/craft.yaml` - CRAFT-specific data config (partial defaults: only transforms/base)
- `data/preprocessing.yaml` - Preprocessing data config

**Legacy Model Configs**:
- `model/optimizer.yaml` - Simplified optimizer config (@package model.optimizer)

**UI Configs** (Intentionally Standalone):
- `ui/inference.yaml` - Streamlit UI config (self-contained, no Hydra runtime)
- `ui/unified_app.yaml` - Unified app config
- `ui/modes/*.yaml` - UI mode configurations

**Override Pattern**: These configs are NOT in `base.yaml` defaults
- For config groups: Use `+` to add them (e.g., `+data=canonical`)
- For UI configs: Load directly without Hydra composition

### Hybrid Configs (Transitional)

**Configs Using Both Patterns**:
- `data/canonical.yaml` - Can be used as standalone OR via override
- `data/craft.yaml` - Partial legacy (has some defaults for transforms)
- Model presets in `model/presets/*.yaml` - Use new architecture but provide alternate configurations

---

## Configuration Inventory by Group

### Group: `configs/_base/` (New Architecture Foundation)

| File | Package | Purpose | Dependencies | Status |
|------|---------|---------|--------------|--------|
| `core.yaml` | `_global_` | Core experiment settings | None | **Active** |
| `data.yaml` | `_global_` | Data composition | /data/base, /data/transforms/base, /data/dataloaders/default | **Active** |
| `model.yaml` | `model` | Model composition | /model/architectures:dbnet, /model/optimizers:adam | **Active** |
| `trainer.yaml` | `trainer` | Trainer settings | None | **Active** |
| `logging.yaml` | `logger` | Logger settings | None | **Active** |
| `preprocessing.yaml` | N/A | Preprocessing settings | None | **Active** |

**Architecture**: New
**Override**: N/A (foundation layer, referenced via defaults)
**Code References**: Used by all entry-point configs (`train.yaml`, `test.yaml`, `predict.yaml`)
**Recommendation**: **KEEP** - Core foundation of new architecture

---

### Group: `configs/model/` (Model Configurations)

#### `model/default.yaml`
**Architecture**: New
**Package**: `model`
**Dependencies**:
```yaml
defaults:
  - architectures: dbnet
  - optimizers: adam
```
**Override Pattern**: `model=default` (model in base.yaml defaults)
**Code References**: Default model config, used by `train.yaml`
**Recommendation**: **KEEP** - Primary model config

#### `model/architectures/*.yaml`
| File | Architecture | Override | Purpose |
|------|-------------|----------|---------|
| `dbnet.yaml` | New | `model/architectures=dbnet` | DBNet architecture config |
| `dbnetpp.yaml` | New | `model/architectures=dbnetpp` | DBNet++ variant |
| `craft.yaml` | New | `model/architectures=craft` | CRAFT architecture config |

**Code References**: Referenced by `model/default.yaml` and `model/presets/*.yaml`
**Recommendation**: **KEEP** - Active architecture definitions

#### `model/optimizers/*.yaml`
| File | Architecture | Override | Purpose |
|------|-------------|----------|---------|
| `adam.yaml` | New | `model/optimizers=adam` | Adam optimizer config |
| `adamw.yaml` | New | `model/optimizers=adamw` | AdamW optimizer variant |

**Code References**: Referenced by `model/default.yaml` and model presets
**Recommendation**: **KEEP** - Active optimizer configs

#### `model/optimizer.yaml` (LEGACY)
**Architecture**: Legacy
**Package**: `model.optimizer`
**Purpose**: Simplified optimizer config (predates optimizers/ directory)
**Code References**: None found in active code
**Difference from `optimizers/*.yaml`**: Single file vs directory-based system
**Recommendation**: **MOVE to `__LEGACY__/model/`** - Superseded by `optimizers/adam.yaml`

#### `model/presets/*.yaml`
| File | Architecture | Override | Purpose |
|------|-------------|----------|---------|
| `craft.yaml` | New | `+model/presets=craft` | Complete CRAFT preset |
| `dbnetpp.yaml` | New | `+model/presets=dbnetpp` | Complete DBNet++ preset |
| `model_example.yaml` | New | `+model/presets=model_example` | Example preset template |

**Architecture**: New (uses defaults composition)
**Override Pattern**: `+model/presets=X` (presets NOT in base.yaml defaults)
**Purpose**: Complete model configurations with all components specified
**Difference from `architectures/*.yaml`**: Presets combine architecture + optimizer + components, while architectures only define component overrides
**Code References**: Likely used via UI or manual overrides
**Recommendation**: **KEEP** - Serve distinct purpose from base architectures

#### `model/encoder/*.yaml`, `model/decoder/*.yaml`, `model/head/*.yaml`, `model/loss/*.yaml`
**Architecture**: New (component-based architecture)
**Override Pattern**: Referenced by architectures and presets
**Code References**: Used by model composition system
**Recommendation**: **KEEP** - Core components of new architecture

---

### Group: `configs/data/` (Data Configurations)

#### `data/base.yaml` (NEW ARCHITECTURE)
**Architecture**: New
**Package**: `_global_`
**Purpose**: Foundation data config with datasets, collate_fn
**Referenced By**: `configs/_base/data.yaml`
**Code References**: Used by all training/testing workflows
**Recommendation**: **KEEP** - Foundation layer

#### `data/default.yaml` (NEW ARCHITECTURE)
**Architecture**: New
**Package**: `_global_`
**Dependencies**:
```yaml
defaults:
  - /data/base
  - /data/transforms/base
  - /data/dataloaders/default
```
**Purpose**: Composition config pulling together data components
**Override Pattern**: Can be used but not necessary (already composed via _base/data.yaml)
**Recommendation**: **KEEP** - Part of new architecture

#### `data/canonical.yaml` (LEGACY)
**Architecture**: Legacy (standalone)
**Package**: `_global_`
**Purpose**: Complete standalone data config for canonical (rotation-corrected) images
**Dependencies**: None (self-contained)
**Differences from `data/base.yaml`**:
- Batch size: 16 (vs 4 in base)
- Uses `images_val_canonical` path
- No transform composition (defines transforms inline - REMOVED in this file)
- No performance optimization features
**Override Pattern**: `+data=canonical` (NOT in base.yaml defaults - ERROR!)
**Code References**:
- `ocr/command_builder/compute.py` - References `data=canonical`
- `tests/unit/test_hydra_overrides.py` - Tests canonical override
**Actual Usage**: Should be `data=canonical` without `+` OR need to add to base.yaml defaults
**Recommendation**: **KEEP but DOCUMENT** - Actively used but needs override pattern clarification

#### `data/craft.yaml` (HYBRID)
**Architecture**: Hybrid (partial legacy)
**Package**: `_global_`
**Dependencies**: `- /data/transforms/base` (partial composition)
**Purpose**: CRAFT-specific data configuration with custom transforms and collate function
**Differences from `data/base.yaml`**:
- Uses `CraftCollateFN` instead of `DBCollateFN`
- Image size: 768x768 (vs 640x640)
- Custom transform parameters (RandomRotate90, different normalization)
**Override Pattern**: `+data=craft` or `data=craft` (unclear - needs testing)
**Code References**: Likely used for CRAFT model training
**Recommendation**: **KEEP** - Serves distinct purpose for CRAFT architecture

#### `data/preprocessing.yaml` (LEGACY)
**Architecture**: Legacy
**Purpose**: Preprocessing-specific data config
**Code References**: Not found in active code
**Recommendation**: **INVESTIGATE** - May be unused, consider moving to `__LEGACY__/`

#### `data/transforms/*.yaml`
| File | Architecture | Purpose |
|------|-------------|---------|
| `base.yaml` | New | Default transform configurations for train/val/test |
| `background_removal.yaml` | New | Transforms with background removal |
| `with_background_removal.yaml` | New | Alternative background removal config |

**Code References**: Referenced by `data/base.yaml` and `data/default.yaml`
**Recommendation**: **KEEP** - Active transform configs

#### `data/dataloaders/*.yaml`
| File | Architecture | Purpose |
|------|-------------|---------|
| `default.yaml` | New | Default dataloader settings (4 workers) |
| `rtx3060_16core.yaml` | New | Hardware-specific dataloader optimization |

**Code References**: Referenced by data composition configs
**Recommendation**: **KEEP** - Active dataloader configs

#### `data/performance_preset/*.yaml`
**Architecture**: New
**Purpose**: Performance optimization presets (balanced, memory_efficient, validation_optimized, etc.)
**Override Pattern**: `+data/performance_preset=X` (NOT in defaults)
**Code References**: Documented in `data/base.yaml` comments
**Recommendation**: **KEEP** - Provides performance tuning options

#### `data/datasets/*.yaml`
**Architecture**: New
**Purpose**: Dataset-specific configurations (db, preprocessing, preprocessing_camscanner, preprocessing_docTR_demo)
**Code References**: Not directly referenced (may be legacy or experimental)
**Recommendation**: **INVESTIGATE** - Determine if actively used or can be archived

---

### Group: `configs/logger/` (Logger Configurations)

#### `logger/consolidated.yaml` (NEW)
**Architecture**: New
**Package**: `_global_`
**Purpose**: Merged logger config (wandb + csv)
**Override Pattern**: `logger=consolidated` (logger in base.yaml defaults)
**Code References**: Default logger in base.yaml
**Recommendation**: **KEEP** - Primary logger config

#### `logger/wandb.yaml` (LEGACY or VARIANT)
**Architecture**: Unclear (may be legacy or alternate config)
**Purpose**: Standalone W&B logger config
**Override Pattern**: `logger=wandb` (overrides consolidated)
**Code References**:
- `tests/unit/test_hydra_overrides.py` - Tests `logger=wandb` override
- Likely used via command-line overrides
**Recommendation**: **KEEP** - Provides W&B-only option (no CSV)

#### `logger/csv.yaml` (LEGACY or VARIANT)
**Architecture**: Unclear
**Purpose**: CSV-only logger config
**Override Pattern**: `logger=csv` (overrides consolidated)
**Code References**: Not found in active code
**Recommendation**: **KEEP** - Provides CSV-only option

#### `logger/default.yaml` (LEGACY?)
**Architecture**: Unknown (not read yet)
**Purpose**: Default logger config (may be superseded by consolidated)
**Recommendation**: **INVESTIGATE** - Check if duplicate of consolidated.yaml

---

### Group: `configs/trainer/` (Trainer Configurations)

#### `trainer/default.yaml`
**Architecture**: New
**Package**: N/A (used as config group)
**Purpose**: Default Lightning trainer settings
**Override Pattern**: `trainer=default` (trainer in base.yaml defaults)
**Code References**: Default trainer in base.yaml
**Recommendation**: **KEEP** - Primary trainer config

#### `trainer/fp16_safe.yaml`, `trainer/rtx3060_12gb.yaml`, `trainer/hardware_rtx3060_12gb_i5_16core.yaml`
**Architecture**: New
**Purpose**: Hardware-specific trainer optimizations
**Override Pattern**: `trainer=X` (overrides default)
**Code References**: Likely used via command-line for specific hardware
**Recommendation**: **KEEP** - Hardware-specific configs serve distinct purpose

---

### Group: `configs/callbacks/` (Callback Configurations)

#### `callbacks/default.yaml`
**Architecture**: New
**Purpose**: Default callback configuration
**Override Pattern**: `callbacks=default` (callbacks in base.yaml defaults)
**Code References**: Default in base.yaml
**Recommendation**: **KEEP** - Primary callbacks config

#### Individual callback configs (early_stopping, model_checkpoint, model_summary, etc.)
**Architecture**: New
**Purpose**: Individual callback definitions referenced by default.yaml
**Code References**: Composed by callbacks/default.yaml
**Recommendation**: **KEEP** - Active callback components

---

### Group: `configs/ui/` and `configs/ui_meta/` (UI Configurations)

**Architecture**: Standalone (intentionally not using Hydra runtime)
**Purpose**: Streamlit UI configurations loaded directly without Hydra composition
**Code References**:
- `apps/ocr-inference-console/backend/` - Loads UI configs directly
- `ui/` directory (if exists)
**Override Pattern**: N/A (not used with Hydra compose)
**Recommendation**: **KEEP** - Intentionally standalone for UI applications

---

### Group: `configs/hydra/` (Hydra-Specific Configurations)

#### `hydra/default.yaml`, `hydra/disabled.yaml`
**Architecture**: New
**Purpose**: Hydra runtime configurations
**Override Pattern**: `hydra=X` (hydra NOT in base.yaml defaults, but handled specially)
**Code References**: Used by Hydra system itself
**Note**: In `base.yaml`, Hydra config is inlined directly (not via config group) due to `@package` limitations with `version_base=None`
**Recommendation**: **KEEP** - Required for Hydra operation

---

### Group: `configs/paths/`, `configs/evaluation/`, `configs/debug/`

**Architecture**: New
**Purpose**: Support configurations for paths, evaluation metrics, debug settings
**Override Pattern**: In base.yaml defaults (use without `+`)
**Code References**: Used by base.yaml composition
**Recommendation**: **KEEP** - Active support configs

---

### Group: `configs/extraction/`, `configs/layout/`, `configs/recognition/`

**Architecture**: New (modular system for OCR pipeline)
**Purpose**: Receipt extraction, layout analysis, text recognition configs
**Override Pattern**: `+extraction=default`, `+layout=default`, `+recognition=default` (NOT in defaults)
**Code References**: May be used by specialized workflows
**Recommendation**: **KEEP** - Part of modular OCR pipeline architecture

---

### Group: `configs/benchmark/`

#### `benchmark/decoder.yaml`
**Architecture**: Unknown
**Purpose**: Decoder benchmarking configuration
**Code References**: Not found in active code
**Recommendation**: **INVESTIGATE** - May be experimental or archival

---

## Override Pattern Analysis

### Rules Summary

| Config Group | In base.yaml defaults? | Override Pattern | Example |
|-------------|----------------------|------------------|---------|
| `model` | ✅ Yes | `model=X` | `model=default` |
| `logger` | ✅ Yes | `logger=X` | `logger=wandb` |
| `trainer` | ✅ Yes | `trainer=X` | `trainer=fp16_safe` |
| `callbacks` | ✅ Yes | `callbacks=X` | `callbacks=default` |
| `debug` | ✅ Yes | `debug=X` | `debug=default` |
| `evaluation` | ✅ Yes | `evaluation=X` | `evaluation=metrics` |
| `paths` | ✅ Yes | `paths=X` | `paths=default` |
| `data` | ❌ No | `+data=X` | `+data=canonical` |
| `ablation` | ❌ No | `+ablation=X` | `+ablation=model_comparison` |
| `hardware` | ❌ No | `+hardware=X` | `+hardware=rtx3060_12gb_i5_16core` |
| `model/architectures` | ❌ No | `model/architectures=X` | `model/architectures=dbnetpp` |
| `model/presets` | ❌ No | `+model/presets=X` | `+model/presets=craft` |
| `data/performance_preset` | ❌ No | `+data/performance_preset=X` | `+data/performance_preset=balanced` |

### Common Errors

1. **Using `+` on configs in defaults**:
   - ❌ `+logger=wandb` → ERROR: "Multiple values for logger"
   - ✅ `logger=wandb` → Correct

2. **Not using `+` for new config groups**:
   - ❌ `ablation=model_comparison` → ERROR: "Could not find ablation"
   - ✅ `+ablation=model_comparison` → Correct

3. **Confusion about data config**:
   - Current: `data` is NOT in `base.yaml` defaults
   - Should use: `+data=canonical` or `+data=craft`
   - BUT: Tests use `data=canonical` (without `+`)
   - **ISSUE**: Inconsistency in override pattern expectations

### Nested Override Patterns

For nested config paths (e.g., `model/architectures`):
- If parent is in defaults, override without `+`: `model/architectures=dbnet`
- If adding new nested group, use `+`: `+model/new_feature=value`

---

## Code Reference Analysis

### Active Hydra Usage

**Runners** (primary entry points):
- `runners/train.py` - Uses `@hydra.main(config_path="configs", config_name="train")`
- `runners/test.py` - Uses `@hydra.main(config_path="configs", config_name="test")`
- `runners/predict.py` - Uses `@hydra.main(config_path="configs", config_name="predict")`
- `runners/generate_synthetic.py` - Uses `@hydra.main(config_path="configs", config_name="synthetic")`

**Utility Modules**:
- `ocr/utils/config_utils.py` - Config utility functions, references model/architectures patterns
- `ocr/utils/command/builder.py` - Command builder for Hydra overrides
- `ocr/inference/config_loader.py` - Loads saved Hydra configs from checkpoints

**Test Suite**:
- `tests/unit/test_hydra_overrides.py` - Comprehensive override pattern tests

**Legacy/Archive**:
- `archive/legacy_ui_code/` - Old UI code with config references (archived)
- `AgentQMS/` - Config generation scripts (specialized tooling)

### Config Group References Found

From code analysis:
1. **data=canonical** - Referenced in `ocr/command_builder/compute.py`
2. **model/architectures** - Referenced in `ocr/utils/config_utils.py`
3. **logger=wandb** - Tested in `test_hydra_overrides.py`
4. **trainer=X** - Overrideable via command-line
5. **UI configs** - Loaded directly by Streamlit apps

### Scripts with Config Overrides

Scripts in `scripts/` directory do not appear to use Hydra compose directly based on grep results. They likely invoke runners with command-line arguments.

---

## Impact Assessment & Removal Risk

### Safe to Move to `__LEGACY__/` (Low Risk)

| Config | Reason | References | Lost Functionality |
|--------|--------|-----------|-------------------|
| `model/optimizer.yaml` | Superseded by `model/optimizers/adam.yaml` | None found | Single-file optimizer config (now directory-based) |
| `data/preprocessing.yaml` | Not found in active code | None found | Preprocessing-specific data config |
| `logger/default.yaml` | Likely duplicate of `logger/consolidated.yaml` | Need to verify | May be duplicate |
| `data/datasets/*.yaml` | Experimental or unused | Not directly referenced | Dataset-specific variants |

### Keep but Document (Medium Risk)

| Config | Reason | References | Action Needed |
|--------|--------|-----------|--------------|
| `data/canonical.yaml` | Actively used | `ocr/command_builder/compute.py`, tests | Document override pattern: should it use `+` or not? |
| `data/craft.yaml` | CRAFT-specific config | Likely used for CRAFT training | Document when to use vs `data/base.yaml` |
| `logger/wandb.yaml` | W&B-only variant | Tests | Document as alternative to consolidated |
| `logger/csv.yaml` | CSV-only variant | Not found but likely usable | Document as alternative to consolidated |

### Must Keep (High Risk if Removed)

| Config | Reason | Impact if Removed |
|--------|--------|------------------|
| All `_base/*.yaml` | Foundation of new architecture | Complete system failure |
| `model/default.yaml` | Primary model config | Training would fail |
| `model/architectures/*.yaml` | Active architecture definitions | Cannot build models |
| `model/optimizers/*.yaml` | Active optimizer configs | Training would fail |
| `model/presets/*.yaml` | Complete model presets | Lose preset functionality |
| `data/base.yaml` | Foundation data config | Data loading would fail |
| `data/transforms/base.yaml` | Default transforms | Data pipeline would fail |
| `data/dataloaders/default.yaml` | Default dataloader settings | Data loading would fail |
| `logger/consolidated.yaml` | Primary logger | Logging would fail |
| `trainer/default.yaml` | Primary trainer config | Training would fail |
| `callbacks/default.yaml` | Primary callbacks | Callback system would fail |
| UI configs (`ui/*`, `ui_meta/*`) | UI functionality | UI apps would fail |
| Entry points (`train.yaml`, `test.yaml`, etc.) | System entry points | Cannot run workflows |

---

## Configuration Duplication Analysis

Based on 2025-11-11 audit findings:

### NOT Duplicates (Confirmed)

1. **`model/optimizer.yaml` vs `model/optimizers/adam.yaml`**:
   - `optimizer.yaml`: Single-file config with `@package model.optimizer` (legacy pattern)
   - `optimizers/adam.yaml`: Directory-based config, part of new architecture
   - **Verdict**: Different patterns, but `optimizer.yaml` is superseded

2. **`data/canonical.yaml` vs `data/base.yaml`**:
   - `canonical.yaml`: Standalone complete config for canonical images (batch_size=16, no composition)
   - `base.yaml`: Foundation config with composition pattern (batch_size=4, uses defaults)
   - **Verdict**: Different purposes - canonical is a preset variant, base is foundation

3. **`data/craft.yaml` vs `data/base.yaml`**:
   - `craft.yaml`: CRAFT-specific with CraftCollateFN, 768x768 images, custom transforms
   - `base.yaml`: Generic DBNet-style config with DBCollateFN, 640x640 images
   - **Verdict**: Different architectures, serve different purposes

4. **`logger/wandb.yaml` vs `logger/consolidated.yaml`**:
   - `wandb.yaml`: W&B-only logger
   - `consolidated.yaml`: W&B + CSV loggers combined
   - **Verdict**: Different logging strategies

### True Duplicates (None Found)

No configurations were identified as true duplicates in the 2025-11-11 audit.

---

## Resolution Plan

### Phase 1: Documentation (Immediate - Low Risk)

**Goal**: Clarify override patterns and document configuration architecture

**Actions**:
1. Create `configs/README.md` documenting:
   - New vs legacy architecture
   - Override pattern rules
   - When to use `+` prefix
   - Config group hierarchy
   - Common errors and solutions

2. Add override pattern comments to `configs/base.yaml`:
   ```yaml
   # Override configs IN this defaults list WITHOUT + prefix
   # Example: logger=wandb (correct), +logger=wandb (error)
   defaults:
     - model: default      # Override: model=X
     - logger: consolidated  # Override: logger=X
     ...
   ```

3. Update `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_OVERRIDE_PATTERNS.md` with complete examples

**Deliverables**:
- `configs/README.md` (new)
- Updated `HYDRA_OVERRIDE_PATTERNS.md`
- Comments in `configs/base.yaml`

**Risk**: None
**Estimated Effort**: 1 hour
**Validation**: Review documentation with users

---

### Phase 2: Legacy Containerization (Short-term - Low to Medium Risk)

**Goal**: Separate legacy configs from new architecture without breaking functionality

**Actions**:
1. Create `configs/__LEGACY__/` directory structure:
   ```
   configs/__LEGACY__/
   ├── README.md (explain purpose, migration path)
   ├── model/
   │   └── optimizer.yaml (moved from configs/model/)
   ├── data/
   │   └── preprocessing.yaml (moved from configs/data/)
   └── logger/
       └── default.yaml (if confirmed duplicate)
   ```

2. Move low-risk legacy configs:
   - `model/optimizer.yaml` → `__LEGACY__/model/optimizer.yaml`
   - `data/preprocessing.yaml` → `__LEGACY__/data/preprocessing.yaml`
   - Any other confirmed unused configs

3. Test that Hydra can still find configs in `__LEGACY__/`:
   - Hydra searches all subdirectories by default
   - Override: `+model/optimizer=__LEGACY__/optimizer` (if needed)
   - OR: Update search path to include `__LEGACY__/`

4. Create `configs/__LEGACY__/README.md`:
   ```markdown
   # Legacy Configuration Archive

   This directory contains legacy Hydra configurations that have been superseded
   by the new architecture using `configs/_base/` foundation.

   ## Why These Configs Are Here
   - Superseded by newer patterns (e.g., optimizer.yaml → optimizers/adam.yaml)
   - Not actively used in current workflows
   - Preserved for reference and backward compatibility

   ## Migration Guide
   - `model/optimizer.yaml` → Use `model/optimizers/adam.yaml` instead
   - `data/preprocessing.yaml` → Use `data/base.yaml` with preprocessing overrides

   ## Removal Timeline
   - Phase 1 (Current): Moved to __LEGACY__/ (still accessible)
   - Phase 2 (Future): Archive to `archive/configs/` (not accessible via Hydra)
   - Phase 3 (Distant): Remove entirely (documented in archive)
   ```

**Deliverables**:
- `configs/__LEGACY__/` directory
- `configs/__LEGACY__/README.md`
- Moved config files
- Updated `.gitignore` if needed

**Risk**: Low (configs remain accessible, just moved)
**Estimated Effort**: 2 hours
**Validation**:
- Test that moved configs are still accessible
- Run `python tests/unit/test_hydra_overrides.py`
- Test manual override: `uv run python runners/train.py +model/optimizer=__LEGACY__/optimizer`

---

### Phase 3: Code Reference Updates (Medium-term - Medium Risk)

**Goal**: Update code to use new architecture patterns exclusively

**Actions**:
1. Update `ocr/command_builder/compute.py`:
   - Change `data=canonical` to use new architecture pattern
   - OR: Add `data` to `base.yaml` defaults and keep `data=canonical` (without `+`)

2. Update any UI code referencing legacy configs:
   - Review `apps/ocr-inference-console/backend/` for config loading
   - Ensure UI uses new architecture or loads configs directly (not via Hydra)

3. Update test suite:
   - `tests/unit/test_hydra_overrides.py` - Update to test new patterns
   - Add tests for `__LEGACY__/` config access

4. Create compatibility layer (if needed):
   - Hydra resolver to map legacy names to new configs
   - Example: `legacy:optimizer` → `optimizers/adam`

**Deliverables**:
- Updated `ocr/command_builder/compute.py`
- Updated test suite
- Compatibility layer (if needed)

**Risk**: Medium (changes active code paths)
**Estimated Effort**: 4-6 hours
**Validation**:
- Full test suite run
- Manual testing of training/testing/prediction workflows
- UI smoke tests

---

### Phase 4: Deprecation Warnings (Long-term - Low Risk)

**Goal**: Warn users when they use legacy configs

**Actions**:
1. Add deprecation detection to config loading:
   ```python
   def check_legacy_config_usage(cfg):
       if "optimizer" in cfg.model and not "optimizers" in cfg.model:
           warnings.warn(
               "model/optimizer.yaml is deprecated. Use model/optimizers/adam.yaml instead.",
               DeprecationWarning
           )
   ```

2. Add to `runners/train.py`, `runners/test.py`, `runners/predict.py`:
   ```python
   @hydra.main(...)
   def main(cfg: DictConfig):
       check_legacy_config_usage(cfg)
       ...
   ```

3. Document deprecation in CHANGELOG.md

**Deliverables**:
- Deprecation warning system
- Updated CHANGELOG.md
- User notification plan

**Risk**: Low (warnings only, no breaking changes)
**Estimated Effort**: 2 hours
**Validation**: Test warnings appear for legacy usage

---

### Phase 5: Archive and Remove (Future - High Risk, Not Recommended Yet)

**Goal**: Remove legacy configs entirely (only after extended deprecation period)

**Status**: NOT RECOMMENDED for current session

**Rationale**:
- Configs provide historical reference
- Low storage cost
- May be needed for reproducing old experiments
- Breaking changes should be avoided

**If Eventual Removal Desired**:
1. Move to `archive/configs/` (not accessible via Hydra)
2. Document what was removed and why
3. Provide migration guide
4. Announce breaking change well in advance
5. Only after 6-12 months of deprecation warnings

**Risk**: High (breaks backward compatibility)
**Estimated Effort**: 1 hour (moving files)
**Validation**: Full regression testing

---

## Session Handover & Continuation Plan

### Current Session Status

**Completed**:
- ✅ Architecture analysis (new vs legacy)
- ✅ Configuration inventory (80+ files classified)
- ✅ Override pattern documentation
- ✅ Code reference search
- ✅ Impact assessment
- ✅ Resolution plan (5 phases)

**Token Usage**: ~150k / 200k (75% utilized)

**Remaining Work**:
- Phase 1: Documentation (can start immediately)
- Phase 2-5: Implementation (requires user approval and testing)

---

### Continuation Instructions for Next Session

**Context Handover**:
1. **Read This Document First**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_CONFIG_AUDIT_ASSESSMENT.md`
2. **Review Audit Prompt**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_CONFIG_AUDIT_PROMPT.md`
3. **Check Override Patterns**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_OVERRIDE_PATTERNS.md`

**Next Steps**:

**Option A: Implement Phase 1 (Documentation)**
```bash
# Create configs README
touch configs/README.md
# Edit configs/base.yaml to add override comments
# Update HYDRA_OVERRIDE_PATTERNS.md with examples
```

**Option B: Implement Phase 2 (Legacy Containerization)**
```bash
# Create legacy directory
mkdir -p configs/__LEGACY__/model configs/__LEGACY__/data configs/__LEGACY__/logger

# Move legacy configs (dry run first)
git mv configs/model/optimizer.yaml configs/__LEGACY__/model/
git mv configs/data/preprocessing.yaml configs/__LEGACY__/data/

# Create README
touch configs/__LEGACY__/README.md

# Test config accessibility
uv run python tests/unit/test_hydra_overrides.py
```

**Option C: Investigate Uncertain Configs**
```bash
# Check if logger/default.yaml is duplicate
diff configs/logger/default.yaml configs/logger/consolidated.yaml

# Search for preprocessing.yaml references
grep -r "preprocessing.yaml" --include="*.py" .

# Test data/canonical override pattern
uv run python runners/train.py +data=canonical --help  # Test if + needed
uv run python runners/train.py data=canonical --help   # Test without +
```

**Questions to Answer**:
1. Should `data` be added to `base.yaml` defaults? (Would change override pattern)
2. Is `logger/default.yaml` identical to `logger/consolidated.yaml`?
3. Are `data/datasets/*.yaml` configs used anywhere?
4. Should `data/canonical.yaml` remain standalone or migrate to new architecture?

---

### Key Decisions Required

**Decision 1: Data Config Override Pattern**
- **Current**: `data` NOT in `base.yaml` defaults → should use `+data=canonical`
- **But**: Tests use `data=canonical` (without `+`)
- **Options**:
  - A: Add `data` to `base.yaml` defaults → `data=canonical` works
  - B: Keep current, update tests to use `+data=canonical`
  - C: Make `data/canonical.yaml` the default, use overrides for variants
- **Recommendation**: **Option A** - Add `data: base` to defaults for consistency

**Decision 2: Legacy Config Fate**
- **Options**:
  - A: Move to `__LEGACY__/` (containerize)
  - B: Leave in place with deprecation warnings
  - C: Archive to `archive/configs/` (remove from Hydra search path)
  - D: Delete entirely (not recommended)
- **Recommendation**: **Option A** - Containerize for clarity while maintaining access

**Decision 3: Model Presets vs Architectures**
- **Current**: Both exist and serve different purposes
- **Question**: Is this distinction clear to users?
- **Options**:
  - A: Keep as-is, document the difference
  - B: Rename one for clarity (e.g., `model/presets` → `model/complete_configs`)
  - C: Merge into single system
- **Recommendation**: **Option A** - Keep distinct, improve documentation

---

### Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking existing workflows | Medium | High | Thorough testing, gradual rollout |
| Config not found after moving | Low | High | Test Hydra search path, maintain `__LEGACY__/` access |
| User confusion about new patterns | High | Medium | Clear documentation, examples, deprecation warnings |
| Old experiments not reproducible | Low | Medium | Keep legacy configs accessible, document migration |
| Override pattern inconsistency | High | Medium | Standardize on one pattern, update base.yaml defaults |

---

### Success Metrics

**Documentation Phase**:
- [ ] `configs/README.md` created
- [ ] Override patterns documented with examples
- [ ] No user questions about override patterns for 1 week

**Containerization Phase**:
- [ ] `__LEGACY__/` directory created
- [ ] All legacy configs moved
- [ ] All tests still pass
- [ ] Configs still accessible via Hydra

**Code Update Phase**:
- [ ] No code references to moved configs
- [ ] All workflows use new architecture
- [ ] Test coverage for new patterns
- [ ] No regressions in functionality

**Deprecation Phase**:
- [ ] Warnings appear for legacy usage
- [ ] Users notified of deprecation
- [ ] Migration guide published
- [ ] No new code using legacy patterns

---

## Appendix A: Complete Configuration Inventory

| Path | Architecture | Package | In Defaults? | Override | Status | Recommendation |
|------|-------------|---------|-------------|----------|--------|----------------|
| `_base/core.yaml` | New | `_global_` | N/A | N/A | Active | KEEP |
| `_base/data.yaml` | New | `_global_` | N/A | N/A | Active | KEEP |
| `_base/model.yaml` | New | `model` | N/A | N/A | Active | KEEP |
| `_base/trainer.yaml` | New | `trainer` | N/A | N/A | Active | KEEP |
| `_base/logging.yaml` | New | `logger` | N/A | N/A | Active | KEEP |
| `_base/preprocessing.yaml` | New | N/A | N/A | N/A | Active | KEEP |
| `base.yaml` | New | N/A | N/A | N/A | Active | KEEP |
| `train.yaml` | New | N/A | N/A | N/A | Active | KEEP |
| `test.yaml` | New | N/A | N/A | N/A | Active | KEEP |
| `predict.yaml` | New | N/A | N/A | N/A | Active | KEEP |
| `synthetic.yaml` | New | N/A | N/A | N/A | Active | KEEP |
| `performance_test.yaml` | New | N/A | N/A | N/A | Active | KEEP |
| `cache_performance_test.yaml` | New | N/A | N/A | N/A | Active | KEEP |
| `train_v2.yaml` | New | N/A | N/A | N/A | Active | INVESTIGATE |
| `model/default.yaml` | New | `model` | Yes | `model=default` | Active | KEEP |
| `model/optimizer.yaml` | Legacy | `model.optimizer` | No | `+model/optimizer=X` | Unused | MOVE TO `__LEGACY__/` |
| `model/architectures/dbnet.yaml` | New | `model` | No | `model/architectures=dbnet` | Active | KEEP |
| `model/architectures/dbnetpp.yaml` | New | `model` | No | `model/architectures=dbnetpp` | Active | KEEP |
| `model/architectures/craft.yaml` | New | `model` | No | `model/architectures=craft` | Active | KEEP |
| `model/optimizers/adam.yaml` | New | N/A | No | `model/optimizers=adam` | Active | KEEP |
| `model/optimizers/adamw.yaml` | New | N/A | No | `model/optimizers=adamw` | Active | KEEP |
| `model/presets/craft.yaml` | New | `model` | No | `+model/presets=craft` | Active | KEEP |
| `model/presets/dbnetpp.yaml` | New | `model` | No | `+model/presets=dbnetpp` | Active | KEEP |
| `model/presets/model_example.yaml` | New | `model` | No | `+model/presets=model_example` | Example | KEEP |
| `model/encoder/*.yaml` | New | N/A | No | Via composition | Active | KEEP |
| `model/decoder/*.yaml` | New | N/A | No | Via composition | Active | KEEP |
| `model/head/*.yaml` | New | N/A | No | Via composition | Active | KEEP |
| `model/loss/*.yaml` | New | N/A | No | Via composition | Active | KEEP |
| `model/lightning_modules/base.yaml` | New | N/A | No | Via composition | Active | KEEP |
| `data/base.yaml` | New | `_global_` | No | Via `_base/data.yaml` | Active | KEEP |
| `data/default.yaml` | New | `_global_` | No | `+data=default` | Active | KEEP |
| `data/canonical.yaml` | Legacy | `_global_` | No | `+data=canonical` or `data=canonical`? | Active | KEEP + DOCUMENT |
| `data/craft.yaml` | Hybrid | `_global_` | No | `+data=craft` | Active | KEEP |
| `data/preprocessing.yaml` | Legacy | N/A | No | `+data=preprocessing` | Unused? | INVESTIGATE → `__LEGACY__/` |
| `data/transforms/base.yaml` | New | `_global_` | No | Via composition | Active | KEEP |
| `data/transforms/background_removal.yaml` | New | `_global_` | No | `+data/transforms=background_removal` | Active | KEEP |
| `data/transforms/with_background_removal.yaml` | New | `_global_` | No | `+data/transforms=with_background_removal` | Active | KEEP |
| `data/dataloaders/default.yaml` | New | `dataloaders` | No | Via composition | Active | KEEP |
| `data/dataloaders/rtx3060_16core.yaml` | New | `dataloaders` | No | `+data/dataloaders=rtx3060_16core` | Active | KEEP |
| `data/performance_preset/*.yaml` | New | N/A | No | `+data/performance_preset=X` | Active | KEEP |
| `data/datasets/*.yaml` | New | N/A | No | Unknown | Unused? | INVESTIGATE |
| `logger/consolidated.yaml` | New | `_global_` | Yes | `logger=consolidated` | Active | KEEP |
| `logger/wandb.yaml` | Legacy/Variant | N/A | Yes (via consolidated) | `logger=wandb` | Active | KEEP |
| `logger/csv.yaml` | Legacy/Variant | N/A | Yes (via consolidated) | `logger=csv` | Unused? | KEEP |
| `logger/default.yaml` | Unknown | N/A | Yes | `logger=default` | Unknown | INVESTIGATE |
| `trainer/default.yaml` | New | N/A | Yes | `trainer=default` | Active | KEEP |
| `trainer/fp16_safe.yaml` | New | N/A | Yes | `trainer=fp16_safe` | Active | KEEP |
| `trainer/rtx3060_12gb.yaml` | New | N/A | Yes | `trainer=rtx3060_12gb` | Active | KEEP |
| `trainer/hardware_rtx3060_12gb_i5_16core.yaml` | New | N/A | Yes | `trainer=hardware_rtx3060_12gb_i5_16core` | Active | KEEP |
| `callbacks/default.yaml` | New | N/A | Yes | `callbacks=default` | Active | KEEP |
| `callbacks/*.yaml` (individual) | New | N/A | No | Via composition | Active | KEEP |
| `evaluation/metrics.yaml` | New | N/A | Yes | `evaluation=metrics` | Active | KEEP |
| `paths/default.yaml` | New | N/A | Yes | `paths=default` | Active | KEEP |
| `debug/default.yaml` | New | N/A | Yes | `debug=default` | Active | KEEP |
| `hydra/default.yaml` | New | N/A | Special | `hydra=default` | Active | KEEP |
| `hydra/disabled.yaml` | New | N/A | Special | `hydra=disabled` | Active | KEEP |
| `ui/*.yaml` | Standalone | N/A | No | Direct load | Active | KEEP |
| `ui_meta/*.yaml` | Standalone | N/A | No | Direct load | Active | KEEP |
| `extraction/default.yaml` | New | N/A | No | `+extraction=default` | Active | KEEP |
| `layout/default.yaml` | New | N/A | No | `+layout=default` | Active | KEEP |
| `recognition/default.yaml` | New | N/A | No | `+recognition=default` | Active | KEEP |
| `benchmark/decoder.yaml` | Unknown | N/A | No | Unknown | Unused? | INVESTIGATE |

**Legend**:
- **Architecture**: New (uses `_base/`), Legacy (standalone), Hybrid (partial), Standalone (intentional)
- **In Defaults**: Whether listed in `base.yaml` defaults
- **Override**: Pattern to use for overriding
- **Status**: Active (used), Unused (no references), Unknown (needs investigation)
- **Recommendation**: KEEP, INVESTIGATE, MOVE TO `__LEGACY__/`, DOCUMENT

---

## Appendix B: Hydra Defaults Hierarchy

```
configs/base.yaml
├── defaults:
│   ├── model: default
│   │   └── configs/model/default.yaml
│   │       ├── defaults:
│   │       │   ├── architectures: dbnet
│   │       │   │   └── configs/model/architectures/dbnet.yaml
│   │       │   └── optimizers: adam
│   │       │       └── configs/model/optimizers/adam.yaml
│   ├── evaluation: metrics
│   │   └── configs/evaluation/metrics.yaml
│   ├── paths: default
│   │   └── configs/paths/default.yaml
│   ├── logger: consolidated
│   │   └── configs/logger/consolidated.yaml
│   ├── trainer: default
│   │   └── configs/trainer/default.yaml
│   ├── callbacks: default
│   │   └── configs/callbacks/default.yaml
│   └── debug: default
│       └── configs/debug/default.yaml

configs/train.yaml
├── defaults:
│   ├── _self_
│   ├── base (includes all above)
│   ├── /_base/model
│   │   └── configs/_base/model.yaml
│   │       ├── defaults:
│   │       │   ├── /model/architectures: dbnet
│   │       │   └── /model/optimizers: adam
│   ├── /_base/data
│   │   └── configs/_base/data.yaml
│   │       ├── defaults:
│   │       │   ├── /data/base
│   │       │   ├── /data/transforms/base
│   │       │   └── /data/dataloaders/default
│   └── /_base/trainer
│       └── configs/_base/trainer.yaml
```

**Note**: This hierarchy shows the composition order. Configs listed in `defaults` can be overridden without `+`. Configs NOT in defaults require `+` to add.

---

## Appendix C: Testing & Validation Commands

### Test Override Patterns
```bash
# Run comprehensive override tests
uv run python tests/unit/test_hydra_overrides.py

# Test specific override (dry run)
uv run python runners/train.py logger=wandb --cfg job --help

# Test problematic pattern
uv run python runners/train.py +logger=wandb --cfg job  # Should fail

# Test data override
uv run python runners/train.py data=canonical --cfg job
uv run python runners/train.py +data=canonical --cfg job  # Test which works
```

### Verify Config Loading
```bash
# Show resolved config
uv run python runners/train.py --cfg job

# Show config tree
uv run python runners/train.py --cfg hydra

# Test compose
python -c "
from hydra import initialize, compose
with initialize(config_path='configs', version_base=None):
    cfg = compose(config_name='train')
    print(cfg.keys())
"
```

### Test Legacy Config Access
```bash
# After moving to __LEGACY__/
uv run python runners/train.py +model/optimizer=__LEGACY__/optimizer --cfg job
```

### Validate All Configs Load
```bash
# Test all entry points
for config in train test predict synthetic; do
    echo "Testing $config..."
    uv run python runners/$config.py --cfg job --help || echo "FAILED: $config"
done
```

---

## Appendix D: Reference Documentation

### Related Documents
- **Audit Prompt**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_CONFIG_AUDIT_PROMPT.md`
- **Audit Context**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_AUDIT_CONTEXT.md`
- **Override Patterns**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_OVERRIDE_PATTERNS.md`
- **Previous Audit**: `archive/archive_docs/docs/completed_plans/2025-11/2025-11-11_1439_implementation_plan_legacy-cleanup-config-consolidation.md`

### External References
- Hydra Documentation: https://hydra.cc/docs/intro/
- Hydra Overrides: https://hydra.cc/docs/advanced/override_grammar/basic/
- Hydra Composition: https://hydra.cc/docs/advanced/defaults_list/

### Project Standards
- **ADS Standards**: `.ai-instructions/tier1-sst/artifact-types.yaml`
- **Naming Convention**: `YYYY-MM-DD_HHMM_{type}_{description}.md`
- **Audit Location**: `docs/artifacts/audits/`

---

**End of Assessment Document**

**Next Action**: Await user decision on which resolution phase to execute, or request clarification on specific findings.
