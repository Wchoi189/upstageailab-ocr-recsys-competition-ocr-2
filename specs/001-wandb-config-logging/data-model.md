# Data Model: WandB Configuration Logging Constraints

**Feature**: `001-wandb-config-logging`
**Date**: February 15, 2026

## Overview

This feature does not introduce new data entities or state management. It establishes configuration constraints and documentation patterns for existing entities.

## Existing Entities Affected

### 1. WandB Logger Configuration (YAML)

**Location**: `/workspaces/configs/train/logger/wandb.yaml`

**Structure**:
```yaml
_target_: lightning.pytorch.loggers.WandbLogger
project: string                    # WandB project name
log_model: "all" | "best" | false  # Checkpoint logging strategy
save_dir: string                   # Local save directory (interpolated path)
enabled: boolean                   # Logger on/off toggle
log_recognition_images: boolean    # Custom domain-specific flag
standardize_name: boolean          # Use generate_run_name() for run naming
log_config: boolean               # CONSTRAINT APPLIES HERE
settings:
  offline: boolean
  save_code: boolean
  sync_dir: string
```

**Constraint Field**: `log_config`
- **Type**: Boolean
- **Default (new)**: `false`
- **Validation**: None (runtime failure if `true` with incompatible config)
- **Relationships**:
  - When `true` + `standardize_name: false`: Uses WandB's auto-naming
  - When `true` + config has `_target_`: Serialization error at trainer initialization
  - When `false`: Essential config still visible via run name encoding

### 2. Hydra Configuration Tree (DictConfig)

**Location**: Runtime composition from `/workspaces/configs/`

**Structure** (relevant subset):
```python
DictConfig:
  model:
    _target_: str                    # Callable reference (PROBLEMATIC FOR SERIALIZATION)
    architecture: str
    num_classes: int
  data:
    _target_: str                    # Callable reference
    batch_size: int
  train:
    optimizer:
      _target_: str                  # Callable reference
      lr: float
    logger:
      wandb:
        log_config: bool             # Constraint configuration point
```

**Serialization Characteristics**:
- **Serializable**: Scalar primitives (int, float, str, bool)
- **Not Serializable**: `_target_` fields, nested DictConfig with callables, instantiated objects
- **Workaround**: `OmegaConf.to_yaml()` converts to string (loses type safety)

### 3. Logger Instantiation Context (Runtime)

**Location**: `/workspaces/ocr/pipelines/orchestrator.py` lines 179-203

**State Flow**:
```
1. Load Hydra config → DictConfig tree
2. Process WandB logger config:
   a. Extract standardize_name → Generate run name if true
   b. Extract log_config → Build config dict if true  ⚠️ CONSTRAINT ENFORCED HERE
   c. Remove internal keys (standardize_name, log_config, enabled, etc.)
3. Instantiate WandbLogger with processed config
4. Pass to Trainer
```

**Constraint Enforcement Point**:
- Line 183: `log_config = bool(logger_cfg.get("log_config", False))`
- Line 185-188: If `true`, creates `config` dict with full YAML dump
  - **Risk**: YAML dump includes `_target_` strings (safe)
  - **Safe because**: `OmegaConf.to_yaml()` converts to string, not raw objects
  - **Trade-off**: Loses structured searchability in WandB dashboard

## Configuration Constraint Documentation Entity

**Location**: `/workspaces/AgentQMS/specs/tier2-framework/configuration.spec.md`

**New Section Structure**:
```markdown
## 5. Serialization Constraints

### WandB Configuration Logging
**Constraint**: Disable full config logging when Hydra DictConfig contains `_target_` fields
**Rationale**: WandB's serialization cannot handle callable references
**Default**: `log_config: false` in `/workspaces/configs/train/logger/wandb.yaml`
**Override**: Users can enable with `train.logger.wandb.log_config=true` (may fail)
**Visibility**: Essential config values captured via run naming convention
```

## No New Data Structures

This feature explicitly avoids:
- ❌ Configuration wrapper classes
- ❌ Serialization helper utilities
- ❌ Custom logger subclasses
- ❌ Validation schema definitions
- ❌ State management objects

**Rationale**: Simplification principle (FR-005 success criterion)

## State Transitions

None. This is a static configuration constraint, not a runtime state machine.

## Validation Rules

- **VR-001**: `log_config` must be boolean (enforced by Hydra type coercion)
- **VR-002**: If `log_config: true` and serialization fails, error message SHOULD reference constraint docs
- **VR-003**: Default value in base config MUST be `false`

**Validation Timing**:
- Design-time: YAML syntax validation (Hydra)
- Runtime: No validation (fail-fast on serialization error)
- Post-deployment: Smoke test verifies no serialization crashes

## Field Lifecycle

| Field | Phase | Responsibility |
|-------|-------|---------------|
| `log_config` in YAML | Configuration | Framework user (scientist) |
| `log_config` extraction | Instantiation | Orchestrator (line 183) |
| `config` dict building | Instantiation | Orchestrator (line 190-192) if enabled |
| Serialization attempt | Logger init | WandB library |
| Error handling | N/A | User-facing error message (future enhancement) |

## References to External Types

- `lightning.pytorch.loggers.WandbLogger`: External library class
- `omegaconf.DictConfig`: Configuration object type
- `hydra.utils.instantiate`: Object factory pattern

No new type definitions required.
