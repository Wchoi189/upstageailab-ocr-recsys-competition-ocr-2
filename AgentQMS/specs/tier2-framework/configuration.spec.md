---
ads_version: '2.0'
id: 'FW-CONFIGURATION.SPEC'
type: 'rule_set'
tier: 2
priority: 'high'
spec_version: '1.0.0'
updated: '2026-02-03'
description: 'Configuration Specification for framework tier'
---

# Configuration Specification

**Tier**: 2 (Framework)
**Scope**: Configuration Management, Hydra, and Externalization.

## 1. Configuration Standards

**Core Principle**: Domain-First Architecture.
*   **Root**: `configs/`
*   **Structure**: `domain/`, `model/`, `data/`, `training/`.
*   **Constraint**: All configurations must be externalized (YAML/JSON), never hardcoded.

### Externalization Checklist
1.  Is the value likely to change experiments? -> **Yes**: Config.
2.  Is it a secret? -> **Yes**: Env Var (via `omega_conf`).
3.  Is it structural? -> **No**: Constant.

## 2. Hydra v5 Rules

**Strict Enforcement**:
*   Use `hydra.main` for entry points.
*   Use `instantiate` for object creation.
*   **Prohibited**: `sys.argv` parsing (Use Hydra CLI).

### Key Patterns
| Pattern | Usage | Example |
| :--- | :--- | :--- |
| **Domain Switch** | `python runners/train.py domain=ocr` | Switch entire behavior set. |
| **Model Override** | `python runners/train.py model=vlm_v2` | Swap model architecture. |
| **Debug Mode** | `python runners/train.py debug=true` | Activate verbose logging. |

## 3. Config Bloat Policy
*   Limit nesting depth to 4 levels.
*   Split files > 200 lines.
4. Hydra Merging Pitfalls
*   **Avoid**: `@package _group_` in domain configs (e.g., `domain/recognition.yaml`).
*   **Reason**: It forces content into a literal `_group_` key instead of merging into the parent node, breaking domain detection logic.

## 5. Serialization Constraints

### WandB Configuration Logging (CONFIG-WANDB-001)
**Rule**: Set `log_config: false` when Hydra config contains `_target_` fields
**Rationale**: WandB dataclass converter cannot serialize callable references
**Default**: Enforced in `/workspaces/configs/train/logger/wandb.yaml`
**Visibility**: Essential config values captured via `generate_run_name()`
**Override**: Possible via CLI (`train.logger.wandb.log_config=true`), may fail
**Discovery**: Keywords: `wandb`, `log_config`, `serialization`, `_target_`
