# Hydra Architectural Laws - System Constraints

**Type:** Protocol
**Purpose:** Enforce strict architectural rules for Hydra configuration to prevent domain leakage, package errors, and configuration drift
**Target:** AI agents and developers working on Hydra configuration refactoring
**Priority:** CRITICAL - Must be followed for all configuration changes

---

## Overview

These are the **4 Fundamental Laws** that govern all Hydra configuration architecture. Violations of these laws lead to CUDA segfaults, domain contamination, and AI agent confusion.

---

## Law I: Absolute Resolution

**Principle:** Never assume configuration values based on YAML files alone.

### Rules
- ✅ **ALWAYS** request the resolved output to see final interpolated values
- ✅ **ALWAYS** verify `${interpolations}` are correctly resolved at runtime
- ❌ **NEVER** trust static YAML inspection for dynamic values

### Verification Command
```bash
python train.py --print-config
```

### Why This Matters
Static YAML files contain `${variable}` references that may resolve differently at runtime. The AI must see the **final resolved state** to understand actual values.

---

## Law II: Domain Controller Isolation

**Principle:** Domain files are the Source of Truth for task-specific logic and MUST enforce strict isolation.

### Rules
- ✅ Each domain file (`configs/domain/*.yaml`) controls ONE task type
- ✅ **MUST** explicitly nullify keys from other domains
- ✅ Domain separation prevents "ghost variables" that cause CUDA segfaults
- ❌ **NEVER** allow cross-domain key contamination

### Example: Recognition Domain
```yaml
# configs/domain/recognition.yaml
defaults:
  - _self_
  - /model/recognition/parseq
  - /data/recognition/lmdb

# CRITICAL: Nullify detection-specific keys
detection: null
max_polygons: null
shrink_ratio: null

# Recognition-specific configuration
recognition:
  max_label_length: 25
  charset: korean
```

### Validation Check
```python
# In recognition domain, these MUST be null or missing
assert cfg.get('max_polygons') is None
assert cfg.get('detection') is None
```

---

## Law III: Package Directive Discipline

**Principle:** Every configuration file MUST declare its namespace location using `@package` directives.

### Package Types

| Directive           | Usage                 | Files                       |
| ------------------- | --------------------- | --------------------------- |
| `@package _global_` | System-wide constants | `hardware/`, `experiment/`  |
| `@package _group_`  | Component libraries   | `model/`, `data/`, `train/` |

### Rules
- ✅ **MANDATORY** `@package` directive at top of every config file
- ✅ `_global_` ONLY for hardware and experiment files
- ✅ `_group_` for ALL component libraries
- ❌ **NO NAKED KEYS** - every key must live in its designated group

### Example: Model Component
```yaml
# @package _group_
# configs/model/recognition/parseq.yaml

architecture: parseq
backbone:
  type: resnet
  depth: 50
decoder:
  num_layers: 6
  hidden_dim: 512
```

### Common Violations
```yaml
# ❌ WRONG - Missing @package directive
batch_size: 32  # Where does this go? Root? data? hardware?

# ✅ CORRECT - Explicit package placement
# @package _group_
data:
  batch_size: 32
```

---

## Law IV: Archive Quarantine

**Principle:** Production configs must be separated from legacy/experimental code to prevent context pollution.

### Directory Rules

| Status           | Location                | Visibility              |
| ---------------- | ----------------------- | ----------------------- |
| **Production**   | `configs/`              | ✅ Active in training    |
| **Legacy**       | `archive/__LEGACY__/`   | ❌ Invisible to training |
| **Experimental** | `archive/__EXTENDED__/` | ❌ Invisible to training |
| **UI/Frontend**  | `archive/ui_configs/`   | ❌ Not for training      |

### Migration Requirements
```bash
# Move legacy files OUT of configs/
mv configs/__LEGACY__/ archive/__LEGACY__/
mv configs/__EXTENDED__/ archive/__EXTENDED__/

# Move UI-related configs
mv configs/training/logger/modes/ archive/ui_configs/
mv configs/training/logger/preprocessing_profiles.yaml archive/ui_configs/
```

### Why This Matters
- Reduces AI token count by ~40%
- Prevents AI from reading outdated logic
- Eliminates "which version is correct?" confusion

---

## Enforcement Checklist

Before any configuration change, verify:

- [ ] Resolved config inspected via `--print-config`
- [ ] Domain files explicitly nullify other domains
- [ ] All files have `@package` directives
- [ ] No legacy files in `configs/` directory
- [ ] Package placement matches group structure

---

## Violation Detection

Use the audit script to detect violations:

```python
# See: 02_SCRIPT_migration_auditor.py
python ocr-refactor-prep/hydra_refactor/02_SCRIPT_migration_auditor.py
```

---

## References

- **Hydra Guard Script:** `03_SCRIPT_hydra_guard.py`
- **Migration Auditor:** `02_SCRIPT_migration_auditor.py`
- **Global Anchor Template:** `04_TEMPLATE_global_paths.yaml`
