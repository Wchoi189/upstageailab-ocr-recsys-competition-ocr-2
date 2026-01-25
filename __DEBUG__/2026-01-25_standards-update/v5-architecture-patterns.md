---
type: standard
category: architecture
tier: 1
version: "1.0"
ads_version: "1.0"
status: active
created: 2026-01-25 21:00 (KST)
updated: 2026-01-25 21:00 (KST)
---

# V5 Architecture Patterns

## Overview

Documents the enforced patterns for V5.0 "Domains First" architecture. These patterns are REQUIRED and violations will fail CI/CD.

## Core Principles

1. **Single Source of Truth:** Each configuration has ONE standard location
2. **Fail Fast:** Explicit errors over silent fallbacks
3. **Domain Separation:** Clear boundaries between detection, recognition, and shared code
4. **Optimizer Agnostic Models:** Models do not create or configure optimizers
5. **Explicit Configuration:** Everything in Hydra configs, nothing hardcoded

---

## Required Patterns

### 1. Optimizer Configuration

**Standard Location:** `config.train.optimizer`

**Status:** ✅ ENFORCED (since 2026-01-25)

**Configuration:**
```yaml
# configs/train/optimizer/adam.yaml
# @package train.optimizer  # ← Must use this package directive

_target_: torch.optim.Adam
lr: 0.001
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0.0001
```

**Usage in Domain Config:**
```yaml
# configs/domain/detection.yaml
defaults:
  - /train/optimizer: adam  # References train/optimizer/adam.yaml
```

**Lightning Module:**
```python
class OCRPLModule(pl.LightningModule):
    def configure_optimizers(self):
        """V5 Standard: Single path, fail-fast"""
        if not hasattr(self.config, "train") or not hasattr(self.config.train, "optimizer"):
            raise ValueError(
                "V5 Hydra config missing: config.train.optimizer is required.\n"
                "Legacy model.get_optimizers() is no longer supported.\n"
                "See configs/train/optimizer/adam.yaml for template."
            )
        
        return instantiate(self.config.train.optimizer, params=self.model.parameters())
```

**Enforcement:**
- ❌ Alternative paths not allowed: `config.model.optimizer`, `config.optimizer`
- ❌ Model methods not allowed: `model.get_optimizers()`
- ❌ Fallback defaults not allowed: Silent Adam creation
- ✅ Must raise `ValueError` with migration guide

**Migration:** See [v5-optimizer-migration.md](../../docs/reference/v5-optimizer-migration.md)

---

### 2. Scheduler Configuration

**Standard Location:** `config.train.scheduler`

**Status:** ⚠️ IN PROGRESS (not yet enforced)

**Configuration:**
```yaml
# configs/train/scheduler/cosine.yaml
# @package train.scheduler

_target_: torch.optim.lr_scheduler.CosineAnnealingLR
T_max: 100
eta_min: 1.0e-6
```

**Usage:**
```python
def configure_optimizers(self):
    optimizer = instantiate(self.config.train.optimizer, params=self.model.parameters())
    
    if hasattr(self.config.train, "scheduler"):
        scheduler = instantiate(self.config.train.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
    
    return optimizer
```

---

### 3. Model Architecture Configuration

**Standard Location:** `config.model.architectures`

**Status:** ✅ ENFORCED

**Configuration:**
```yaml
# configs/model/architectures/dbnet_atomic.yaml
# @package model.architectures

_target_: ocr.core.models.architecture.OCRModel
_recursive_: false  # ← CRITICAL: Prevents premature optimizer instantiation

encoder:
  _target_: ocr.core.models.encoder.get_encoder_by_cfg
  # ... encoder config

decoder:
  _target_: ocr.core.models.decoder.get_decoder_by_cfg
  # ... decoder config

head:
  _target_: ocr.core.models.head.get_head_by_cfg
  # ... head config

loss:
  _target_: ocr.core.models.loss.get_loss_by_cfg
  # ... loss config
```

**Usage:**
```python
# ocr/core/models/__init__.py
def get_model_by_cfg(config: DictConfig) -> nn.Module:
    architectures = config.model.architectures
    
    # CRITICAL: Disable recursive instantiation
    return hydra.utils.instantiate(
        architectures,
        cfg=config,
        _recursive_=False  # ← Prevents Hydra from instantiating nested optimizer
    )
```

**Enforcement:**
- ✅ Must use `_recursive_=False` when instantiating models
- ✅ Architecture configs must use `@package model.architectures`
- ❌ No inline architecture definitions

---

### 4. Dataset Configuration

**Standard Location:** `config.data.datasets`

**Status:** ✅ ENFORCED

**Configuration:**
```yaml
# configs/data/datasets/detection.yaml
# @package data.datasets

train:
  _target_: ocr.data.datasets.get_datasets_by_cfg
  # ... dataset config

val:
  _target_: ocr.data.datasets.get_datasets_by_cfg
  # ... dataset config
```

---

### 5. Checkpoint Loading

**Standard:** 2-level fallback maximum

**Status:** ✅ ENFORCED (since 2026-01-25)

**Allowed Fallbacks:**
1. Direct load with `strict=True`
2. torch.compile prefix handling (`._orig_mod.`)

**Not Allowed:**
- ❌ `strict=False` (except in migration scripts)
- ❌ Automatic "model." prefix removal
- ❌ > 2 fallback levels
- ❌ Silent failures

**Implementation:**
```python
def load_state_dict_with_fallback(
    model: torch.nn.Module,
    state_dict: Mapping[str, Any],
    strict: bool = True
) -> tuple[list[str], list[str]]:
    """Load state dict with torch.compile prefix handling ONLY."""
    
    # Try 1: Direct load
    try:
        result = model.load_state_dict(state_dict, strict=strict)
        return result.missing_keys, result.unexpected_keys
    except RuntimeError as e:
        # Only handle torch.compile prefix
        if "_orig_mod" not in str(e):
            raise RuntimeError(
                f"Checkpoint incompatible with current model architecture.\n"
                f"Original error: {e}\n"
                f"For legacy checkpoints, use: scripts/checkpoints/convert.py"
            ) from e
    
    # Try 2: torch.compile prefix handling ONLY
    modified = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    result = model.load_state_dict(modified, strict=strict)
    return result.missing_keys, result.unexpected_keys
```

**Enforcement:**
- Max 2 fallback levels
- Must raise explicit errors with migration guidance
- No automatic format conversion

---

## Domain Separation Patterns

### Directory Structure

```
ocr/
├── core/                    # Domain-agnostic shared code
│   ├── models/             # Base architecture (shared)
│   ├── lightning/          # Lightning modules (shared)
│   └── utils/              # Truly shared utilities
│
├── domains/                # Domain-specific code
│   ├── detection/          # Detection-only
│   │   ├── models/        # Detection models
│   │   ├── utils/         # Detection utils
│   │   └── module.py      # DetectionPLModule
│   │
│   └── recognition/        # Recognition-only
│       ├── models/        # Recognition models
│       ├── utils/         # Recognition utils
│       └── module.py      # RecognitionPLModule
│
└── data/                   # Data handling (shared)
    ├── datasets/          # Dataset factories
    └── transforms/        # Transform factories
```

### Import Rules

**Allowed:**
```python
# ✅ Domain importing from core
from ocr.core.models import OCRModel

# ✅ Domain importing from shared data
from ocr.data.datasets import get_datasets_by_cfg

# ✅ Domain self-imports
from ocr.domains.detection.utils import parse_boxes
```

**Not Allowed:**
```python
# ❌ Cross-domain imports
from ocr.domains.detection.utils import parse_boxes  # In recognition code

# ❌ Core importing from domain
from ocr.domains.detection.models import DBNet  # In core code

# ❌ Domain-specific logic in core
def process_output(output, domain):  # In core/utils
    if domain == "detection":
        # Detection logic
    elif domain == "recognition":
        # Recognition logic
```

---

## Configuration Structure

### Hydra Config Hierarchy

```yaml
# configs/main.yaml
defaults:
  - domain: ???              # MUST be provided
  - hardware: default
  - global: default
  - _self_

# Domain configs define their own needs
# configs/domain/detection.yaml
defaults:
  - /global/default
  - /model/architectures/dbnet_atomic
  - /model/loss/db_loss
  - /data/datasets/detection
  - /train/optimizer: adam     # ← Standard location
  - _self_

task: detection
batch_size: 16
```

### Package Directives

**Required Format:**
```yaml
# Tier 1: Top-level packages
# @package _global_          # Merges at root level
# @package _group_           # Merges at group level

# Tier 2: Nested packages
# @package train.optimizer   # Creates config.train.optimizer
# @package model.architectures  # Creates config.model.architectures
# @package data.datasets     # Creates config.data.datasets
```

**Rules:**
- Optimizer MUST use: `@package train.optimizer`
- Scheduler MUST use: `@package train.scheduler`
- Architecture MUST use: `@package model.architectures`
- Dataset MUST use: `@package data.datasets`

---

## Error Handling Patterns

### Fail-Fast with Helpful Messages

**Required Format:**
```python
if not <condition>:
    raise <ExceptionType>(
        "What went wrong: <clear description>\n"
        "Why it happened: <explain the cause>\n"
        "How to fix it: <specific actionable steps>\n"
        "See: <link to docs/examples>"
    )
```

**Example:**
```python
if not hasattr(config, "train") or not hasattr(config.train, "optimizer"):
    raise ValueError(
        "What: V5 Hydra config missing: config.train.optimizer is required.\n"
        "Why: Legacy model.get_optimizers() is no longer supported.\n"
        "Fix: Add 'defaults: [/train/optimizer: adam]' to your domain config.\n"
        "See: docs/reference/v5-optimizer-migration.md"
    )
```

### No Silent Fallbacks

**Bad:**
```python
try:
    value = config.train.optimizer
except:
    value = default_optimizer()  # ❌ Silent fallback
```

**Good:**
```python
if not hasattr(config.train, "optimizer"):
    raise ValueError("config.train.optimizer required")  # ✅ Explicit error

value = config.train.optimizer
```

---

## Testing Patterns

### Model Testing

**Models should be tested WITHOUT optimizer logic:**
```python
def test_model_forward():
    """Test model forward pass only"""
    model = MyModel(config)
    output = model(dummy_input)
    assert output.shape == expected_shape
    # No optimizer testing here!

def test_optimizer_configuration():
    """Test optimizer config in Lightning module"""
    module = MyLightningModule(model, dataset, config)
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
```

---

## Validation Tools

### AST Grep Rules

```yaml
# rules/v5-optimizer-location.yaml
id: v5-optimizer-location
language: python
rule:
  any:
    - pattern: config.model.optimizer
    - pattern: config.optimizer
message: "Use config.train.optimizer only (V5 standard)"
severity: error
```

### Pre-commit Hook

```python
# AgentQMS/tools/check_v5_compliance.py
def check_optimizer_location(file_path: Path) -> list[str]:
    """Check optimizer is accessed from correct location."""
    errors = []
    content = file_path.read_text()
    
    bad_patterns = [
        "config.model.optimizer",
        "config.optimizer",
        "model.get_optimizers",
        "model._get_optimizers_impl"
    ]
    
    for pattern in bad_patterns:
        if pattern in content:
            errors.append(
                f"{file_path}: Found deprecated pattern '{pattern}'. "
                f"Use 'config.train.optimizer' instead."
            )
    
    return errors
```

---

## Migration Guidelines

### For New Code

- ✅ Follow patterns in this document
- ✅ Use templates from `configs/` directory
- ✅ Review [anti-patterns.md](./anti-patterns.md)

### For Legacy Code

- 📖 See [v5-optimizer-migration.md](../../docs/reference/v5-optimizer-migration.md)
- 🔍 Run `uv run python AgentQMS/tools/check_v5_compliance.py`
- ✅ Fix violations before merging

---

## Related Standards

- [Anti-Patterns Catalog](./anti-patterns.md)
- [Naming Conventions](../tier1-sst/naming-conventions.yaml)
- [File Placement Rules](../tier1-sst/file-placement-rules.yaml)

---

**Version History:**
- 1.0 (2026-01-25): Initial patterns from Legacy Purge Audit

**Maintenance:**
- Review after major refactors
- Update when adding new required patterns
- Deprecate obsolete patterns with migration guide
