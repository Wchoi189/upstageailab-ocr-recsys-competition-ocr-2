# Hydra Configuration Patterns & Failure Modes

> **Purpose**: Technical reference for Hydra 1.3.2 configuration architecture patterns, common failure modes, and resolution strategies for "Domains First" v5.0 architecture.

---

## Critical Rules

### 1. The Flattening Rule

**Rule**: Files using `# @package _group_` MUST NOT contain a top-level key matching the folder name.

**Failure Mode**: Double namespacing (e.g., `data.data.train_num_samples` instead of `data.train_num_samples`)

**Example - INCORRECT**:
```yaml
# configs/data/default.yaml
# @package _group_
data:  # ❌ Creates data.data namespace
  train_num_samples: 1000
```

**Example - CORRECT**:
```yaml
# configs/data/default.yaml
# @package _group_
train_num_samples: 1000  # ✅ Creates data.train_num_samples
```

---

### 2. Absolute Interpolation Law

**Rule**: All cross-file references MUST use absolute paths from root namespace.

**Failure Mode**: `InterpolationKeyError: 'key' not found`

**Pattern**:
- ❌ NEVER: `${train_transform}`
- ✅ ALWAYS: `${data.transforms.train_transform}`
- ✅ ALWAYS: `${global.paths.data_dir}`

---

### 3. Package Directive Behavior

**Critical Understanding**: `# @package _group_` automatically wraps content in parent folder's namespace.

**Location**: [configs/data/default.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/data/default.yaml)
**Package**: `# @package _group_`
**Result**: All keys placed under `data.*` namespace

**Double-Wrap Bug**: If file contains `data:` key + `# @package _group_`, result is `data.data.*`

---

## Component Placement Rules

| Component | Correct Location | Rationale |
|-----------|-----------------|-----------|
| Neural network layers | `model/architectures/` | Pure structural definition |
| Tokenizer | `domain/` or `data/` | Data-dependent mapping |
| Loss function | `domain/` | Task-dependent logic |
| Optimizer/LR | `train/optimizer/` | Hardware/schedule config |
| Transforms | `data/transforms/` | Image alteration logic |
| Dataset paths | `data/datasets/` | Source identity only |

---

## Atomic Architecture Pattern

**Principle**: Model presets should define ONLY neural network structure, not training logic.

### Example: Recognition Model

```yaml
# configs/model/architectures/parseq.yaml
# @package _group_
# v5.0 | Atomic Recognition Architecture

_target_: ocr.domains.recognition.models.PARSeq

backbone:
  _target_: ocr.core.models.encoder.TimmBackbone
  model_name: resnet18
  pretrained: true

decoder:
  _target_: ocr.domains.recognition.models.decoder.PARSeqDecoder
  d_model: 512
  nhead: 8
  num_layers: 12

max_len: 25
```

**Anti-Pattern**: Including `optimizer:` or `loss:` in architecture files causes "Passive Refactor" cycle.

---

## Domain Injection Pattern

**Problem**: Tokenizers in architecture files prevent language reuse.

**Solution**: Domain controller injects data-dependent components.

```yaml
# configs/domain/recognition.yaml
# @package _group_
defaults:
  - /model/architectures: parseq
  - /data/datasets: recognition_canonical
  - _self_

# Domain controller injects tokenizer
model:
  tokenizer:
    _target_: ocr.domains.recognition.data.tokenizer.KoreanOCRTokenizer
    char_path: ${global.paths.root_dir}/ocr/data/charset.json
    max_len: 25

# Loss function lives in domain (task-dependent)
train:
  loss:
    _target_: torch.nn.CrossEntropyLoss
    ignore_index: 0
```

---

## Callback Flattening Pattern

**Failure Mode**: `train.callbacks.early_stopping.early_stopping` (double nesting)

### INCORRECT
```yaml
# configs/train/callbacks/early_stopping.yaml
# @package _group_
early_stopping:  # ❌ Creates double namespace
  _target_: lightning.pytorch.callbacks.EarlyStopping
  monitor: "val/hmean"
```

### CORRECT
```yaml
# configs/train/callbacks/early_stopping.yaml
# @package _group_
_target_: lightning.pytorch.callbacks.EarlyStopping
monitor: "val/hmean"
min_delta: 0.0
patience: 5
mode: "max"
```

---

## Multi-Logger Aliasing

**Problem**: Multiple loggers without aliasing cause namespace collision.

**Solution**: Use `@_group_.alias` syntax to separate logger configs.

### Syntax Breakdown: `wandb@_group_.wandb_logger`

- `wandb` = File selection (`configs/train/logger/wandb.yaml`)
- `@_group_` = Package directive (place in parent namespace)
- `.wandb_logger` = Custom alias (unique key for this logger)

### Implementation

```yaml
# configs/train/logger/default.yaml
# @package _group_
defaults:
  - wandb@_group_.wandb_logger  # Creates train.logger.wandb_logger
  - csv@_group_.csv_logger      # Creates train.logger.csv_logger
  - _self_
```

```yaml
# configs/train/logger/wandb.yaml
# @package _group_
_target_: lightning.pytorch.loggers.WandbLogger
project: "receipt-ocr-project"
log_model: "all"
save_dir: "${global.paths.wandb}"

settings:
  offline: false
```

**Result**: `train.logger.wandb_logger._target_` and `train.logger.csv_logger._target_` coexist.

---

## Dataset Source Identity Pattern

**Principle**: Dataset configs define ONLY paths and metadata, NOT transforms.

```yaml
# configs/data/datasets/canonical.yaml
# @package _group_
# v5.0 | Dataset Source Paths ONLY

dataset_base_path: "${global.paths.datasets_root}"

train_dataset:
  _target_: ocr.data.datasets.ValidatedOCRDataset
  image_path: ${data.dataset_base_path}/images/train
  annotation_path: ${data.dataset_base_path}/jsons/train.json

val_dataset:
  _target_: ocr.data.datasets.ValidatedOCRDataset
  image_path: ${data.dataset_base_path}/images_val_canonical
  annotation_path: ${data.dataset_base_path}/jsons/val.json
```

**Transforms injected separately** via domain controller or `data/transforms/`.

---

## Transform Atomicity Pattern

```yaml
# configs/data/transforms/base.yaml
# @package _group_
# v5.0 | Atomic Transform Definitions

default_interpolation: 1

train_transform:
  _target_: ocr.data.datasets.DBTransforms
  transforms:
    - _target_: albumentations.LongestMaxSize
      max_size: 640
      interpolation: ${data.transforms.default_interpolation}
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

---

## Domain Isolation Pattern

**Principle**: Domain controllers nullify irrelevant keys from other domains.

```yaml
# configs/domain/recognition.yaml
# @package _group_
defaults:
  - /model/presets: parseq
  - /data/datasets: recognition_canonical
  - /train/optimizer: adamw
  - /train/logger: default
  - /train/callbacks: default
  - _self_

# Nullify detection-specific keys
detection: null
max_polygons: null
shrink_ratio: null
thresh_min: null
thresh_max: null

# Recognition-specific overrides
recognition:
  max_label_length: 25
  charset: korean
  decode_mode: greedy
```

**Rationale**: Prevents CUDA segfaults and logic leakage from unused domain configs.

---

## Validation Commands

### Verify Configuration Composition
```bash
python scripts/utils/show_config.py main domain=detection
python scripts/utils/show_config.py main domain=recognition
```

### Check for Interpolation Errors
Look for:
- `InterpolationKeyError: 'key' not found` → Namespace mismatch
- Nested duplicate keys (e.g., `data.data.*`) → Flattening violation

---

## Known Failure Modes

### 1. Orphaned Logic Files

**Example**: `configs/data/datasets/db.yaml` referencing `/dataloaders/default` (non-existent)

**Resolution**: Delete orphaned files; move logic to domain controller or experiment configs.

---

### 2. Preprocessing Misplacement

**Problem**: `configs/data/datasets/preprocessing.yaml` in wrong location.

**Reason**: Preprocessing is a *transformation*, not a *data source*.

**Fix**: Move to `configs/data/transforms/preprocessing.yaml`

---

### 3. Passive Refactor Cycle

**Symptom**: Model presets include `- /train/optimizer: adamw` in defaults.

**Problem**: Optimizer override invisible to user, causes training inconsistencies.

**Fix**: Remove all `optimizer:` and `loss:` from `model/architectures/`; manage in `train/` or `domain/`.

---

### 4. Shadow Interpolations

**Symptom**: `${encoder_path}` fails with "key not found"

**Problem**: Relative interpolation without absolute root anchor.

**Fix**: Define in `main.yaml` or use absolute path `${global.paths.encoder_path}`

---

## Migration Checklist

- [ ] All `# @package _group_` files have no top-level key matching folder name
- [ ] All interpolations use absolute paths (`${namespace.key}`)
- [ ] All paths anchor to `${global.paths.*}`
- [ ] Model architectures contain NO `optimizer:` or `loss:` keys
- [ ] Tokenizers moved to domain controllers or data configs
- [ ] Callbacks/loggers flattened (no redundant wrapper keys)
- [ ] Multi-logger configs use aliasing (`@_group_.alias`)
- [ ] Dataset configs define ONLY paths, NOT transforms
- [ ] Domain controllers nullify irrelevant cross-domain keys
- [ ] Orphaned files (invalid defaults) deleted
- [ ] Preprocessing moved to `data/transforms/`

---

## Verification Script

```python
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

def verify_flattening():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="main", overrides=["domain=recognition"])
        
        # Check logger aliasing
        if "wandb_logger" in cfg.train.logger:
            print("✅ Logger Aliasing: Found 'train.logger.wandb_logger'")
            if "_target_" in cfg.train.logger.wandb_logger:
                print("✅ Logger Flattening: '_target_' at alias root")
        
        # Check callback flattening
        if "early_stopping" in cfg.train.callbacks:
            if "_target_" in cfg.train.callbacks.early_stopping:
                print("✅ Callback Flattening: Correct structure")
        
        # Check interpolation resolution
        try:
            OmegaConf.to_container(cfg, resolve=True)
            print("✅ Interpolation: All paths resolved")
        except Exception as e:
            print(f"❌ Interpolation Error: {e}")

if __name__ == "__main__":
    verify_flattening()
```

---

## References

- Hydra Version: 1.3.2
- Architecture: "Domains First" v5.0
- Source: Session `20260118_013857_session-refactor-execution`
