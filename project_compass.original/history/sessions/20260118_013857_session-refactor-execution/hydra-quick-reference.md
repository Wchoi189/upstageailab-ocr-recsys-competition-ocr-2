# Hydra v5.0 Quick Reference Card

> **1-Page Cheat Sheet** for Hydra 1.3.2 "Domains First" Architecture

---

## 🔴 Critical Rules (Never Violate)

| Rule | Pattern | Failure Mode |
|------|---------|--------------|
| **Flattening** | `# @package _group_` → NO top-level key matching folder | `data.data.*` double namespace |
| **Absolute Paths** | `${data.transforms.train}` NOT `${train}` | `InterpolationKeyError` |
| **Anchoring** | `${global.paths.data_dir}` NOT `./data` | Resolution failure |

---

## 📍 Component Placement

```
model/architectures/  → Neural network layers ONLY
domain/              → Tokenizers, loss functions, task logic
train/optimizer/     → Optimizer, LR schedules
data/datasets/       → Paths and metadata ONLY
data/transforms/     → Image transformations
```

---

## ✅ Correct Patterns

### Flattened Callback
```yaml
# configs/train/callbacks/early_stopping.yaml
# @package _group_
_target_: lightning.pytorch.callbacks.EarlyStopping
monitor: "val/hmean"
patience: 5
```

### Multi-Logger Aliasing
```yaml
# configs/train/logger/default.yaml
# @package _group_
defaults:
  - wandb@_group_.wandb_logger
  - csv@_group_.csv_logger
```

### Atomic Architecture
```yaml
# configs/model/architectures/parseq.yaml
# @package _group_
_target_: ocr.domains.recognition.models.PARSeq
backbone:
  _target_: ocr.core.models.encoder.TimmBackbone
  model_name: resnet18
# NO optimizer or loss here!
```

### Domain Injection
```yaml
# configs/domain/recognition.yaml
# @package _group_
defaults:
  - /model/architectures: parseq
  - _self_
model:
  tokenizer:  # Inject data-dependent components
    _target_: ocr.domains.recognition.data.tokenizer.KoreanOCRTokenizer
```

---

## ❌ Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| `data:` key in `data/default.yaml` with `@package _group_` | Double wrap → `data.data.*` | Remove `data:` wrapper |
| `${train_transform}` | Relative interpolation | `${data.transforms.train_transform}` |
| `optimizer:` in model preset | Passive refactor cycle | Move to `train/optimizer/` |
| `transforms:` in dataset config | Logic in source definition | Move to `data/transforms/` |
| Multiple loggers without alias | Namespace collision | Use `@_group_.alias` |

---

## 🔍 Debugging Commands

```bash
# Verify composition
python scripts/utils/show_config.py main domain=detection

# Check for errors
# Look for: InterpolationKeyError, nested duplicate keys
```

---

## 🛡️ Pre-Commit Checklist

- [ ] No top-level key matches folder name in `@package _group_` files
- [ ] All interpolations absolute (`${namespace.key}`)
- [ ] All paths anchor to `${global.paths.*}`
- [ ] Model architectures have NO `optimizer:` or `loss:`
- [ ] Callbacks/loggers flattened (no wrapper keys)
- [ ] Multi-loggers use aliasing

---

## 📖 Aliasing Syntax

`wandb@_group_.wandb_logger`
- `wandb` = File to load
- `@_group_` = Place in parent namespace
- `.wandb_logger` = Custom key (prevents collision)

**Result**: `train.logger.wandb_logger.*`

---

## 🚨 Common Failure Modes

| Symptom | Root Cause | Resolution |
|---------|------------|------------|
| `data.data.train_num_samples` | Double namespace | Remove redundant `data:` key |
| `InterpolationKeyError: 'train_transform'` | Relative interpolation | Use `${data.transforms.train_transform}` |
| Optimizer changes ignored | Model preset override | Remove optimizer from architecture |
| CUDA segfault in validation | Cross-domain key leakage | Nullify unused domain keys |

---

**Full Documentation**: [hydra_config_patterns_knowledge_base.md](file:///home/vscode/.gemini/antigravity/brain/f127f3e4-e4b1-45c8-ae9e-45291dff2672/hydra_config_patterns_knowledge_base.md)
