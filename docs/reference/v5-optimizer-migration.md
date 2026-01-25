# V5 Migration Guide - Legacy Optimizer Patterns

## Breaking Changes (2026-01-25)

This guide helps migrate from legacy optimizer patterns to V5 "Domains First" architecture.

## What Changed

### 1. Optimizer Configuration Location

**Old (REMOVED):**
```yaml
# configs/model/some_config.yaml
model:
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
```

**New (V5 Standard):**
```yaml
# configs/train/optimizer/adam.yaml
# @package train.optimizer

_target_: torch.optim.Adam
lr: 0.001
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0.0001
```

**In domain config:**
```yaml
# configs/domain/detection.yaml
defaults:
  - /train/optimizer: adam  # References train/optimizer/adam.yaml
```

### 2. Model Methods Removed

**Old (REMOVED):**
```python
class MyModel(nn.Module):
    def get_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return [optimizer], []
    
    def _get_optimizers_impl(self):
        # ... config detection logic ...
        return [optimizer], [scheduler]
```

**New (V5 Standard):**
```python
class MyModel(nn.Module):
    # Models should NOT create optimizers!
    # This is handled by Lightning module's configure_optimizers()
    pass
```

### 3. Lightning Module Changes

**Old (REMOVED):**
```python
# Lightning module would call:
optimizers, schedulers = self.model.get_optimizers()
```

**New (V5 Standard):**
```python
def configure_optimizers(self):
    # Reads from config.train.optimizer ONLY
    return instantiate(self.config.train.optimizer, params=self.model.parameters())
```

## Migration Steps

### Step 1: Update Experiment Configs

If your experiment uses `config.model.optimizer`:

```yaml
# OLD - Will raise ValueError
model:
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
```

**Action:** Add optimizer to defaults in domain config or experiment config:

```yaml
# NEW - Domain config
defaults:
  - /train/optimizer: adam  # Use existing template
  
# OR create custom optimizer
defaults:
  - /train/optimizer: custom_adam

# Then create configs/train/optimizer/custom_adam.yaml:
# @package train.optimizer
_target_: torch.optim.Adam
lr: 0.002  # Custom value
```

### Step 2: Remove Model Optimizer Methods

If your custom model has `get_optimizers()` or `_get_optimizers_impl()`:

```python
# DELETE these methods entirely
class MyCustomModel(nn.Module):
    # ❌ DELETE THIS
    # def get_optimizers(self):
    #     ...
    #
    # def _get_optimizers_impl(self):
    #     ...
    pass
```

### Step 3: Update Tests

If your tests mock or call optimizer methods:

```python
# OLD
optimizers, schedulers = model.get_optimizers()
assert len(optimizers) == 1

# NEW - Test at Lightning module level
from hydra.utils import instantiate
optimizer = instantiate(config.train.optimizer, params=model.parameters())
assert isinstance(optimizer, torch.optim.Optimizer)
```

### Step 4: Handle Legacy Checkpoints

If you have old checkpoints with "model." prefix:

**Option A: Convert once (RECOMMENDED)**
```bash
python scripts/checkpoints/convert_legacy_checkpoints.py \
  --input outputs/checkpoints/old_format.ckpt \
  --output outputs/checkpoints/v5_format.ckpt
```

**Option B: Load with manual filtering (for one-time use)**
```python
# In your loading script
state_dict = torch.load(checkpoint_path)
# Remove "model." prefix manually
new_state = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state)
```

## Error Messages and Solutions

### Error 1: Missing config.train.optimizer

```
ValueError: V5 Hydra config missing: config.train.optimizer is required.
Legacy model.get_optimizers() is no longer supported.
See configs/train/optimizer/adam.yaml for template.
```

**Solution:** Add optimizer to your config defaults:
```yaml
defaults:
  - /train/optimizer: adam  # or adamw
```

### Error 2: Incompatible Checkpoint

```
RuntimeError: Checkpoint incompatible with current model architecture.
For legacy checkpoints, use: scripts/checkpoints/convert_legacy_checkpoints.py
```

**Solution:** Convert your checkpoint:
```bash
python scripts/checkpoints/convert_legacy_checkpoints.py --input <old.ckpt> --output <new.ckpt>
```

### Error 3: Checkpoint Missing Keys

```
RuntimeError: Checkpoint missing 15 required keys: ['encoder.layer1.weight', ...]
This checkpoint is incompatible with current model architecture.
Convert it using scripts/checkpoints/convert_legacy_checkpoints.py
```

**Solution:** Your checkpoint and model architecture don't match. Either:
1. Use the correct checkpoint for your model
2. Train a new model from scratch
3. Migrate your old checkpoint format

## Examples

### Example 1: Detection Experiment

```yaml
# configs/experiment/my_detection.yaml
# @package _global_

defaults:
  - override /domain: detection
  - override /hardware: rtx3060
  - /train/optimizer: adamw  # ← Add this!
  - _self_

model:
  encoder:
    model_name: "resnet50"

trainer:
  max_epochs: 100
```

### Example 2: Recognition Experiment

```yaml
# configs/experiment/my_recognition.yaml
# @package _global_

defaults:
  - override /domain: recognition
  - /train/optimizer: adam  # ← Add this!
  - _self_

model:
  encoder:
    model_name: "parseq"

recognition:
  max_label_length: 25
  charset: korean
```

### Example 3: Custom Optimizer

```yaml
# configs/train/optimizer/custom_sgd.yaml
# @package train.optimizer

_target_: torch.optim.SGD
lr: 0.01
momentum: 0.9
weight_decay: 0.0001
nesterov: true
```

Then use in experiment:
```yaml
defaults:
  - /train/optimizer: custom_sgd
```

## Verification

Test your migration:

```bash
# Verify config composes correctly
python -c "
from hydra import compose, initialize_config_dir
from pathlib import Path

with initialize_config_dir(config_dir=str(Path.cwd() / 'configs'), version_base=None):
    cfg = compose(config_name='main', overrides=['experiment=YOUR_EXPERIMENT'])
    
    # Should find optimizer at config.train.optimizer
    assert hasattr(cfg.train, 'optimizer'), 'Missing config.train.optimizer'
    print(f'✅ Optimizer configured: {cfg.train.optimizer._target_}')
"

# Run fast_dev_run to test training loop
python runners/train.py experiment=YOUR_EXPERIMENT +trainer.fast_dev_run=True
```

## FAQ

**Q: Can I keep using `config.model.optimizer`?**  
A: No. This path is no longer supported. Use `config.train.optimizer` only.

**Q: My model needs a custom optimizer. Where do I put it?**  
A: Create a new file in `configs/train/optimizer/` with `@package train.optimizer`, then reference it in your defaults.

**Q: Can I still use model-level `get_optimizers()`?**  
A: No. Models should be optimizer-agnostic. Configure optimizers through Hydra configs only.

**Q: What happened to KIE domain?**  
A: Archived to `archive/kie_domain_2026_01_25/`. See ARCHIVE_README.md for restoration instructions if needed.

**Q: Will my old checkpoints work?**  
A: Maybe. If you get "incompatible checkpoint" errors, use the conversion script in `scripts/checkpoints/`.

## Support

For issues or questions:
1. Check error message - they now include migration guidance
2. Review examples in `configs/experiment/`
3. See working domains: detection, recognition
4. Consult: [V5 Architecture Guide](../../configs/README.md)

---

**Last Updated:** 2026-01-25  
**Migration Deadline:** None - enforced immediately  
**Rollback:** Not supported - V5 is the only supported pattern
