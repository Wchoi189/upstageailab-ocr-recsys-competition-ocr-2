# Hydra Override Patterns Quick Reference

## When to Use `+` Prefix

### ✅ Use `+` (Add New Config Group)
- Config is **NOT** in `base.yaml` defaults
- Adding a new config group that doesn't exist
- Example: `+ablation=model_comparison` (ablation not in defaults)

### ❌ Don't Use `+` (Override Existing)
- Config **IS** in `base.yaml` defaults
- Overriding an existing config group
- Example: `logger=wandb` (logger is in defaults, use without `+`)

## Override Rules

### Rule 1: Check `base.yaml` defaults
If config group is listed in `configs/base.yaml` defaults:
```yaml
defaults:
  - model: default
  - logger: consolidated
  - trainer: default
```
Then use **without** `+`:
```bash
logger=wandb        # ✅ Correct (logger in defaults)
model=default       # ✅ Correct (model in defaults)
```

### Rule 2: New Config Groups
If config group is **NOT** in defaults:
```bash
+ablation=model_comparison    # ✅ Correct (ablation not in defaults)
+new_feature=value            # ✅ Correct (new feature)
```

### Rule 3: Nested Overrides
For nested configs, use path notation:
```bash
model/architectures=dbnet     # ✅ Override nested config
+model/new_component=value    # ✅ Add new nested component
```

## Common Patterns

### Working Patterns ✅
```bash
# Basic overrides
trainer.max_epochs=10
seed=123

# Group overrides (in defaults)
data=canonical
logger=wandb
model=default
trainer=default

# New groups (not in defaults)
+ablation=model_comparison
+hardware=rtx3060_12gb

# Multirun
trainer.max_epochs=5,10,15 -m
```

### Failing Patterns ❌
```bash
# Multiple values error
+logger=wandb        # ❌ Fails: logger already in defaults
+data=canonical       # ❌ Fails: data already in defaults

# Invalid syntax
override logger: wandb    # ❌ Invalid syntax
```

## Configuration Groups in Defaults

From `configs/base.yaml`:
- `model` - Use without `+`
- `evaluation` - Use without `+`
- `paths` - Use without `+`
- `logger` - Use without `+`
- `trainer` - Use without `+`
- `callbacks` - Use without `+`
- `debug` - Use without `+`

## Configuration Groups NOT in Defaults

These require `+`:
- `ablation` - Use with `+`
- `hardware` - Use with `+`
- Any new/custom groups - Use with `+`

## Testing Overrides

Run the test suite:
```bash
python tests/unit/test_hydra_overrides.py
```

## Quick Decision Tree

```
Is config in base.yaml defaults?
├─ YES → Use without + (e.g., logger=wandb)
└─ NO  → Use with + (e.g., +ablation=model_comparison)
```

## Examples from Codebase

### ✅ Correct Usage
```python
# From diagnose_cuda_issue.py
"+hardware=rtx3060_12gb_i5_16core"  # hardware not in defaults
```

### ❌ Problematic Usage
```python
# From test_hydra_overrides.py
"+logger=wandb"  # Fails: logger in defaults, should be logger=wandb
```

## Migration Notes

When moving configs to `__LEGACY__/`:
- Override patterns may need updating
- Test all override patterns after migration
- Document any breaking changes
