# Quickstart: WandB Configuration Logging (Safe Defaults)

**Feature**: `001-wandb-config-logging`
**Audience**: Data Scientists, ML Engineers, AI Agents
**Time**: 2 minutes read

## What Changed

WandB configuration logging is now **disabled by default** to prevent serialization errors with Hydra configs containing `_target_` fields.

## For Users: Default Behavior

### Before (would crash)
```bash
uv run python scripts/runners/train.py experiment=parseq_flash_fast
# ❌ Error: Cannot serialize DictConfig with _target_ fields
```

### After (works out of box)
```bash
uv run python scripts/runners/train.py experiment=parseq_flash_fast
# ✅ Training starts successfully
# ✅ Run name encodes key config: model, batch size, LR, optimizer
# ✅ Metrics logged normally to WandB dashboard
```

**No action required** - training works with default config.

## Configuration Visibility

### What You Still See in WandB

1. **Run Name** (auto-generated):
   ```
   parseq-b32-lr0.001-adam-mjsynth
   ```
   Includes: model architecture, batch size, learning rate, optimizer, dataset

2. **Metrics Tab**: All scalar metrics (loss, accuracy, etc.)

3. **Files Tab**: Full config saved in checkpoint metadata

4. **Model Tab**: Saved checkpoints (when `log_model: "all"`)

### What's Different

- **Config Tab**: Now empty (was causing crashes)
- **Workaround**: View config in Files tab or local `outputs/` directory

## Re-enabling Full Config Logging (Not Recommended)

If you need the Config tab populated **and** understand the risks:

```bash
# Override at runtime
uv run python scripts/runners/train.py \
  experiment=my_experiment \
  train.logger.wandb.log_config=true

# ⚠️ May crash if config has _target_ fields
# ⚠️ Error: "Cannot serialize callable reference"
```

## For AI Agents: Discovery Pattern

### When to Check This Constraint

**Triggers**:
- Modifying WandB logger configuration
- Adding new experiment tracking integrations
- Debugging serialization errors in training initialization

**Discovery**:
```bash
# Semantic search: "WandB config logging constraints"
# Returns: /workspaces/AgentQMS/specs/tier2-framework/configuration.spec.md

# Code search: "log_config" in configs/train/logger/wandb.yaml
# Shows: log_config: false  # Default safe value

# Context bundle: HYDRA-CONFIGURATION
# References: Serialization constraints section
```

### Decision Tree for Config Logging

```
Does config contain _target_ fields?
├─ YES → Set log_config: false (safe)
│   └─ Use run naming for visibility
├─ NO → log_config: true is safe
    └─ Full config tab will populate
```

### Code Location Reference

| File | Line | Purpose |
|------|------|---------|
| `/workspaces/configs/train/logger/wandb.yaml` | 10 | Default `log_config: false` |
| `/workspaces/ocr/pipelines/orchestrator.py` | 183 | Extract `log_config` value |
| `/workspaces/ocr/pipelines/orchestrator.py` | 190-192 | Build config dict if enabled |
| `/workspaces/ocr/core/utils/wandb_base.py` | 110+ | `generate_run_name()` function |

## Troubleshooting

### Symptom: "Cannot serialize config" error

**Cause**: `log_config: true` with Hydra config containing `_target_` fields

**Fix**:
```bash
# Method 1: Use default config (log_config: false)
uv run python scripts/runners/train.py experiment=my_experiment

# Method 2: Explicitly disable
uv run python scripts/runners/train.py \
  experiment=my_experiment \
  train.logger.wandb.log_config=false
```

### Symptom: Can't find config values in WandB

**Solution**: Check these locations:
1. Run name (filter by model, batch size, etc.)
2. Files tab → `hydra_config.yaml`
3. Local `outputs/<run_name>/hydra/` directory
4. Checkpoint file metadata

### Symptom: Need config search in WandB dashboard

**Workaround**:
1. Use run tags: `train.logger.wandb.tags=['batch32','lr0.001']`
2. Use run notes: `train.logger.wandb.notes='Config: batch=32, lr=0.001'`
3. Filter by run name patterns

## Examples

### Standard Training (Default Behavior)
```bash
cd /workspaces
uv run python scripts/runners/train.py \
  experiment=parseq_flash_fast \
  trainer.max_epochs=10

# ✅ Config logging disabled automatically
# ✅ Run name: parseq-b32-lr0.001-adam-mjsynth
```

### Override Config Logging (Advanced)
```bash
# Only if you removed all _target_ fields from config
uv run python scripts/runners/train.py \
  experiment=simple_experiment \
  train.logger.wandb.log_config=true \
  +train.logger.wandb.config.custom_note='Manual config tracking'
```

### Check Current Setting
```bash
# View current logger config
cat /workspaces/configs/train/logger/wandb.yaml | grep log_config
# Output: log_config: false
```

## Related Documentation

- **Feature Spec**: `/workspaces/specs/001-wandb-config-logging/spec.md`
- **Research**: `/workspaces/specs/001-wandb-config-logging/research.md`
- **Config Spec**: `/workspaces/AgentQMS/specs/tier2-framework/configuration.spec.md`
- **Context Bundle**: `/workspaces/AgentQMS/.agentqms/plugins/context_bundles/hydra-configuration.yaml`

## Questions?

- **Why disabled?**: Hydra's `_target_` fields (callable references) can't be JSON-serialized by WandB
- **Lost visibility?**: No - run names encode key values, metrics/files still logged
- **Can I re-enable?**: Yes, but may crash if config has `_target_` anywhere in tree
- **Better solution?**: Future: selective scalar whitelisting (P3 priority)
