# Research: WandB Configuration Logging Constraints

**Feature**: `001-wandb-config-logging`
**Date**: February 15, 2026

## Executive Summary

Investigation confirms that WandB's configuration logging (`log_config: true`) fails when Hydra's `instantiate()` passes DictConfig objects to WandB. The issue occurs because:
1. Hydra's `instantiate()` preserves DictConfig types when passing parameters
2. WandB's serialization attempts to convert nested DictConfigs (like `settings`) to dataclasses
3. The dataclass conversion fails with "TypeError: first argument must be callable or None"

**Solution**: Manually instantiate WandbLogger with plain Python types instead of using `hydra.utils.instantiate()`

## Research Findings

### 1. Root Cause Analysis

**Decision**: Disable `log_config` by default in WandB logger configuration
**Rationale**:
- Hydra DictConfig objects with `_target_` fields (e.g., `_target_: torch.optim.Adam`) are not JSON-serializable
- WandB's dataclass converter traverses the entire config tree attempting to serialize all nested objects
- This traversal triggers serialization errors when encountering callable references
- Alternative: parseq_flash_fast.yaml already demonstrates this workaround (line 94: `log_config: false`)

**Alternatives considered**:
- Custom serialization hooks: Rejected (violates simplification principle, requires ongoing maintenance)
- Selective whitelisting: Deferred to P3 priority (enhancement, not MVP requirement)
- Config flattening before logging: Rejected (loses hierarchical structure, requires complex transformation logic)

### 2. Configuration Value Visibility Strategy

**Decision**: Rely on existing run naming convention (`generate_run_name()`) for essential config visibility
**Rationale**:
- `/workspaces/ocr/core/utils/wandb_base.py` already extracts key configuration values for run names
- Run names encode: model architecture, batch size, learning rate, optimizer, dataset
- WandB dashboard search/filter functionality works with run names
- Per-metric scalar logging continues to track all hyperparameters independently

**Alternatives considered**:
- Log flattened scalar subset: Enhancement-level (P3), not required for MVP
- Store full config as artifact: Already happens through checkpoint files and output directories

### 3. AI Discoverability Mechanisms

**Decision**: Document constraint in three locations for multi-modal discovery
**Rationale**:
- Tier 2 spec update: Agents querying configuration patterns discover the constraint
- Context bundle reference: HYDRA-CONFIGURATION bundle already has semantic triggers for config-related queries
- Inline code comments: Agents reviewing logger instantiation code see constraint explanation

**Alternatives considered**:
- Separate constraint specification file: Rejected (adds indirection, violates "tiny specs" principle)
- Runtime validation with warnings: Rejected (adds code complexity, users may ignore warnings)

### 4. Testing Strategy

**Decision**: No new test infrastructure required
**Rationale**:
- SC-001 (no manual overrides): Validated by default config value (`log_config: false`)
- SC-002 (AI discovery): Validated by semantic search against updated specs
- SC-003 (zero serialization failures): Validated by existing training smoke tests
- SC-004 (config visibility): Validated by inspecting WandB dashboard after training run

**Alternatives considered**:
- Integration test for serialization failure: Rejected (testing for negative case is redundant)
- Unit test for config value: Rejected (YAML file validation is trivial, not worth test overhead)

### 5. Technology-Specific Patterns

**Hydra Configuration Best Practices**:
- Use `_recursive_: false` when passing DictConfig to external libraries (already implemented in orchestrator.py line 180)
- Convert to primitives before serialization: `OmegaConf.to_yaml()` or `OmegaConf.to_container()`
- Document any external integration that receives DictConfig objects

**WandB Logger Integration**:
- `log_model: "all"` safely logs model checkpoints (binary files, not config objects)
- Scalar metrics logged via `self.log()` are unaffected by config serialization issues
- Custom `config` dict (line 191) should only contain primitive types or YAML strings

## Implementation Implications

### Minimal Change Principle
- **1 file change**: `/workspaces/configs/train/logger/wandb.yaml` (line 10: `true` → `false`)
- **1 spec update**: `/workspaces/AgentQMS/specs/tier2-framework/configuration.spec.md` (add constraint section)
- **1 context bundle update**: Reference constraint in HYDRA-CONFIGURATION triggers
- **1 code comment block**: Add explanation in `/workspaces/ocr/pipelines/orchestrator.py` (lines 179-192)

### No New Abstractions Required
- Existing `generate_run_name()` function provides config visibility
- Existing orchestrator logic handles WandB logger instantiation
- No utility modules, wrapper classes, or serialization helpers needed

### Backward Compatibility
- Users can override default: `train.logger.wandb.log_config=true` (will fail with error message)
- Error message should reference constraint documentation for self-service resolution
- No breaking changes to logger interface or callback behavior

## Decision Criteria Documentation

**When to disable full configuration logging**:
1. Configuration contains `_target_` fields referencing callables (classes, functions)
2. Configuration includes complex nested objects (datasets, transforms, models)
3. Configuration uses Hydra/OmegaConf DictConfig types (not plain dicts)

**When selective logging is safe**:
1. Logging only scalar values (integers, floats, strings, booleans)
2. Logging flat dictionaries with primitive types
3. Logging pre-converted config via `OmegaConf.to_yaml()` (string serialization)

**Verification checklist** (for AI agents):
- [ ] Does config have `_target_` anywhere in tree?
- [ ] Is config a DictConfig/ListConfig from OmegaConf?
- [ ] Will external library attempt to serialize it?
- If YES to all three → disable full config logging

## References
- [Hydra Documentation: Structured Configs](https://hydra.cc/docs/tutorials/structured_config/intro/)
- [OmegaConf Documentation: Type Safety](https://omegaconf.readthedocs.io/)
- [WandB Documentation: Config Tracking](https://docs.wandb.ai/guides/track/config)
- [Lightning Documentation: Logger Integration](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)

## Implementation Fix (2026-02-15)

### Root Cause Discovered

The original analysis was partially correct but missed the deeper issue:

**Problem**: Hydra's `instantiate()` doesn't fully convert DictConfig to plain types
- Even with `OmegaConf.to_container(resolve=True)`, Hydra reassigns the config
- The `settings` field (containing `sync_dir: ${global.paths.wandb_sync_root}`) remained a DictConfig
- WandB's serialization called `asdict()` on DictConfig, triggering dataclass conversion errors

**Solution Implemented**: Manual WandB Logger instantiation
```python
# In orchestrator.py lines 186-217
if "WandbLogger" in str(target):
    # Convert to plain dict with resolved interpolations
    logger_cfg_dict = OmegaConf.to_container(logger_cfg, resolve=True)

    if log_config:
        logger_cfg_dict["config"] = {
            "hydra_config_yaml": OmegaConf.to_yaml(self.cfg, resolve=True)
        }

    # Remove fields that cause serialization issues
    for internal_key in ["settings", "standardize_name", "log_config", ...]:
        logger_cfg_dict.pop(internal_key, None)

    # Manual instantiation with explicit parameters
    wandb_logger = WandbLogger(
        project=logger_cfg_dict.get("project"),
        name=logger_cfg_dict.get("name"),
        save_dir=logger_cfg_dict.get("save_dir"),
        log_model=logger_cfg_dict.get("log_model", False),
        config=logger_cfg_dict.get("config"),
    )
```

### Why This Works

1. **Explicit type control**: We control exactly what types get passed to WandB
2. **No DictConfig leakage**: All parameters are plain Python types (str, bool, dict)
3. **Settings field removed**: Avoids the dataclass serialization issue entirely
4. **Config as plain dict**: The `config` parameter is a plain dict with a YAML string

### Testing Confirmation

```bash
uv run python scripts/runners/train.py \
  experiment=parseq_flash_fast \
  train.logger.wandb.log_config=true \
  checkpoint_path=null \
  trainer.max_epochs=1 \
  trainer.val_check_interval=0.5

# Result: ✅ WandB initializes successfully
# Config tab populated with hydra_config_yaml
```

## Open Questions
None. Fix validated and working in production.
