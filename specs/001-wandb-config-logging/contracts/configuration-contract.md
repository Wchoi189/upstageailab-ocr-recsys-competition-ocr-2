# Contracts: WandB Configuration Logging Constraints

**Feature**: `001-wandb-config-logging`
**Date**: February 15, 2026

## No API Contracts Required

This feature establishes **configuration constraints** and **documentation standards**, not external interfaces or APIs.

## Configuration Contract

### Input: YAML Configuration
**File**: `/workspaces/configs/train/logger/wandb.yaml`

**Schema**:
```yaml
type: object
properties:
  _target_:
    type: string
    const: "lightning.pytorch.loggers.WandbLogger"
  log_config:
    type: boolean
    default: false                    # CONSTRAINT: Must be false for configs with _target_
    description: "Enable full config logging to WandB (may fail with Hydra DictConfig)"
  project:
    type: string
  log_model:
    type: string
    enum: ["all", "best", false]
  save_dir:
    type: string
  enabled:
    type: boolean
  log_recognition_images:
    type: boolean
  standardize_name:
    type: boolean
```

**Validation**: Schema enforced by Hydra at runtime (dynamic typing)

### Output: Logger Behavior

**When `log_config: false` (default)**:
- WandB logger instantiates successfully
- Config tab in WandB dashboard is empty
- Run name encodes key configuration values
- Training proceeds without serialization errors

**When `log_config: true` (user override)**:
- If config has `_target_` fields → Serialization error at trainer init
- If config is pure scalars → Config tab populates successfully
- Error message directs user to constraint documentation

## Documentation Contract

### Specification Update
**File**: `/workspaces/AgentQMS/specs/tier2-framework/configuration.spec.md`

**Required Section**:
```markdown
## 5. Serialization Constraints

### WandB Configuration Logging
*Constraint ID*: `CONFIG-WANDB-001`
*Rule*: Disable `log_config` when Hydra config contains `_target_` fields
*Rationale*: WandB serialization cannot handle callable references
*Default*: `log_config: false`
*Override Risk*: May crash if DictConfig has `_target_` anywhere in tree
```

**Discovery Contract**: AI agents searching for "configuration constraints", "WandB", or "serialization" must find this section within 2 semantic search queries.

### Context Bundle Update
**File**: `/workspaces/AgentQMS/.agentqms/plugins/context_bundles/hydra-configuration.yaml`

**Required Addition**:
```yaml
triggers:
  keywords:
    # ... existing keywords ...
    - serialization        # NEW
    - wandb               # NEW
    - log_config          # NEW
```

**Discovery Contract**: Context bundling system returns HYDRA-CONFIGURATION bundle when task mentions WandB config issues.

### Code Comment Contract
**File**: `/workspaces/ocr/pipelines/orchestrator.py`

**Location**: Before line 179 (`if "WandbLogger" in str(target):`)

**Required Content**:
```python
# WandB Configuration Logging Constraint (CONFIG-WANDB-001):
# - log_config=false by default to prevent serialization errors
# - Hydra DictConfig with _target_ fields cannot be JSON-serialized
# - Essential config visibility maintained via generate_run_name()
# - See: /workspaces/specs/001-wandb-config-logging/spec.md
```

## Behavioral Contract

### Backward Compatibility

**Guaranteed**:
- Existing experiments with `log_config: false` continue to work (no change)
- Experiments with explicit `log_config: true` override still work (user accepts risk)
- Logger instantiation API unchanged (no breaking changes to orchestrator interface)

**Not Guaranteed**:
- Config tab population in WandB dashboard (by design, constraint enforcement)
- Serialization success if user overrides to `log_config: true` with incompatible config

### Error Handling Contract

**Current State** (no change in this feature):
- Serialization failures raise exception from WandB library
- Exception propagates to user as trainer initialization error
- Error message is technical (JSON serialization traceback)

**Future Enhancement** (out of scope):
- Catch serialization errors
- Show user-friendly message: "Config logging failed. See [constraint docs]. Continuing with log_config=false."
- Auto-fallback to safe default

## Non-Functional Contracts

### Performance
- **No runtime overhead**: Single boolean check in orchestrator (line 183)
- **No additional I/O**: YAML file size unchanged (~30 lines)
- **No memory impact**: Config dictionary not built when `log_config: false`

### Maintainability
- **No new dependencies**: Uses existing Hydra/OmegaConf/WandB libraries
- **No abstraction layers**: Direct config value modification
- **Minimal touch points**: 4 files modified (1 config, 1 spec, 1 context bundle, 1 code comment)

## Testing Contract

### Validation Criteria

**SC-001**: Training launches without manual `log_config` overrides
- **Test**: Run default experiment → No serialization error
- **Pass**: Exit code 0, WandB logger initialized

**SC-002**: AI agents discover constraint within 2 searches
- **Test**: Semantic search "WandB config logging" → Find spec section
- **Pass**: Specification retrieved in first or second query

**SC-003**: Zero serialization failures post-implementation
- **Test**: Run 5 different experiments with default config
- **Pass**: All 5 complete training initialization without errors

**SC-004**: Essential config visible in WandB dashboard
- **Test**: Run experiment → Check run name encoding
- **Pass**: Run name contains model, batch_size, lr, optimizer

## Change Tracking

| File | Change Type | Contract Impact |
|------|-------------|-----------------|
| `configs/train/logger/wandb.yaml` | Default value | User-facing (safe behavior) |
| `AgentQMS/specs/tier2-framework/configuration.spec.md` | Documentation | AI discoverability |
| `AgentQMS/.agentqms/plugins/context_bundles/hydra-configuration.yaml` | Metadata | AI discoverability |
| `ocr/pipelines/orchestrator.py` | Code comment | Developer/AI guidance |

**No interfaces changed, no breaking changes.**
