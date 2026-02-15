# Implementation Plan: WandB Configuration Logging Constraints

**Branch**: `001-wandb-config-logging` | **Date**: February 15, 2026 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from`/workspaces/specs/001-wandb-config-logging/spec.md`

## Summary

Establish safe defaults and documentation for WandB configuration logging to prevent serialization failures when Hydra configurations contain callable references (`_target_` fields). The solution disables full config logging by default (`log_config: false`) while maintaining visibility through run naming conventions, and documents the constraint for AI agent discoverability across specifications, context bundles, and inline code comments.

**Technical Approach**: Minimal change strategy - modify one configuration default value, document constraint in three discovery locations, add explanatory code comments. No new utilities, abstractions, or validation layers.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: Hydra 1.3+, OmegaConf 2.3+, WandB 0.16+, PyTorch Lightning 2.1+
**Storage**: N/A (configuration-only feature)
**Testing**: pytest (existing test suite, no new tests required)
**Target Platform**: Linux server (ML training environment)
**Project Type**: Single (ML training pipeline)
**Performance Goals**: N/A (constraint documentation, no runtime performance impact)
**Constraints**: Zero breaking changes, backward compatible, no new dependencies
**Scale/Scope**: Affects all training experiments using WandB logger (~15 experiment configs, 1 default logger config)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### AgentQMS Constitution Compliance

**Principle 1: The Registry is Truth**
- ✅ **PASS** (Pre-Design): Changes documented in specs first (this plan), then code implementation
- ✅ **PASS** (Post-Design): Implementation plan confirms spec-first approach with 4-file modification strategy
- Action: Update `/workspaces/AgentQMS/specs/tier2-framework/configuration.spec.md` before modifying config files

**Principle 2: Atomic Modularity**
- ✅ **PASS** (Pre-Design): All documentation files <600 tokens
  - `spec.md`: ~550 tokens
  - `research.md`: ~480 tokens
  - `data-model.md`: ~520 tokens
  - `quickstart.md`: ~410 tokens
  - `contracts/configuration-contract.md`: ~580 tokens
- ✅ **PASS** (Post-Design): Implementation confirms zero new functions, only 7-line comment block addition
- Code changes: 1-line config value modification, 7-line comment block, 3-line spec section, 3 keywords

**Principle 3: Explicit Context**
- ✅ **PASS** (Pre-Design): Context bundle (HYDRA-CONFIGURATION) updated with new keywords for constraint discovery
- ✅ **PASS** (Post-Design): Three-path discovery strategy (spec, bundle, inline comments) ensures targeted context loading
- Agents load only configuration.spec.md when working on config-related tasks

**Principle 4: Continuous Validation**
- ✅ **PASS** (Pre-Design): Success criteria measurable without new validation tooling (SC-001 through SC-005)
- ✅ **PASS** (Post-Design): Validation strategy uses existing training runners + semantic search tools, no new test infrastructure
- Constraint violation (user overrides to `log_config: true`) fails fast with clear error from WandB library

**Constitution Grade**: APPROVED ✅ (Pre & Post Design)
**Justification**: Documentation-focused feature with minimal code changes, follows all AgentQMS principles. Post-design review confirms no new abstractions, explicit context targeting, and atomic modularity maintained.

## Project Structure

### Documentation (this feature)

```text
specs/001-wandb-config-logging/
├── spec.md                          # Feature specification (input)
├── plan.md                          # This file (implementation plan)
├── research.md                      # Phase 0: Root cause analysis and decision rationale
├── data-model.md                    # Phase 1: Configuration entities and constraints
├── quickstart.md                    # Phase 1: User and AI agent guide
└── contracts/
    └── configuration-contract.md    # Phase 1: Configuration schema and behavioral contracts
```

### Source Code (repository root)

This feature modifies existing configuration and documentation files only. No new source files created.

```text
/workspaces/
├── configs/
│   └── train/
│       └── logger/
│           └── wandb.yaml                                    # MODIFY: Change log_config default
├── AgentQMS/
│   ├── specs/
│   │   └── tier2-framework/
│   │       └── configuration.spec.md                         # MODIFY: Add constraint section
│   └── .agentqms/
│       └── plugins/
│           └── context_bundles/
│               └── hydra-configuration.yaml                  # MODIFY: Add keyword triggers
└── ocr/
    └── pipelines/
        └── orchestrator.py                                   # MODIFY: Add explanatory comments
```

**Structure Decision**: Constraint-focused feature with zero new files. All changes are updates to existing configuration, specification, and code documentation. Aligns with simplification principle (SC-005: "No new utility modules or abstraction layers").

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**No violations** - Constitution Check passed all principles. No complexity justification required.

## Implementation Phases

### Phase 0: Research & Analysis ✅ COMPLETE
- [x] Analyze root cause of serialization failures
- [x] Identify decision criteria for safe config logging
- [x] Document alternatives and trade-offs
- [x] Generate `research.md` artifact

### Phase 1: Design & Documentation ✅ COMPLETE
- [x] Create data model for configuration entities (`data-model.md`)
- [x] Define configuration contracts (`contracts/configuration-contract.md`)
- [x] Write user and AI agent quickstart guide (`quickstart.md`)
- [x] Update agent context files

### Phase 2: Implementation (Minimal Code Changes)
**Estimated Time**: 30 minutes

#### 2.1 Specification Update (5 minutes)
**File**: `/workspaces/AgentQMS/specs/tier2-framework/configuration.spec.md`
- **Location**: After section 4 "Hydra Merging Pitfalls"
- **Action**: Add new section "## 5. Serialization Constraints"
- **Content**: Document CONFIG-WANDB-001 constraint (WandB log_config + _target_ incompatibility)
- **Validation**: Run semantic search for "WandB config constraints" → Verify spec appears in results

#### 2.2 Context Bundle Update (5 minutes)
**File**: `/workspaces/AgentQMS/.agentqms/plugins/context_bundles/hydra-configuration.yaml`
- **Location**: `triggers.keywords` list (after line 30)
- **Action**: Add keywords: `serialization`, `wandb`, `log_config`
- **Validation**: Query context bundling system with "WandB configuration logging" → Verify HYDRA-CONFIGURATION bundle recommended

#### 2.3 Configuration Default Change (2 minutes)
**File**: `/workspaces/configs/train/logger/wandb.yaml`
- **Location**: Line 10
- **Action**: Change `log_config: true` to `log_config: false`
- **Add Comment**:
  ```yaml
  # CONFIG-WANDB-001: Disabled to prevent serialization errors with Hydra _target_ fields
  # See: /workspaces/specs/001-wandb-config-logging/spec.md for details
  log_config: false
  ```
- **Validation**: Run training with default config → No serialization errors

#### 2.4 Code Documentation (10 minutes)
**File**: `/workspaces/ocr/pipelines/orchestrator.py`
- **Location**: Before line 179 (`if "WandbLogger" in str(target):`)
- **Action**: Add comment block:
  ```python
  # WandB Configuration Logging Constraint (CONFIG-WANDB-001)
  # -----------------------------------------------------
  # Full config logging (log_config=true) disabled by default to prevent
  # serialization errors when Hydra DictConfig contains callable references
  # (_target_ fields). Essential config visibility maintained via run naming.
  # Override at own risk: train.logger.wandb.log_config=true
  # Spec: /workspaces/specs/001-wandb-config-logging/spec.md
  ```
- **Validation**: AI agents reading orchestator code discover constraint explanation

### Phase 3: Validation & Testing
**Estimated Time**: 20 minutes

#### 3.1 Success Criteria Validation

**SC-001: Training launches without manual overrides**
```bash
cd /workspaces
uv run python scripts/runners/train.py \
  experiment=parseq_flash_fast \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10
# Expected: ✅ Trainer initializes without serialization errors
```

**SC-002: AI agents discover constraint within 2 searches**
```bash
# Test semantic search
uv run python AgentQMS/tools/utilities/suggest_context.py "WandB configuration logging constraints"
# Expected: ✅ HYDRA-CONFIGURATION bundle recommended

# Simulate AI agent query to specification
grep -r "CONFIG-WANDB-001" AgentQMS/specs/tier2-framework/
# Expected: ✅ configuration.spec.md appears in results
```

**SC-003: Zero new serialization failures**
```bash
# Run 3 different experiments with default config
for exp in parseq_flash_fast rec_optimized_rtx3090 vlm_base; do
  uv run python scripts/runners/train.py \
    experiment=$exp \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=5 || echo "FAIL: $exp"
done
# Expected: ✅ All 3 experiments initialize successfully
```

**SC-004: Essential config visible in WandB dashboard**
- Manual: Launch training run → Check WandB dashboard → Verify run name contains model/batch/lr
- Expected: ✅ Run name format: `{model}-b{batch}-lr{lr}-{optimizer}-{dataset}`

**SC-005: No new utility modules introduced**
```bash
# Check that no new Python files created in this changeset
git diff --name-status main...001-wandb-config-logging | grep "^A.*\.py$"
# Expected: ✅ Empty output (no new .py files)
```

#### 3.2 Backward Compatibility Test

**Test Override Behavior**:
```bash
# User explicitly enables config logging (accepts risk)
uv run python scripts/runners/train.py \
  experiment=parseq_flash_fast \
  train.logger.wandb.log_config=true \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=5
# Expected: ⚠️  May fail with serialization error (acceptable, user override)
# Error message should mention DictConfig serialization
```

### Phase 4: Rollout & Documentation

#### 4.1 Branch & Pull Request
- **Branch**: `001-wandb-config-logging` (already created by setup script)
- **PR Title**: "feat: Establish WandB config logging constraints for Hydra compatibility"
- **PR Description**: Reference spec.md, list 4 modified files, link success criteria validation
- **Reviewers**: Tag AI quality assurance

#### 4.2 Documentation Checklist
- [x] Feature specification (`spec.md`)
- [x] Research rationale (`research.md`)
- [x] Data model documentation (`data-model.md`)
- [x] Quickstart guide (`quickstart.md`)
- [x] Configuration contracts (`contracts/configuration-contract.md`)
- [x] Implementation plan (`plan.md` - this file)
- [ ] Update CHANGELOG.md with constraint establishment notice
- [ ] Add entry to `.specify/memory/decisions.md` (if exists)

#### 4.3 Communication Plan
**Internal (Team)**:
- Slack announcement: "Default WandB config logging now disabled. Run names still show key values. Full config in checkpoints/outputs."
- Link quickstart guide for override instructions

**External (AI Agents)**:
- Update `.ai-instructions/` if project-specific agent docs exist
- Verify context bundling system surfaces constraint docs for WandB-related queries

## File Modification Strategy

### 1. Configuration File (`wandb.yaml`)
**Change Type**: Single-line value modification + 2-line comment
```diff
  standardize_name: true
- log_config: true
+ # CONFIG-WANDB-001: Disabled to prevent serialization errors with Hydra _target_ fields
+ # See: /workspaces/specs/001-wandb-config-logging/spec.md
+ log_config: false
```

**Risk**: Low (default value change, user can override)
**Rollback**: Change back to `true` (reverts to previous behavior with serialization risk)

### 2. Specification File (`configuration.spec.md`)
**Change Type**: New section addition (after existing section 4)
```markdown
## 5. Serialization Constraints

### WandB Configuration Logging (CONFIG-WANDB-001)
**Rule**: Set `log_config: false` when Hydra config contains `_target_` fields
**Rationale**: WandB dataclass converter cannot serialize callable references
**Default**: Enforced in `/workspaces/configs/train/logger/wandb.yaml`
**Visibility**: Essential config values captured via `generate_run_name()`
**Override**: Possible via CLI (`train.logger.wandb.log_config=true`), may fail
**Discovery**: Keywords: `wandb`, `log_config`, `serialization`, `_target_`
```

**Risk**: None (documentation only, non-breaking)
**Rollback**: Remove section 5 (no impact on code behavior)

### 3. Context Bundle (`hydra-configuration.yaml`)
**Change Type**: Keyword list extension
```diff
  triggers:
    keywords:
      # ... existing keywords ...
+     - serialization
+     - wandb
+     - log_config
```

**Risk**: None (improves discoverability, doesn't break existing triggers)
**Rollback**: Remove 3 keywords (context bundling reverts to previous trigger set)

### 4. Orchestrator Code (`orchestrator.py`)
**Change Type**: Comment block insertion (before line 179)
```diff
          if is_config(logger_cfg):
              target = logger_cfg.get("_target_", "")
+             # WandB Configuration Logging Constraint (CONFIG-WANDB-001)
+             # -----------------------------------------------------
+             # Full config logging (log_config=true) disabled by default to prevent
+             # serialization errors when Hydra DictConfig contains callable references
+             # (_target_ fields). Essential config visibility maintained via run naming.
+             # Override at own risk: train.logger.wandb.log_config=true
+             # Spec: /workspaces/specs/001-wandb-config-logging/spec.md
              if "WandbLogger" in str(target):
```

**Risk**: None (comments only, zero runtime impact)
**Rollback**: Delete comment block (no functional change)

## Testing Strategy

### No New Test Infrastructure Required

**Rationale**:
- Configuration value change validated by existing training smoke tests
- Documentation discoverability validated by semantic search tooling
- Backward compatibility ensured by Hydra's override system

### Validation Approach

| Success Criterion | Validation Method | Tooling |
|-------------------|-------------------|---------|
| SC-001: No manual overrides | Run training with default config | Existing `train.py` runner |
| SC-002: AI discovery | Semantic search + grep | `suggest_context.py`, `grep` |
| SC-003: Zero failures | Multiple experiment runs | Existing `train.py` runner |
| SC-004: Config visibility | Manual WandB dashboard inspection | WandB web UI |
| SC-005: No new modules | Git diff analysis | `git diff --name-status` |

### Test Scenarios

**Positive Tests** (Expected to succeed):
1. Default training with `log_config: false` → ✅ Trainer initializes
2. Run name generation → ✅ Contains model/batch/lr/optimizer
3. Metrics logging → ✅ Scalars appear in WandB dashboard
4. Checkpoint saving → ✅ Full config in checkpoint metadata

**Negative Tests** (Expected behavior documented):
1. Override to `log_config: true` with `_target_` in config → ⚠️ May fail (user responsibility)
2. Semantic search for non-existent constraint → ❌ Should not return false positives

**Boundary Tests**:
1. Config without `_target_` fields + `log_config: true` → ✅ Should work (safe case)
2. Empty logger config → ✅ Falls back to Lightning defaults

## Rollout Plan

### Pre-Deployment Checklist
- [ ] All Phase 2 modifications complete (4 files modified)
- [ ] All Phase 3 validation tests passing (5 success criteria validated)
- [ ] PR review approved (code comments + config change reviewed)
- [ ] Documentation committed to branch `001-wandb-config-logging`

### Deployment Steps

**Step 1: Merge to Main (Immediate Effect)**
```bash
# After PR approval
git checkout main
git merge 001-wandb-config-logging --no-ff
git push origin main
```

**Impact**:
- New training runs use `log_config: false` by default
- Existing runs unaffected (checkpoints already saved)
- No service restarts required (config loaded per-run)

**Step 2: Validate in Production (Next Training Run)**
```bash
# First production run after merge
uv run python scripts/runners/train.py experiment=production_baseline
# Monitor: No serialization errors in logs
```

**Step 3: Monitor for 1 Week**
- Track training run success rate (should remain 100%)
- Monitor Slack/support channels for user confusion
- Verify WandB dashboard still shows essential metrics

### Rollback Procedure

**If issues detected**:
```bash
# Quick rollback (revert single line in wandb.yaml)
git revert <merge_commit_sha>
# Or emergency config-only fix:
echo "log_config: true" > /workspaces/configs/train/logger/wandb.yaml
```

**Rollback Impact**:
- Reverts to previous behavior (serialization errors may return)
- No data loss (checkpoints and metrics unaffected)
- Users can immediately override if needed

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Users confused by empty Config tab | Medium | Low | Quickstart guide explains alternatives |
| Override fails silently | Low | Medium | Error message from WandB library guides user |
| AI agents miss constraint docs | Low | Medium | Three discovery paths (spec, bundle, comments) |
| Breaking change for existing workflows | Very Low | High | Override system preserves backward compat |
| Performance regression | None | None | Zero runtime overhead (boolean check only) |

**Overall Risk Level**: **LOW**
**Justification**: Documentation-focused change with explicit override path, extensive validation, and zero new abstractions.

## Success Metrics

### Post-Deployment (Week 1)
- **Training Success Rate**: 100% (no serialization failures)
- **User Support Tickets**: 0 related to config logging
- **AI Agent Discovery**: <2 searches to find constraint docs

### Long-Term (Month 1)
- **Constraint Violations**: Track users overriding to `log_config: true` (should be <5% of runs)
- **Documentation Clarity**: Measure time-to-resolution for config visibility questions
- **Specification Coverage**: Ensure all new Hydra integrations reference serialization constraints

## References

- **Feature Spec**: `/workspaces/specs/001-wandb-config-logging/spec.md`
- **Research**: `/workspaces/specs/001-wandb-config-logging/research.md`
- **Data Model**: `/workspaces/specs/001-wandb-config-logging/data-model.md`
- **Quickstart**: `/workspaces/specs/001-wandb-config-logging/quickstart.md`
- **Contracts**: `/workspaces/specs/001-wandb-config-logging/contracts/configuration-contract.md`
- **AgentQMS Constitution**: `/workspaces/AgentQMS/governance/constitution.md`
- **Hydra Documentation**: https://hydra.cc/docs/tutorials/structured_config/intro/
- **WandB Config Tracking**: https://docs.wandb.ai/guides/track/config

---

**Implementation Status**: Phase 0 & Phase 1 complete (research + design). Ready for Phase 2 (code changes).
**Next Action**: Execute Phase 2 implementation tasks (estimated 30 minutes).
**Branch**: `001-wandb-config-logging`
**Planning Complete**: February 15, 2026
