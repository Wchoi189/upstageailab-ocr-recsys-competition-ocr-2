# Hydra Configuration System Audit Prompt

## Context

You are auditing the Hydra configuration system for an OCR project. The project has undergone refactoring and now uses `configs/_base/` as the foundation for all configurations. However, legacy configurations still exist alongside the new architecture, creating confusion about which configurations are active and which override patterns are required.

## Audit Objectives

1. **Identify Legacy vs New Architecture Configurations**
   - Determine which configs belong to the old architecture
   - Identify which configs use the new `configs/_base/` foundation
   - Map dependencies and relationships between configs

2. **Analyze Override Patterns**
   - Identify which configs require explicit `+` prefix for overrides
   - Determine which configs can be overridden without `+`
   - Document the override rules for each configuration group

3. **Assess Removal Impact**
   - Identify unused or redundant configurations
   - Find all references to each config (code, scripts, documentation)
   - Assess impact of removing legacy configs
   - Document configuration options that would be lost if removed

4. **Propose Migration Strategy**
   - Recommend which configs can be safely removed
   - Suggest which configs should be moved to `__LEGACY__/` folder
   - Provide migration path for any configs that need updating

## Current Configuration Structure

### New Architecture (Foundation)
- `configs/_base/` - Foundation configs (core, data, logging, model, preprocessing, trainer)
- `configs/base.yaml` - Main base configuration
- Entry-point configs: `train.yaml`, `test.yaml`, `predict.yaml`, etc.

### Configuration Groups
- `configs/model/` - Model configurations (architectures, encoder, decoder, head, loss, presets)
- `configs/data/` - Data configurations (base, canonical, craft, datasets, transforms)
- `configs/trainer/` - Trainer configurations
- `configs/logger/` - Logger configurations
- `configs/callbacks/` - Callback configurations
- `configs/evaluation/` - Evaluation configurations
- `configs/paths/` - Path configurations
- `configs/hydra/` - Hydra-specific configurations
- `configs/ui/` - UI configurations
- `configs/ui_meta/` - UI metadata configurations

### Known Issues
- Mixed architectures causing confusion
- Unclear which overrides require `+` prefix
- Legacy configs may still be referenced in code
- Some configs may be duplicated between old and new systems

## Audit Tasks

### Task 1: Architecture Classification

For each configuration file in `configs/`, determine:

1. **Architecture Type**:
   - [ ] New architecture (uses `configs/_base/` foundation)
   - [ ] Legacy architecture (old system)
   - [ ] Hybrid (references both)
   - [ ] Unknown (needs investigation)

2. **Dependencies**:
   - Which configs does it depend on?
   - Which configs depend on it?
   - Is it referenced in `defaults:` sections?

3. **Usage**:
   - Is it referenced in code? (search codebase)
   - Is it used in scripts? (search runners/, scripts/)
   - Is it documented? (search docs/)
   - Is it used in UI? (search ui/, apps/)

### Task 2: Override Pattern Analysis

For each configuration group, determine:

1. **Override Requirements**:
   - Does it require `+` prefix? (e.g., `+model/architectures=dbnet`)
   - Can it be overridden without `+`? (e.g., `model=default`)
   - What causes "Multiple values" errors?

2. **Override Rules**:
   - If config is in `defaults:` in base.yaml → use `config=value` (no `+`)
   - If config is NOT in `defaults:` → use `+config=value` (with `+`)
   - Document exceptions and edge cases

3. **Test Override Patterns**:
   - Test each config group with both `+` and without `+`
   - Document which patterns work and which fail
   - Note any ambiguous cases

### Task 3: Reference Analysis

For each configuration file, find:

1. **Code References**:
   ```bash
   # Search for references
   grep -r "config_name" --include="*.py" .
   grep -r "configs/path/to/config" --include="*.py" .
   ```

2. **Script References**:
   - Check `runners/` for config usage
   - Check `scripts/` for config references
   - Check command-line examples

3. **Documentation References**:
   - Check `docs/` for config mentions
   - Check changelog for config history
   - Check README files

4. **UI References**:
   - Check `ui/` for config usage
   - Check `apps/` for config references
   - Check UI metadata configs

### Task 4: Impact Assessment

For each legacy configuration:

1. **Removal Impact**:
   - [ ] Safe to remove (no references found)
   - [ ] Has references (list all)
   - [ ] Critical (actively used)
   - [ ] Unknown (needs manual review)

2. **Lost Functionality**:
   - What configuration options would be lost?
   - Are these options available in new architecture?
   - Can they be migrated to new system?

3. **Migration Path**:
   - Can it be migrated to `configs/_base/` system?
   - What changes are needed?
   - Is migration worth the effort?

### Task 5: Legacy Containerization Strategy

Evaluate moving legacy configs to `__LEGACY__/` folder:

1. **Compatibility Check**:
   - Will Hydra still find configs in `__LEGACY__/`?
   - Do paths need to be updated?
   - Will existing code break?

2. **Migration Plan**:
   - Which configs should move to `__LEGACY__/`?
   - What folder structure should be used?
   - How to maintain backward compatibility?

3. **Documentation**:
   - Create `__LEGACY__/README.md` explaining purpose
   - Document which configs are legacy and why
   - Provide migration guide for users

## Deliverables

### 1. Configuration Inventory

Create a comprehensive inventory with:
- File path
- Architecture type (new/legacy/hybrid)
- Dependencies
- References found
- Override pattern requirements
- Removal impact assessment
- Migration recommendation

### 2. Override Pattern Guide

Document for each config group:
- When to use `+` prefix
- When NOT to use `+` prefix
- Common errors and solutions
- Examples of correct usage

### 3. Legacy Config Report

List all legacy configs with:
- Current location
- Proposed action (remove/move to `__LEGACY__`/keep)
- Migration steps if applicable
- Risk assessment

### 4. Migration Plan

Provide step-by-step plan:
1. Phase 1: Move safe-to-move configs to `__LEGACY__/`
2. Phase 2: Update references or create compatibility layer
3. Phase 3: Remove truly unused configs
4. Phase 4: Update documentation

## Constraints & Considerations

### Do NOT Remove Without:
- ✅ Confirming no code references exist
- ✅ Checking all scripts and runners
- ✅ Verifying UI doesn't use it
- ✅ Documenting what would be lost

### Preserve:
- ✅ Configuration options even if unused (for future reference)
- ✅ Examples of how configs were structured
- ✅ Migration paths for users

### Containerization Benefits:
- ✅ Clear separation of old vs new
- ✅ Maintains compatibility
- ✅ Reduces cognitive load
- ✅ Preserves configuration history

## Analysis Approach

1. **Start with `configs/_base/`** - These are the foundation, understand them first
2. **Map dependencies** - Build dependency graph of all configs
3. **Search codebase** - Find all references systematically
4. **Test overrides** - Use `test_hydra_overrides.py` as reference
5. **Document findings** - Create comprehensive report
6. **Propose actions** - Recommend specific steps

## Expected Output Format

### For Each Configuration File:

```markdown
## configs/path/to/config.yaml

**Architecture**: [New/Legacy/Hybrid]
**Foundation**: [Uses _base/ | Standalone | Mixed]
**Dependencies**: [List]
**Referenced By**: [List files/code]

**Override Pattern**:
- With `+`: `+config=value` ✅/❌
- Without `+`: `config=value` ✅/❌
- Notes: [Any special cases]

**References Found**:
- Code: [List files]
- Scripts: [List files]
- Docs: [List files]
- UI: [List files]

**Impact Assessment**:
- Removal Impact: [Safe/Moderate/High/Critical]
- Lost Options: [List if removed]
- Migration: [Possible/Not Needed/Complex]

**Recommendation**:
- [ ] Remove (no references)
- [ ] Move to `__LEGACY__/` (has references but legacy)
- [ ] Keep (actively used)
- [ ] Migrate to new architecture
```

## Questions to Answer

1. Which configs are truly legacy and unused?
2. Which configs are legacy but still referenced?
3. Which configs need `+` prefix and which don't?
4. Can we safely move legacy configs to `__LEGACY__/`?
5. What configuration options would be lost if we purge?
6. How can we maintain backward compatibility?
7. What's the migration path for each legacy config?

## Success Criteria

✅ All configs classified as new/legacy/hybrid
✅ Override patterns documented for all config groups
✅ All references found and documented
✅ Impact assessment complete for each config
✅ Migration plan created with clear steps
✅ Legacy containerization strategy validated

---

## Usage Instructions

1. **Run this audit** using Claude or Copilot
2. **Provide access** to the codebase for reference searching
3. **Review findings** systematically
4. **Validate recommendations** before implementing
5. **Execute migration** in phases with testing between phases

---

## Additional Context

### Previous Work Completed

**Legacy Cleanup & Config Consolidation (2025-11-11)** - COMPLETED:
- See: `archive/archive_docs/docs/completed_plans/2025-11/2025-11-11_1439_implementation_plan_legacy-cleanup-config-consolidation.md`
- **Phase 3 Findings**: No true duplicate configs found - files with same names serve different purposes in different contexts
- **Example**: `configs/model/optimizer.yaml` vs `configs/model/optimizers/adam.yaml` are similar but serve different purposes (simplified vs complete)
- **Result**: No configs were removed in this phase - all configs serve different purposes
- **Note**: This audit should focus on **architecture separation** (new vs legacy), not duplicate removal

**Related Completed Plans** (for context):
- `archive/archive_docs/docs/completed_plans/2025-11/2025-11-11_1439_implementation_plan_core-training-stabilization.md`
- `archive/archive_docs/docs/completed_plans/2025-11/2025-11-11_1439_implementation_plan_import-time-optimization.md`
- `archive/archive_docs/docs/completed_plans/2025-11/2025-11-12_1439_implementation_plan_revised-inference-consolidation.md`

**Recent Audit Work (2025-12-24)**:
- See: `docs/artifacts/implementation_plans/2025-12-24_1236_implementation_plan_audit-resolution.md`
- Focused on code refactoring and validation consolidation
- Not directly related to Hydra config architecture separation

### Technical References
- Hydra version: Check `pyproject.toml` or `requirements.txt`
- Test suite: `tests/unit/test_hydra_overrides.py` has override pattern tests
- Config Architecture: See `CHANGELOG.md` - Config Architecture Consolidation (Phases 5-8)

### Important Notes

1. **Previous consolidation work exists** - Review what was already done in the 2025-11-11 plan
2. **Archive is temporarily visible** - `/archive/` was removed from `.gitignore` to access old plans
3. **Focus on remaining legacy configs** - Don't re-audit what was already consolidated
4. **Two architectures still exist** - New (`configs/_base/`) and legacy configs need separation

---

**Begin the audit by:**
1. **Review Previous Work**: Read `archive/archive_docs/docs/completed_plans/2025-11/2025-11-11_1439_implementation_plan_legacy-cleanup-config-consolidation.md` to understand:
   - What was already analyzed (Phase 3: Consolidate Config Presets)
   - What was found (no true duplicates, all configs serve different purposes)
   - What remains to be done (architecture separation, not duplicate removal)

2. **Focus on Architecture Separation**: This audit is about **separating new vs legacy architectures**, not finding duplicates:
   - New architecture: Uses `configs/_base/` foundation
   - Legacy architecture: Old system without `_base/` foundation
   - Hybrid: Mixed references

3. **Analyze `configs/_base/`**: Understand the new foundation first, then identify which configs use it vs which don't

4. **Build on Previous Work**: The 2025-11-11 plan already checked for duplicates - focus on architecture classification and override patterns
