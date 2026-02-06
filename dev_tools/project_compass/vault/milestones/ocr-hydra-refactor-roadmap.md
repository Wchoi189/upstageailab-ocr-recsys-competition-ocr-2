# OCR Hydra Refactor Roadmap

**Milestone ID**: `ocr-hydra-refactor`
**Status**: `active`
**Created**: 2026-01-25
**Target Completion**: 2026-02-22
**Related Debug Session**: [__DEBUG__/2026-01-22_hydra_configs_legacy_imports](__DEBUG__/2026-01-22_hydra_configs_legacy_imports)

---

## Big Picture

Resolve legacy import and Hydra config issues to enable stable training pipelines. This roadmap addresses the systematic breakage identified in the 2026-01-22 debug session:
- **53 broken Python imports** → 0
- **13 broken Hydra targets** → 0
- **2 non-functional pipelines** → 2 validated pipelines

---

## Phases

### Phase 1: Audit & Validation (Week 1)
**Target**: Environment validated, baseline metrics established

**Tasks**:
- [x] Run master audit and identify all broken imports/targets
- [x] Create migration guard script
- [x] Document ghost code phenomenon
- [ ] Establish editable install as permanent requirement
- [ ] Create pre-flight check in CI/CD
- [ ] Validate both pipelines can instantiate configs

**Success Criteria**:
- ✅ Environment validation script exists
- ✅ Master audit runs successfully
- ⏳ Editable install enforced project-wide
- ⏳ Baseline metrics documented

**Metrics**:
| Metric                 | Initial | Current | Target |
| ---------------------- | ------- | ------- | ------ |
| Broken Python imports  | 53      | 49      | 0      |
| Broken Hydra targets   | 18      | 13      | 0      |
| Detection pipeline     | ❌       | ⏳       | ✅      |
| Recognition pipeline   | ❌       | ⏳       | ✅      |

---

### Phase 2: Import Resolution (Week 2)
**Target**: 0 broken imports

**Tasks**:
- [ ] Batch-fix ocr.core.* → ocr.domains.* imports
- [ ] Fix cross-domain import leakage
- [ ] Update all __init__.py files for lazy loading
- [ ] Migrate registry patterns to Hydra _target_
- [ ] Run import validation suite
- [ ] Test import resolution doesn't break pipelines

**Success Criteria**:
- ✅ All 49 remaining imports fixed
- ✅ No new import errors introduced
- ✅ Both pipelines can import dependencies

**Tools**:
- `scripts/audit/master_audit.py` - Systematic scanning
- `adt` (agent-debug-toolkit) - AST analysis
- grep/sed for batch replacements

---

### Phase 3: Hydra Alignment (Week 3)
**Target**: 0 broken targets

**Tasks**:
- [ ] Run auto-alignment script on all configs
- [ ] Fix recursive instantiation traps (_recursive_=True issues)
- [ ] Validate all Hydra interpolations resolve
- [ ] Test factory pattern compatibility
- [ ] Document Hydra best practices
- [ ] Create config validation pre-commit hook

**Success Criteria**:
- ✅ All 13 broken Hydra targets fixed
- ✅ No recursive instantiation errors
- ✅ Config validation automated

**Tools**:
- `scripts/audit/auto_align_hydra.py` - Auto-healing
- `yq` - YAML analysis (see artifacts/tool_guides/yq_mastery_guide.md)
- Hydra debug mode

**Critical Patterns**:
```yaml
# BAD: Premature instantiation
model:
  _target_: ...
  optimizer:
    _target_: torch.optim.Adam
    _recursive_: True  # ❌ Instantiates before model exists

# GOOD: Factory pattern
model:
  _target_: ...
  optimizer:
    _target_: ocr.core.factories.create_optimizer
    params: ${model.parameters}  # ✅ Lazy evaluation
```

---

### Phase 4: Testing & Integration (Week 4)
**Target**: Both pipelines validated, CI/CD integrated

**Tasks**:
- [ ] Run detection pipeline (det_resnet50_v1) full training cycle
- [ ] Run recognition pipeline (rec_baseline_v1) full training cycle
- [ ] Validate vocab injection mechanism
- [ ] Add regression tests for import/Hydra errors
- [ ] Integrate pre-flight checks into CI/CD
- [ ] Document pipeline execution patterns
- [ ] Create rollback plan

**Success Criteria**:
- ✅ Detection pipeline runs end-to-end
- ✅ Recognition pipeline runs end-to-end
- ✅ No import or Hydra errors in CI/CD
- ✅ Compass tracking shows 100% completion

**Test Commands**:
```bash
# Detection (Priority 1)
uv run python runners/train.py experiment=det_resnet50_v1 +trainer.fast_dev_run=True

# Recognition (Priority 2)
uv run python runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True

# Full validation
uv run python scripts/audit/master_audit.py
bash scripts/preflight.sh
```

---

## Success Criteria (Project-Wide)

- **Zero Broken Imports**: All Python imports resolve correctly
- **Zero Broken Targets**: All Hydra `_target_` paths are valid
- **Pipeline Stability**: Both detection and recognition pipelines run successfully
- **Environment Validation**: Editable install enforced and verified
- **Compass Tracking**: 100% task completion in Project Compass
- **Documentation**: All patterns documented in AgentQMS artifacts

---

## Links and Resources

### Debug Session
- [Current Debug Session](__DEBUG__/2026-01-22_hydra_configs_legacy_imports/README.md)
- [Initial Analysis](__DEBUG__/2026-01-22_hydra_configs_legacy_imports/01_initial_analysis.md)
- [Investigation](__DEBUG__/2026-01-22_hydra_configs_legacy_imports/02_investigation.md)
- [Findings](__DEBUG__/2026-01-22_hydra_configs_legacy_imports/03_findings.md)

### AI-Optimized Artifacts
- [Artifact Index](__DEBUG__/2026-01-22_hydra_configs_legacy_imports/artifacts/README.md)
- [Quick Start](__DEBUG__/2026-01-22_hydra_configs_legacy_imports/artifacts/QUICK_START.md)
- [YQ Mastery](__DEBUG__/2026-01-22_hydra_configs_legacy_imports/artifacts/tool_guides/yq_mastery_guide.md)
- [ADT Usage Patterns](__DEBUG__/2026-01-22_hydra_configs_legacy_imports/artifacts/tool_guides/adt_usage_patterns.md)
- [Shim Antipatterns](__DEBUG__/2026-01-22_hydra_configs_legacy_imports/artifacts/refactoring_patterns/shim_antipatterns_guide.md)

### Standards & Tools
- [AgentQMS Standards](AgentQMS/standards/registry.yaml)
- [OCR Domain Refactor Milestone](project_compass/vault/milestones/ocr-domain-refactor.md)
- [Master Audit Script](scripts/audit/master_audit.py)
- [Auto-Alignment Script](scripts/audit/auto_align_hydra.py)

---

## Project Compass Integration

### Initialize Pulse
```bash
uv run compass pulse-init \
  --id "hydra-refactor-2026-01-22" \
  --obj "Resolve all Hydra and import issues toward zero errors" \
  --milestone "ocr-domain-refactor" \
  --phase "integration"
```

### Update Status
```bash
# Check current pulse status
uv run compass pulse-status

# Sync progress from debug session
uv run compass pulse-sync \
  --path __DEBUG__/2026-01-22_hydra_configs_legacy_imports/README.md \
  --type walkthrough

# Update token burden if working on complex phase
uv run compass pulse-checkpoint --token-burden high
```

### Export Completed Work
```bash
# After completing a phase
uv run compass pulse-export --force
```

---

## Verification Commands

Run these after each phase to verify progress:

```bash
# 1. Environment check
uv run python -c "import ocr; assert 'site-packages' not in ocr.__file__, 'Not editable install!'"

# 2. Import validation
uv run python scripts/audit/master_audit.py | grep "Broken imports:"

# 3. Hydra validation
uv run python scripts/audit/master_audit.py | grep "Broken targets:"

# 4. Pipeline smoke test (detection)
uv run python runners/train.py experiment=det_resnet50_v1 +trainer.fast_dev_run=True

# 5. Pipeline smoke test (recognition)
uv run python runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True

# 6. Full compliance check
uv run aqms validate --all
```

---

## Constraints

1. **Do Not Modify Artifacts Outside Pulse Staging**: All new artifacts must be created via `uv run compass pulse-sync` after initialization.
2. **Use AgentQMS Standards**: All validation must follow [AgentQMS standards](AgentQMS/standards/registry.yaml).
3. **Preserve Debug Session**: Keep `__DEBUG__/2026-01-22_hydra_configs_legacy_imports/` as raw archive; do not restructure.
4. **Incremental Validation**: Run validation after each import/config fix to avoid cascading errors.
5. **Document Patterns**: Add reusable patterns to AgentQMS artifacts for future reference.

---

## Risk Mitigation

| Risk                              | Impact | Mitigation                                   |
| --------------------------------- | ------ | -------------------------------------------- |
| Fixing imports breaks pipelines   | High   | Test pipelines after each batch of fixes     |
| Hydra recursion traps reappear    | Medium | Document patterns, add config validation     |
| Environment reverts to standard   | High   | Add CI/CD check, migration guard in setup.py |
| New imports introduce leakage     | Medium | Pre-commit hook for cross-domain imports     |
| Scope creep (refactoring too much)| Medium | Stay focused on import/Hydra fixes only      |

---

## Notes for AI Agents

- **Focus**: Track metrics in Phase 1 table. Update after each batch fix.
- **Tools**: Prefer automated scripts (master_audit.py, auto_align_hydra.py) over manual edits.
- **Verification**: Run verification commands before marking phase complete.
- **Documentation**: Link to debug session artifacts instead of duplicating content.
- **Compass Sync**: Use `pulse-sync` only after creating pulse; avoid manual artifact creation.

---

## Changelog

| Date       | Event                                      | Status     |
| ---------- | ------------------------------------------ | ---------- |
| 2026-01-22 | Debug session identified critical issues   | Discovered |
| 2026-01-25 | Roadmap created, pulse initialized         | Planning   |
| TBD        | Phase 1 complete (validation)              | Pending    |
| TBD        | Phase 2 complete (imports fixed)           | Pending    |
| TBD        | Phase 3 complete (Hydra aligned)           | Pending    |
| TBD        | Phase 4 complete (pipelines validated)     | Pending    |
