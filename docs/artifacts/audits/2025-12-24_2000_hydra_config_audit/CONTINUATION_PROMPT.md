# Hydra Config Audit - Continuation Prompt

**Purpose**: This document provides the exact prompt to use when continuing this work in a new session.

**Status**: Phases 1-2 Complete, Phases 3-5 Pending

---

## What to Say to Start Next Session

Copy and paste this prompt to continue the work:

```
Continue the Hydra configuration audit implementation from the previous session.

Context:
- Session handover: docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/SESSION_HANDOVER.md
- Audit assessment: docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_CONFIG_AUDIT_ASSESSMENT.md

Completed phases:
- ✅ Phase 1: Documentation (configs/README.md created, base.yaml updated with override comments)
- ✅ Phase 2: Legacy Containerization (__LEGACY__/ directory created, 2 configs moved)
- ✅ Investigation: Uncertain configs analyzed (logger/default serves different purpose, data/preprocessing unused, train_v2 is experimental)
- ✅ base.yaml updated: Added data to defaults, now data=canonical works without +

Remaining work (Phase 3+):
1. Update code references (ocr/command_builder/compute.py still uses old patterns)
2. Test that moved configs are still accessible
3. Run test suite to validate changes
4. Update test suite if needed for new override patterns
5. Create implementation completion report

Start with Phase 3: Code Reference Updates per the resolution plan in the assessment document.
```

---

## Alternative Prompts for Specific Tasks

### If You Want to Just Test Changes

```
Test the Hydra configuration changes from the previous audit session.

Context:
- Session handover: docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/SESSION_HANDOVER.md

Changes made:
1. Added data to base.yaml defaults
2. Added override pattern comments to base.yaml
3. Created configs/README.md documentation
4. Moved 2 configs to __LEGACY__/ directory

Tasks:
1. Run: uv run python tests/unit/test_hydra_overrides.py
2. Test: data=canonical override works without +
3. Verify: __LEGACY__/ configs still accessible
4. Report: Any failures or issues found
```

### If You Want to Complete Phase 3 Only

```
Complete Phase 3 (Code Reference Updates) of the Hydra config audit.

Context:
- Resolution plan: docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_CONFIG_AUDIT_ASSESSMENT.md (see Phase 3)
- Session handover: docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/SESSION_HANDOVER.md

Phase 3 tasks:
1. Update ocr/command_builder/compute.py to use new override patterns
2. Review UI config loading in apps/ directory
3. Update test suite for new patterns (data in defaults now)
4. Create compatibility layer if needed

Follow the Phase 3 plan from the assessment document.
```

### If You Want to Skip to Documentation Update

```
Update documentation and create completion report for the Hydra config audit.

Context:
- Session handover: docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/SESSION_HANDOVER.md

Completed work:
- Phases 1-2 complete (documentation + containerization)
- base.yaml updated with data in defaults
- 2 configs moved to __LEGACY__/

Tasks:
1. Update CHANGELOG.md with configuration changes
2. Create completion report in docs/artifacts/completed_plans/
3. Document what was changed and why
4. List validation steps performed
```

---

## Documents to Reference

### Primary Documents (Must Read)

1. **[SESSION_HANDOVER.md](SESSION_HANDOVER.md)**
   - Session summary and context
   - What was completed
   - What remains to be done
   - Critical decisions made

2. **[HYDRA_CONFIG_AUDIT_ASSESSMENT.md](HYDRA_CONFIG_AUDIT_ASSESSMENT.md)**
   - Complete audit findings
   - 5-phase resolution plan
   - Configuration inventory
   - Override pattern rules

### Supporting Documents

3. **[HYDRA_OVERRIDE_PATTERNS.md](HYDRA_OVERRIDE_PATTERNS.md)**
   - Quick reference for override patterns
   - Common errors and solutions

4. **[configs/README.md](../../../configs/README.md)**
   - User-facing configuration guide
   - Created in Phase 1

5. **[configs/__LEGACY__/README.md](../../../configs/__LEGACY__/README.md)**
   - Legacy config migration guide
   - Created in Phase 2

### Optional Context

6. **[HYDRA_CONFIG_AUDIT_PROMPT.md](HYDRA_CONFIG_AUDIT_PROMPT.md)**
   - Original audit instructions

7. **[HYDRA_AUDIT_CONTEXT.md](HYDRA_AUDIT_CONTEXT.md)**
   - Previous audit work (2025-11-11)

---

## Quick Status Check

To see what's been done and what's left:

```bash
# Check if documentation exists
ls -lh docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/
ls -lh configs/README.md
ls -lh configs/__LEGACY__/README.md

# Check if configs were moved
ls -lh configs/__LEGACY__/model/
ls -lh configs/__LEGACY__/data/

# Check base.yaml changes
grep "data: default" configs/base.yaml

# Check if tests pass
uv run python tests/unit/test_hydra_overrides.py
```

---

## Expected State After Phases 1-2

### Files Created:
- ✅ `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_CONFIG_AUDIT_ASSESSMENT.md`
- ✅ `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/SESSION_HANDOVER.md`
- ✅ `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/CONTINUATION_PROMPT.md`
- ✅ `configs/README.md`
- ✅ `configs/__LEGACY__/README.md`
- ✅ `configs/__LEGACY__/model/optimizer.yaml` (moved)
- ✅ `configs/__LEGACY__/data/preprocessing.yaml` (moved)

### Files Modified:
- ✅ `configs/base.yaml` (added data to defaults, added override comments)

### Git Status:
```bash
# Should show:
M  configs/base.yaml
A  configs/README.md
A  configs/__LEGACY__/README.md
R  configs/model/optimizer.yaml -> configs/__LEGACY__/model/optimizer.yaml
R  configs/data/preprocessing.yaml -> configs/__LEGACY__/data/preprocessing.yaml
```

---

## Remaining Work (Phases 3-5)

### Phase 3: Code Reference Updates (4-6 hours)
**Status**: Not Started

**Tasks**:
1. Update `ocr/command_builder/compute.py`
   - Change data override patterns
   - Remove legacy references
2. Review `apps/ocr-inference-console/backend/`
   - Check UI config loading
3. Update test suite
   - `tests/unit/test_hydra_overrides.py`
   - Add tests for new patterns
4. Create compatibility layer (if needed)

**Deliverables**:
- Updated code files
- Updated tests
- Validation report

---

### Phase 4: Deprecation Warnings (2 hours)
**Status**: Not Started

**Tasks**:
1. Add deprecation detection function
2. Update runners (train.py, test.py, predict.py)
3. Document in CHANGELOG.md

**Deliverables**:
- Deprecation warning system
- Updated CHANGELOG
- User notification

---

### Phase 5: Archive and Remove (Future)
**Status**: Not Planned

**Recommendation**: Skip this phase for now. Keep legacy configs accessible.

---

## Validation Checklist

Before marking work complete, verify:

- [ ] All tests pass: `uv run python tests/unit/test_hydra_overrides.py`
- [ ] data=canonical works without +: `uv run python runners/train.py data=canonical --cfg job`
- [ ] data=craft works without +: `uv run python runners/train.py data=craft --cfg job`
- [ ] Legacy configs accessible: `ls configs/__LEGACY__/model/optimizer.yaml`
- [ ] Documentation complete: `cat configs/README.md`
- [ ] base.yaml has data in defaults: `grep "data: default" configs/base.yaml`
- [ ] Override comments added: `grep "OVERRIDE PATTERN GUIDE" configs/base.yaml`

---

## Success Criteria

### Phase 1-2 Success (Current):
- [x] configs/README.md created with comprehensive guide
- [x] base.yaml updated with override comments
- [x] base.yaml updated with data in defaults
- [x] __LEGACY__/ directory created
- [x] 2 legacy configs moved successfully
- [x] __LEGACY__/README.md created with migration guide
- [ ] Tests still pass (needs validation)
- [ ] Configs still accessible (needs validation)

### Phase 3+ Success (Future):
- [ ] Code updated to use new patterns
- [ ] All tests pass
- [ ] No regressions in functionality
- [ ] Deprecation warnings in place
- [ ] Completion report created

---

## Common Issues and Solutions

### Issue: Tests fail after changes

**Solution**:
```bash
# Check what failed
uv run python tests/unit/test_hydra_overrides.py -v

# If data override tests fail, update test to use data=canonical (no +)
# If legacy config tests fail, update test to use __LEGACY__/ prefix
```

### Issue: Config not found

**Solution**:
```bash
# Check config exists
ls configs/data/canonical.yaml

# Check Hydra can find it
uv run python runners/train.py data=canonical --cfg job

# If moved to __LEGACY__, update path
uv run python runners/train.py +data=__LEGACY__/preprocessing --cfg job
```

### Issue: Git conflicts

**Solution**:
```bash
# Check what changed
git status
git diff configs/base.yaml

# If conflicts, resolve manually
# Ensure data: default is in defaults list
# Ensure override comments are present
```

---

## Key Decisions Summary

**Decision 1**: Add data to base.yaml defaults
- **Made**: ✅ Yes
- **Result**: data=canonical, data=craft now work without +
- **Impact**: Consistent override pattern for all primary config groups

**Decision 2**: Containerize legacy configs
- **Made**: ✅ Yes
- **Result**: Moved to __LEGACY__/ directory
- **Impact**: Clear separation, configs still accessible

**Decision 3**: Keep presets vs architectures distinct
- **Made**: ✅ Yes (keep distinct)
- **Result**: Both serve different purposes, documented in README
- **Impact**: None - maintain current structure

---

## Timeline Estimate

**Phase 1-2** (Complete): ~3 hours
- Documentation: 1 hour
- Containerization: 2 hours

**Phase 3** (Pending): 4-6 hours
- Code updates: 3-4 hours
- Testing: 1-2 hours

**Phase 4** (Pending): 2 hours
- Deprecation warnings: 1 hour
- Documentation: 1 hour

**Total Remaining**: 6-8 hours

---

## Contact Information

**Session**: 2025-12-24
**Agent**: Claude Sonnet 4.5
**Token Usage**: ~118k / 200k (59% utilized at handover)

**For Questions**:
1. Review SESSION_HANDOVER.md
2. Review HYDRA_CONFIG_AUDIT_ASSESSMENT.md
3. Check configs/README.md for usage patterns

---

## Quick Start Next Session

1. **Read**: [SESSION_HANDOVER.md](SESSION_HANDOVER.md)
2. **Check Status**: Run validation checklist above
3. **Choose Phase**: Decide whether to proceed with Phase 3 or test Phase 1-2 changes first
4. **Use Prompt**: Copy the appropriate prompt from this document
5. **Execute**: Follow the resolution plan in HYDRA_CONFIG_AUDIT_ASSESSMENT.md

---

**Ready to Continue**: ✅ Yes
**Documentation**: ✅ Complete
**Next Action**: Choose a continuation prompt above and start new session
