# ⚠️ DEPRECATION COMPLETE - NUCLEAR REFACTOR EXECUTED

**Status:** ✅ COMPLETED (Phase 3 - Nuclear Refactor)
**Completion Date:** 2026-01-20
**Outcome:** Legacy system completely removed

## What Happened

**Nuclear refactor executed instead of gradual 3-phase deprecation.**

### Actions Taken:
- ✅ **All legacy tool scripts DELETED** (artifact_workflow.py, validate_artifacts.py, etc.)
- ✅ **Archived discovery files DELETED** (AgentQMS/standards/.archive/)
- ✅ **settings.yaml cleaned** - Only qms CLI remains in tool_mappings
- ✅ **Documentation rewritten** - Single source of truth (AgentQMS/README.md)
- ✅ **AGENTS.yaml updated** - All legacy references removed
- ✅ **qms CLI made globally accessible** - Symlink to /usr/local/bin/qms

### Breaking Changes:
- ⚠️ No backward compatibility for legacy tool calls
- ⚠️ Old Makefile commands removed/simplified
- ⚠️ Legacy imports still work via compatibility shims (OCR domain refactor)

### Why Nuclear Instead of Gradual:
- Hybrid state caused confusion and split-brain syndrome
- Legacy system continued getting updates
- Risk of divergence too high
- Clean break better than gradual deprecation

---

## Current State (Post-Nuclear)

**ONLY supported interface:** `qms` CLI

```bash
# All operations use qms
qms --help
qms validate --all
qms artifact create --type implementation_plan --name my-feature
qms generate-config --path ocr/inference
```

**Documentation:** See `AgentQMS/README.md`

---

## For Historical Reference: Original Deprecation Plan

The content below is the original gradual deprecation plan that was **NOT** executed.
Instead, a nuclear refactor was performed.

---

# AgentQMS Legacy Tool Deprecation Plan (ORIGINAL - NOT EXECUTED)

**Original Version:** 0.3.0
**Original Date:** 2026-01-20
**Original Status:** Phase 1 (Soft Deprecation) - Planned (but not executed)

---

## Overview (ORIGINAL PLAN)

This document outlined the deprecation plan for legacy AgentQMS tools and discovery files that have been replaced by the unified registry and QMS CLI.

**Original Goal:** Gracefully transition from 5 separate tools + 2 discovery files to 1 unified CLI + 1 registry while maintaining backward compatibility.

**Actual Outcome:** Nuclear refactor executed - all legacy removed immediately.

---

## Deprecated Components

### 1. Discovery Files

| Component | Status | Replacement | Location |
|-----------|--------|-------------|----------|
| `INDEX.yaml` | ⚠️ Deprecated | `registry.yaml` | `AgentQMS/standards/.archive/INDEX.yaml` |
| `standards-router.yaml` | ⚠️ Deprecated | `registry.yaml` | `AgentQMS/standards/.archive/standards-router.yaml` |

**Reason:** Logic fragmentation between glob-based and keyword-based discovery.

### 2. Tool Scripts

| Component | Status | Replacement | Location |
|-----------|--------|-------------|----------|
| `artifact_workflow.py` | ⚠️ Deprecated | `qms artifact` | `AgentQMS/tools/core/artifact_workflow.py` |
| `validate_artifacts.py` | ⚠️ Deprecated | `qms validate` | `AgentQMS/tools/compliance/validate_artifacts.py` |
| `monitor_artifacts.py` | ⚠️ Deprecated | `qms monitor` | `AgentQMS/tools/compliance/monitor_artifacts.py` |
| `agent_feedback.py` | ⚠️ Deprecated | `qms feedback` | `AgentQMS/tools/utilities/agent_feedback.py` |
| `documentation_quality_monitor.py` | ⚠️ Deprecated | `qms quality` | `AgentQMS/tools/compliance/documentation_quality_monitor.py` |

**Reason:** Tool fragmentation increasing AI decision space and token usage.

---

## Deprecation Phases

### Phase 1: Soft Deprecation (Current - 2026 Q1-Q2)

**Timeline:** 3-6 months
**Status:** ✅ IN PROGRESS

**Actions Completed:**
- [x] Create unified `qms` CLI with all subcommands
- [x] Archive old discovery files to `.archive/`
- [x] Mark legacy tools as deprecated in `settings.yaml`
- [x] Add deprecation notices to all legacy tools
- [x] Create migration guide
- [x] Update `AGENTS.yaml` with new commands
- [x] Ensure backward compatibility via tool mappings

**Actions Remaining:**
- [ ] Monitor adoption in AI agent usage
- [ ] Collect feedback on new CLI
- [ ] Update all internal documentation
- [ ] Add migration reminders to legacy tool output

**Success Criteria:**
- All new AI prompts reference `qms` CLI
- 80%+ of agent interactions use new CLI
- Zero migration issues reported

---

### Phase 2: Hard Deprecation (2026 Q3-Q4)

**Timeline:** 6-12 months after Phase 1
**Status:** ⏳ PLANNED

**Actions:**
- [ ] Remove legacy tools from primary tool mappings
- [ ] Move legacy tools to `AgentQMS/tools/.deprecated/`
- [ ] Add loud deprecation warnings when legacy tools are used
- [ ] Update CI to fail if legacy tools are called
- [ ] Remove `.archive/` files from active documentation

**Success Criteria:**
- 95%+ adoption of new CLI
- No active use of legacy tools in CI/CD
- All documentation updated

---

### Phase 3: Complete Removal (2027 Q1+)

**Timeline:** 12+ months after Phase 2
**Status:** ⏳ FUTURE

**Actions:**
- [ ] Delete legacy tool files
- [ ] Delete archived discovery files
- [ ] Remove all backward compatibility shims
- [ ] Clean up `settings.yaml` legacy references
- [ ] Archive this deprecation plan

**Success Criteria:**
- 100% migration complete
- Clean codebase with no legacy references
- All tests passing without legacy tools

---

## Migration Assistance

### For AI Agents

**Old Tool Usage Pattern:**
```python
# Directly calling Python scripts
python AgentQMS/tools/compliance/validate_artifacts.py --all
python AgentQMS/tools/core/artifact_workflow.py create --type plan
```

**New Recommended Pattern:**
```bash
# Using unified CLI
qms validate --all
qms artifact create --type implementation_plan --name my-feature
```

**Helper:** Run `qms --help` for command reference

---

### For Developers

**No immediate action required.** All imports and tool calls continue to work via:

1. **Compatibility Shims** (OCR domain refactor)
   ```python
   # Old import (still works)
   from ocr.data.datasets.db_collate_fn import DBCollateFN

   # Points to: ocr.domains.detection.data.collate_db.DBCollateFN
   ```

2. **Tool Mapping Fallbacks**
   ```yaml
   # settings.yaml maintains both old and new references
   tool_mappings:
     qms: (new unified CLI)
     artifact_workflow: (marked deprecated, still functional)
   ```

---

## Monitoring & Metrics

### Usage Tracking

Monitor legacy tool usage via:

```bash
# Check if legacy tools are being called
grep -r "artifact_workflow.py\|validate_artifacts.py\|monitor_artifacts.py" .github/workflows/
grep -r "INDEX.yaml\|standards-router.yaml" AgentQMS/

# Analyze token savings
python AgentQMS/bin/monitor-token-usage.py --detailed
```

### Key Metrics

Track the following metrics during Phase 1:

| Metric | Target | Current |
|--------|--------|---------|
| QMS CLI adoption rate | 80% | TBD |
| Legacy tool calls in CI | 0% | 0% ✅ |
| Migration issues reported | 0 | 0 ✅ |
| Token usage reduction | 85%+ | 85.6% ✅ |

---

## Rollback Plan

If critical issues arise during Phase 1:

### Immediate Rollback
```bash
# Restore old discovery files from archive
cp AgentQMS/standards/.archive/INDEX.yaml AgentQMS/standards/
cp AgentQMS/standards/.archive/standards-router.yaml AgentQMS/standards/

# Update AGENTS.yaml to reference old files
# Revert settings.yaml changes
```

### Partial Rollback
- Keep `registry.yaml` but temporarily re-enable archived files
- Keep `qms` CLI but allow legacy tools without warnings
- Investigate and fix issues before proceeding

**Note:** Rollback should only be used for critical failures. Minor issues should be fixed forward.

---

## Communication Plan

### Phase 1 Announcements

**Week 1 (Current):**
- [x] Update `AGENTS.yaml` with new commands
- [x] Create migration guide
- [x] Update internal documentation

**Week 2-4:**
- [ ] Add deprecation notices to legacy tool output
- [ ] Send announcement to team about new CLI
- [ ] Update project README

**Month 2-3:**
- [ ] Collect feedback from AI agents and developers
- [ ] Address migration issues
- [ ] Refine CLI based on usage patterns

**Month 4-6:**
- [ ] Prepare for Phase 2 transition
- [ ] Final reminder about upcoming hard deprecation
- [ ] Ensure 80%+ adoption before proceeding

---

## Risk Assessment

### Low Risk ✅
- **Backward Compatibility:** All legacy tools continue to work
- **Testing:** Comprehensive compatibility shims tested
- **Documentation:** Migration guide available

### Medium Risk ⚠️
- **AI Agent Adaptation:** Agents may continue using old patterns if not updated
- **Discovery Confusion:** Agents might reference archived files

**Mitigation:**
- Clear deprecation warnings in legacy tools
- Updated `AGENTS.yaml` with preferred commands
- Migration reminders in documentation

### High Risk ❌
- None identified. All changes are backward compatible.

---

## Success Indicators

### Phase 1 Success
- ✅ Zero breaking changes
- ✅ CI workflows passing with new system
- ✅ Token usage reduced by 85%+
- ⏳ 80%+ adoption of new CLI
- ⏳ Positive feedback from users

### Phase 2 Readiness
- 95%+ adoption of `qms` CLI
- No active use of legacy tools in production
- All documentation updated
- Zero migration blockers reported

### Phase 3 Readiness
- 100% migration complete
- 3+ months of stable usage
- No rollback requests
- Clean architecture validation

---

## Appendix

### A. Deprecation Notice Template

Add to legacy tools:

```python
import warnings

warnings.warn(
    "This tool is deprecated and will be removed in AgentQMS v0.4.0. "
    "Please use 'qms <subcommand>' instead. See AgentQMS/MIGRATION_GUIDE.md",
    DeprecationWarning,
    stacklevel=2
)
```

### B. Alternative Command Reference

| Legacy | New | Notes |
|--------|-----|-------|
| `make validate` | `qms validate --all` | Direct CLI preferred |
| `make compliance` | `qms monitor --check` | More explicit |
| `make create-plan` | `qms artifact create --type implementation_plan` | Clearer intent |
| Manual standard loading | `qms generate-config --path <path>` | New feature |

### C. Contact & Support

For deprecation-related questions:
- **Migration Issues:** `qms feedback report --issue-type "migration" --description "..."`
- **Documentation:** See `AgentQMS/MIGRATION_GUIDE.md`
- **Monitoring:** Run `python AgentQMS/bin/monitor-token-usage.py`

---

**Document Version:** 1.0
**Last Updated:** 2026-01-20
**Next Review:** Phase 1 completion (2026 Q2)
