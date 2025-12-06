---
title: "2025 12 06 Design Toolkit Deprecation Roadmap"
date: "2025-12-06 18:08 (KST)"
type: "design"
category: "architecture"
status: "active"
version: "1.0"
tags: ['design', 'architecture', 'documentation']
---





# AgentQMS Toolkit Deprecation Roadmap

## Executive Summary

This document outlines the controlled deprecation of `AgentQMS.toolkit` in favor of the new `AgentQMS.agent_tools` canonical surface. The toolkit contains legacy code that has been progressively migrated to agent_tools with improved modularity, better type hints, and clearer canonical boundaries.

**Timeline**: Version 0.3.2 → 0.4.0 (3-4 months)
**Risk Level**: Low (backward compatible deprecation path)
**Impact**: ~46 imports across the codebase

---

## Current State

### Toolkit Structure (Legacy)

```
AgentQMS/toolkit/
├── audit/                    # Audit utilities (deprecated)
├── compliance/               # Compliance tools (deprecated)
├── core/                     # Core tools (partially deprecated)
├── documentation/            # Doc generation (deprecated)
├── maintenance/              # Maintenance utils (deprecated)
├── migration/                # Migration utilities
├── utilities/                # General utilities
└── utils/                    # Path/runtime utilities
```

### Agent Tools Structure (Canonical - v0.3.2+)

```
AgentQMS/agent_tools/
├── audit/                    # ✓ New audit framework
├── compliance/               # ✓ New validation framework
├── core/                     # ✓ New core tools
├── documentation/            # ✓ New doc tools
├── utilities/                # ✓ New utilities
└── utils/                    # ✓ New path/runtime utils
```

### Migration Status

**Completed Migrations** (0/46 imports):
- None yet; deprecation warnings planned in 0.3.2, removal in 0.4.0

**In-Progress** (agent_tools wrapping toolkit):
- `AgentQMS/agent_tools/audit/*` (wraps toolkit versions)
- `AgentQMS/agent_tools/documentation/*` (wraps toolkit versions)
- `AgentQMS/agent_tools/utilities/*` (wraps toolkit versions)

**Pending Migration** (direct toolkit imports):
- test_branch_metadata.py → ArtifactTemplates
- interface/cli_tools/* (multiple runtime utilities)
- toolkit/* (internal cross-dependencies)

---

## Deprecation Timeline

### Phase 1: 0.3.2 (Current + 2 weeks)
**Goal**: Add deprecation warnings, minimal disruption

**Actions**:
- [ ] Add `DeprecationWarning` to `AGENTQMs/toolkit/__init__.py`
- [ ] Log warnings when toolkit modules are imported
- [ ] Emit warnings on module load: `"AgentQMS.toolkit is deprecated as of 0.3.2. Use AgentQMS.agent_tools instead. See docs/artifacts/design/toolkit-deprecation-roadmap.md for migration guide."`
- [ ] Create `.copilot/context/migration-guide.md` with mapping table
- [ ] Update `CHANGELOG.md` with deprecation notice

**Backward Compatibility**: ✅ 100% maintained (warnings only)

**Code Changes**:
1. Update `AgentQMS/toolkit/__init__.py` to emit deprecation warnings
2. Create migration guide in `.copilot/context/`
3. Add migration mapping table (toolkit → agent_tools)

---

### Phase 2: 0.4.0 (4-6 weeks)
**Goal**: Remove toolkit code, complete migration

**Actions**:
- [ ] Remove `AgentQMS/toolkit/` directory entirely
- [ ] Verify all imports updated to `AgentQMS/agent_tools/`
- [ ] Run full test suite: `make validate && make compliance && make boundary`
- [ ] Update documentation to reflect removal

**Backward Compatibility**: ❌ Breaking changes (toolkit removed)

**Code Changes**:
1. Delete `AgentQMS/toolkit/` (backup on release branch)
2. Update all remaining imports in agent_tools wrappers
3. Update Makefile to remove toolkit references
4. Publish migration summary report

---

## Migration Mapping Table

### High Priority (Direct Imports)

| Legacy Import | New Location | Status | Priority |
|---|---|---|---|
| `AgentQMS.toolkit.core.artifact_templates` | `AgentQMS.agent_tools.core.artifact_templates` | Wrapped | 🔴 Critical |
| `AgentQMS.toolkit.utils.runtime` | `AgentQMS.agent_tools.utils.runtime` | Duplicated | 🔴 Critical |
| `AgentQMS.toolkit.compliance.documentation_quality_monitor` | `AgentQMS.agent_tools.compliance.monitor_artifacts` | Merged | 🟡 High |
| `AgentQMS.toolkit.utilities.agent_feedback` | `AgentQMS.agent_tools.utilities.feedback_integration` | New | 🟡 High |
| `AgentQMS.toolkit.documentation.*` | `AgentQMS.agent_tools.documentation.*` | Wrapped | 🟡 High |
| `AgentQMS.toolkit.audit.*` | `AgentQMS.agent_tools.audit.*` | Wrapped | 🟡 High |

### Medium Priority (Internal Cross-Dependencies)

| Legacy Import | New Location | Status | Priority |
|---|---|---|---|
| `AgentQMS.toolkit.utils.config` | `AgentQMS.agent_tools.utils.config` | Not created | 🟠 Medium |
| `AgentQMS.toolkit.utils.migration` | `AgentQMS.agent_tools.utilities.migration_log` | New | 🟠 Medium |
| `AgentQMS.toolkit.utils.paths` | `AgentQMS.agent_tools.utils.paths` | Duplicated | 🟠 Medium |
| `AgentQMS.toolkit.utilities.tracking.db` | `AgentQMS.agent_tools.utilities.tracking.db` | Duplicated | 🟠 Medium |

### Low Priority (Rarely Used)

| Legacy Import | New Location | Status | Priority |
|---|---|---|---|
| `AgentQMS.toolkit.maintenance.*` | Consolidate into agent_tools | Not planned | 🟢 Low |
| `AgentQMS.toolkit.migration.*` | Keep for batch migration only | Legacy | 🟢 Low |

---

## Migration Guide for Users

### Step 1: Identify Toolkit Imports

```bash
# Find all toolkit imports in your code
grep -r "from AgentQMS.toolkit" . --include="*.py"
grep -r "import.*toolkit" . --include="*.py"
```

### Step 2: Update to Agent Tools

**Before**:
```python
from AgentQMS.toolkit.core.artifact_templates import ArtifactTemplates
from AgentQMS.toolkit.utils.runtime import ensure_project_root_on_sys_path
from AgentQMS.toolkit.documentation.auto_generate_index import main
```

**After**:
```python
from AgentQMS.agent_tools.core.artifact_templates import ArtifactTemplates
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path
from AgentQMS.agent_tools.documentation.auto_generate_index import main
```

### Step 3: Test Imports

```bash
# Verify new imports work
python -c "from AgentQMS.agent_tools.core.artifact_templates import ArtifactTemplates; print('✓ Import successful')"
```

### Step 4: Run Validation

```bash
# Ensure no regressions
make validate && make compliance && make boundary
```

---

## Utility Consolidation Plan

### Path Utilities

**Current State**:
- `AgentQMS.toolkit.utils.paths` (16 functions)
- `AgentQMS.agent_tools.utils.paths` (15 functions)

**Action**: Merge and standardize in agent_tools, maintain backward compatibility wrapper in toolkit until 0.4.0

### Runtime Utilities

**Current State**:
- `AgentQMS.toolkit.utils.runtime` (5 functions)
- `AgentQMS.agent_tools.utils.runtime` (5 functions - identical)

**Action**: Remove toolkit version, use agent_tools directly

### Configuration

**Current State**:
- `AgentQMS.toolkit.utils.config` (exists)
- `AgentQMS.agent_tools.utils.config` (does not exist)

**Action**: Migrate config utilities to agent_tools.utils.config

---

## Agent Tools Enhancements (0.3.2+)

### New Modules

- ✅ `AgentQMS.agent_tools.audit.framework_audit` (Phase 2 - created)
- ✅ `AgentQMS.agent_tools.compliance.validate_artifacts` (Phase 1 - enhanced)
- 🟡 `AgentQMS.agent_tools.utils.git` (Phase 1 - created)
- 🟡 `AgentQMS.agent_tools.utils.timestamps` (Phase 1 - created)

### Improved Structure

- **Modularity**: Clear separation of concerns (audit, compliance, core, documentation, utilities)
- **Type Hints**: Full type annotations in agent_tools (partial in toolkit)
- **Testing**: Comprehensive unit tests (toolkit has minimal tests)
- **Documentation**: Auto-generated docstrings and examples

---

## Risk Mitigation

### Data Loss Prevention
- ✅ Toolkit remains functional through 0.4.0
- ✅ All migrations are backward compatible (v0.3.2)
- ✅ No automatic changes to user code

### Testing Strategy
1. Add deprecation warnings in 0.3.2 (users see warnings, code still works)
2. Run full suite: `make validate && make compliance && make boundary`
3. Test all interface commands: `make help`, `make discover`, `make status`
4. Verify Docker deployment with new imports

### Rollback Plan
If issues arise:
1. Revert commit with toolkit removal
2. Restore deprecation warnings only
3. Extend timeline for 0.4.0

---

## Success Criteria

### Phase 1 (v0.3.2)
- [ ] All toolkit modules emit deprecation warnings
- [ ] Migration guide published and comprehensive
- [ ] No toolkit imports in new code
- [ ] Documentation reflects migration path
- [ ] Zero breaking changes (100% backward compatible)

### Phase 2 (v0.4.0)
- [ ] Toolkit directory removed
- [ ] All 46 imports migrated to agent_tools
- [ ] Full validation suite passes: `make validate && make compliance && make boundary`
- [ ] Interface commands work: `make help`, `make discover`, `make audit-framework`
- [ ] No remaining toolkit references in code

---

## Communication Plan

### Announcement (Now - v0.3.2)
- [ ] Update `CHANGELOG.md` with deprecation notice
- [ ] Post announcement in project README
- [ ] Add notice to toolkit `__init__.py` docstring
- [ ] Notify team of migration timeline

### Guidance (Throughout 0.3.2 - 0.4.0)
- [ ] Keep migration guide updated in `.copilot/context/`
- [ ] Answer migration questions in code comments
- [ ] Link to migration guide in deprecation warnings

### Finalization (v0.4.0)
- [ ] Publish migration summary report
- [ ] Document lessons learned
- [ ] Archive old toolkit code in git history

---

## References

- Implementation Plan: `docs/artifacts/implementation_plans/2025-12-06_1200_implementation_plan_agentqms-metadata-branch-versioning.md`
- Canonical Surface: `AgentQMS/knowledge/agent/system.md`
- Agent Tools: `AgentQMS/agent_tools/README.md`
- Toolkit (Legacy): `AgentQMS/toolkit/README.md`

---

## Appendix: Toolkit Import Audit

**Total Imports**: 46 across codebase
**Critical Path**: 8 (test_branch_metadata.py, interface/cli_tools/*)
**Internal**: 38 (toolkit cross-dependencies)

**Breakdown by Source**:
- test_branch_metadata.py: 1
- interface/cli_tools/: 4
- agent_tools wrappers: 12
- toolkit internal: 29

**Estimated Migration Effort**: 3-4 hours total (spread across 0.3.2 → 0.4.0)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-06 14:30 KST
**Next Review**: Post-Phase 3 completion
**Status**: Awaiting Phase 3 implementation
