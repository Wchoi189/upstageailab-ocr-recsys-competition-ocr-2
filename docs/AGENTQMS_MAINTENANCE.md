# AgentQMS Maintenance & Audit Instructions

**Last Updated**: 2026-01-10 (KST)  
**Status**: Operational  
**Health Score**: 7/10 (Architecture stable, technical debt documented)

---

## Quick Start for Auditors

### System Health Check (5 minutes)

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# 1. Verify plugins load correctly
python -m AgentQMS.tools.core.plugins --list 2>&1 | head -20

# 2. Run validation tests
python -m pytest AgentQMS/tests/ -v --tb=short 2>&1 | tail -30

# 3. Check artifact validation
cd AgentQMS/bin && make plugins-snapshot

# 4. Verify no import errors
python -c "from AgentQMS.tools.utilities.grok_linter import *" 2>&1
python -c "from AgentQMS.tools.utilities.grok_fixer import *" 2>&1
```

**Expected Output**:
- ✅ 9 artifact types discovered
- ✅ 12 context bundles discovered
- ✅ 18 tests passing
- ✅ No import errors

---

## Architecture Overview

### 1. Active Components (82% - 52 files, 14,600 LOC)

#### **Core Plugin System** (8 files)
- `tools/core/plugins/discovery.py` - Plugin discovery mechanism
- `tools/core/plugins/loader.py` - Plugin loading & registration
- `tools/core/plugins/validation.py` - Schema validation
- `tools/core/plugins/registry.py` - Plugin registry management
- `tools/core/plugins/__main__.py` - CLI interface

**Key Responsibility**: Define and load artifact types, context bundles, validators

#### **Artifact System** (5 files)
- `tools/core/artifact_workflow.py` - Artifact creation & lifecycle
- `tools/core/artifact_templates.py` - Template management
- `tools/core/discover.py` - Discovery utilities
- `tools/core/tool_registry.py` - Tool registration

**Key Responsibility**: Create, validate, track artifacts

#### **Compliance Layer** (4 files - 1,800 LOC)
- `tools/compliance/validate_artifacts.py` - Primary validator (1,123 LOC)
- `tools/compliance/monitor_artifacts.py` - Monitoring
- `tools/compliance/validate_boundaries.py` - Boundary rules
- `tools/compliance/documentation_quality_monitor.py` - QA

**Key Responsibility**: Ensure artifact quality, compliance, consistency

#### **Utilities** (17 files)
- CLI support tools serving Makefile targets
- Context system, tracking, smart population, etc.

#### **Infrastructure** (6 files)
- Path resolution, runtime, git integration, timestamps, config management
- **Key**: All timestamps standardized to KST (Asia/Seoul)

---

### 2. Legacy Components (12% - 18 files, 2,100 LOC)

#### **Plugin Legacy** (2 files - 295 LOC) - SHOULD REMOVE
```
tools/archive/plugins_legacy.py         (90 LOC)
tools/archive/plugin_loader_shim.py     (205 LOC)
```
**Status**: Superseded by `tools/core/plugins/`  
**Action**: DELETE - no active references

#### **Audit Framework** (6 files - 1,829 LOC) - SHOULD REMOVE
```
tools/archive/audit/audit_validator.py
tools/archive/audit/audit_generator.py
tools/archive/audit/framework_audit.py
tools/archive/audit/analyze_graph.py
tools/archive/audit/checklist_tool.py
tools/archive/audit/artifact_audit.py
```
**Status**: Legacy, superseded by `tools/compliance/`  
**Action**: MOVE to `archive/deprecated_audit/` with documentation

#### **OCR-Specific** (7 files - 3,500+ LOC) - SHOULD ISOLATE
```
tools/archive/ocr/*
```
**Status**: Domain-specific, not framework tools  
**Action**: Move to `archive/domain_specific/ocr/` with README

---

## Maintenance Procedures

### Audit Checklist

**Weekly Check** (15 minutes):
- [ ] Tests pass: `pytest AgentQMS/tests/ -v`
- [ ] Plugins load: `python -m AgentQMS.tools.core.plugins --list`
- [ ] No import errors: Check imports of changed utilities

**Monthly Audit** (1 hour):
- [ ] Check for new test files outside `/tests/`: `find AgentQMS -name "test_*.py" | grep -v /tests/`
- [ ] Check for deprecated imports: `grep -r "plugins_legacy\|archive.audit" AgentQMS/tools/ --include="*.py" | grep -v "^AgentQMS/tools/archive"`
- [ ] Review tool usage: `grep -r "grok_linter\|grok_fixer" . --include="*.py" | grep -v test`

**Quarterly Review** (4 hours):
- [ ] Run full validation: `cd AgentQMS/bin && make validate && make compliance`
- [ ] Review new utilities: Check tools/utilities/ for single-use files
- [ ] Update technical debt log: Add new legacy discoveries
- [ ] Review artifact naming: Check docs/artifacts/ for convention violations

---

## Common Issues & Solutions

### Issue 1: Plugin Discovery Failing

**Symptoms**:
```
python -m AgentQMS.tools.core.plugins --list
# Returns: No artifact types discovered
```

**Diagnosis**:
```bash
python -m AgentQMS.tools.core.plugins --validate
# Should show validation errors
```

**Fix**:
- Check plugin YAML files in `.agentqms/plugins/artifact_types/`
- Verify `ads_version` field is present (required as of 2026-01-10)
- Run: `python -m AgentQMS.tools.core.plugins --write-snapshot` to rebuild registry

**Prevention**: Keep `.agentqms/schemas/plugin_*.json` files up-to-date when changing plugin format

---

### Issue 2: Validation Failures

**Symptoms**:
```
cd AgentQMS/bin && make validate
# Shows: FIX SUGGESTIONS for artifact naming
```

**Common Issues**:
1. **Artifact naming** - Must follow: `YYYY-MM-DD_HHMM_{type}_{name}.md`
   - Fix: Rename file to conform to pattern
   
2. **Missing frontmatter** - Required fields missing
   - Fix: Add mandatory fields: type, category, status, version
   - Use `make autofix` to auto-repair (if available)

3. **Invalid artifact type** - Type not in plugin registry
   - Fix: Ensure plugin exists in `.agentqms/plugins/artifact_types/`
   - Must be one of: assessment, audit, bug_report, change_request, design_document, implementation_plan, ocr_experiment, vlm_report, walkthrough

---

### Issue 3: Import Errors

**Symptoms**:
```python
from AgentQMS.tools.utilities.grok_linter import *
# ModuleNotFoundError or ImportError
```

**Diagnosis**:
- Check that all dependencies are installed: `pip list | grep -E "jsonschema|pyyaml|requests"`
- Verify Python path: `echo $PYTHONPATH` (should include project root)

**Fix**:
- For Python dependency issues: `pip install -r requirements.txt`
- For path issues: Run from project root: `cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2`

---

## Phase 2 Cleanup Plan

### Timeline: Week 1-2
**Estimated Effort**: 10-14 hours

#### Phase 2a: Archive Cleanup (4-5 hours)
1. Create directory structure:
   ```bash
   mkdir -p AgentQMS/tools/archive/deprecated_audit
   mkdir -p AgentQMS/tools/archive/domain_specific/ocr
   ```

2. Move deprecated audit files:
   ```bash
   mv AgentQMS/tools/archive/audit/* AgentQMS/tools/archive/deprecated_audit/
   ```

3. Move OCR tools:
   ```bash
   mv AgentQMS/tools/archive/ocr/* AgentQMS/tools/archive/domain_specific/ocr/
   ```

4. Delete legacy plugin system:
   ```bash
   rm AgentQMS/tools/archive/plugins_legacy.py
   rm AgentQMS/tools/archive/plugin_loader_shim.py
   ```

5. Create documentation:
   - `AgentQMS/tools/archive/deprecated_audit/README.md`
   - `AgentQMS/tools/archive/domain_specific/README.md`

6. Test: `pytest AgentQMS/tests/ -v` (should still pass)

#### Phase 2b: Code Consolidation (4-6 hours)
1. Move test file from production:
   ```bash
   mkdir -p AgentQMS/tests/compliance
   mv AgentQMS/tools/compliance/test_*.py AgentQMS/tests/compliance/
   ```

2. Review context system:
   - Document relationship between `context_control.py` and `context_inspector.py`
   - Consolidate if appropriate
   - Update docstrings

3. Review tracking subsystem:
   - Clarify `tracking_integration.py` vs `tools/utilities/tracking/` directory
   - Ensure no duplication

4. Audit utilities:
   - Review each of 17 utility modules
   - Remove dead code
   - Add comprehensive docstrings

#### Phase 2c: Documentation (1-2 hours)
1. Update CHANGELOG.md with all changes
2. Update AgentQMS/tools/archive/README.md
3. Create migration guide if any external tools depend on archived code
4. Update architecture documentation

---

## Versioning & Standards

### Plugin Versioning
**Standard**: Use `ads_version` field (as of 2026-01-10)

```yaml
name: my_artifact_type
ads_version: "1.0"  # ✅ CORRECT
version: "1.0"      # ⚠️ Still supported for backward compatibility
```

**Schema Files**:
- `AgentQMS/standards/schemas/plugin_artifact_type.json` - Required: `ads_version`
- `AgentQMS/standards/schemas/plugin_context_bundle.json` - Optional: `ads_version`
- `AgentQMS/standards/schemas/plugin_validators.json` - Optional: `ads_version`

### Timestamp Standard
**Format**: ISO-8601 with KST timezone
```python
# ✅ CORRECT
"2026-01-10T17:26:49+0900"

# ❌ INCORRECT
"2026-01-10T17:26:49Z"      # UTC
"2026-01-10 17:26:49"       # No timezone
```

**Implementation**:
```bash
# In Makefile/scripts
LAST_UPDATED := $(shell TZ=Asia/Seoul date +%Y-%m-%dT%H:%M:%S%z)
```

---

## Audit Tools & Commands

### Quick Diagnostics
```bash
# List all plugins
python -m AgentQMS.tools.core.plugins --list

# Show specific plugin
python -m AgentQMS.tools.core.plugins --show audit

# Validate all plugins
python -m AgentQMS.tools.core.plugins --validate

# Get JSON output for processing
python -m AgentQMS.tools.core.plugins --list --json > /tmp/plugins.json

# Count lines of code
find AgentQMS/tools -name "*.py" -exec wc -l {} \; | awk '{s+=$1} END {print s}'

# Find old files (not updated in 30 days)
find AgentQMS/tools -name "*.py" -mtime +30 -type f

# Check imports in a file
python -c "import ast; ast.parse(open('AgentQMS/tools/utilities/grok_linter.py').read())"
```

### Compliance Checks
```bash
# Run all validations
cd AgentQMS/bin && make validate && make compliance

# Check specific artifact type
python -m pytest AgentQMS/tests/test_artifact_type_validation.py::test_canonical_type_assessment_accepted -v

# Dry-run artifact creation (if available)
cd AgentQMS/bin && make create-plan NAME=test-plan TITLE="Test" --dry-run
```

---

## Emergency Procedures

### If Plugin System Breaks

1. **Restore from Git**:
   ```bash
   git status                    # Check what changed
   git diff AgentQMS/           # Review changes
   git checkout HEAD -- AgentQMS/ # Revert to last commit
   ```

2. **Rebuild Plugin Registry**:
   ```bash
   python -m AgentQMS.tools.core.plugins --write-snapshot
   ```

3. **Verify Recovery**:
   ```bash
   python -m pytest AgentQMS/tests/ -v
   python -m AgentQMS.tools.core.plugins --list
   ```

### If Tests Start Failing

1. **Check for import issues**:
   ```bash
   python -m pytest AgentQMS/tests/ -v --tb=short
   ```

2. **Check for schema changes**:
   ```bash
   python -m AgentQMS.tools.core.plugins --validate
   ```

3. **Rebuild everything**:
   ```bash
   cd AgentQMS/bin && make clean && make plugins-snapshot
   ```

---

## Documentation References

### For Understanding AgentQMS
1. **Architecture**: `AgentQMS/bin/README.md`
2. **Standards**: `AgentQMS/standards/INDEX.yaml`
3. **Framework Overview**: `AgentQMS/AGENTS.yaml`
4. **Plugin System**: `AgentQMS/tools/core/plugins/` (docstrings)
5. **Validation Rules**: `.agentqms/schemas/artifact_type_validation.yaml`

### For Understanding Specific Components
- **Artifact Creation**: `AgentQMS/tools/core/artifact_workflow.py`
- **Validation**: `AgentQMS/tools/compliance/validate_artifacts.py`
- **Plugin Discovery**: `AgentQMS/tools/core/plugins/discovery.py`
- **Configuration**: `AgentQMS/tools/utils/` directory

### Audit Reports
- **Current Audit**: `docs/artifacts/2026-01-10_1730_assessment_agentqms-audit.md`
- **Previous Work**: Check `docs/artifacts/` for history

---

## Contact & Escalation

**For Technical Issues**:
1. Check this guide (AGENTQMS_MAINTENANCE.md)
2. Review audit report: `docs/artifacts/2026-01-10_1730_assessment_agentqms-audit.md`
3. Check Git history: `git log --oneline AgentQMS/`
4. Run diagnostics: See "Audit Tools & Commands" above

**For Architecture Questions**:
1. Read AgentQMS/standards/INDEX.yaml
2. Review AgentQMS/bin/README.md
3. Check plugin definitions in `.agentqms/plugins/`

**For Phase 2 Cleanup Authorization**:
- Current cleanup plan documented in audit report
- Ready to execute when approved
- Estimated 10-14 hours of work

---

## Session History

| Date | Session | Work Done | Status |
|------|---------|-----------|--------|
| 2026-01-10 | Audit & CI Fix | Fixed plugin schema validation, comprehensive audit | ✅ COMPLETE |
| 2026-01-10 | Workflow Scripts | Fixed deprecated paths in validate.sh, compliance.sh, init_framework.sh | ✅ COMPLETE |
| Previous | README/Makefile | Updated documentation with KST timestamps, fixed paths | ✅ COMPLETE |
| Previous | Plugin Standardization | Standardized all plugins to use ads_version | ✅ COMPLETE |

---

## Key Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Architecture Health | 7/10 | ⚠️ Good but needs cleanup |
| Active Code | 52 files, 14,600 LOC | ✅ Well-organized |
| Legacy Code | 18 files, 2,100 LOC | ⚠️ Documented, not blocking |
| Test Coverage | 18/18 tests passing | ✅ Healthy |
| Plugin System | 9 types + 12 bundles | ✅ Fully functional |
| Documentation | Complete for active areas | ⚠️ Needs updates after cleanup |

---

## Final Notes

✅ **System Status**: Fully operational and stable  
✅ **Technical Debt**: Documented and prioritized  
✅ **Remediation Plan**: Ready for execution  
✅ **Documentation**: Comprehensive and up-to-date  

**Next Priority**: Execute Phase 2 cleanup when ready (10-14 hours estimated)

**Ready for**: Production use, continued development, team handoff
