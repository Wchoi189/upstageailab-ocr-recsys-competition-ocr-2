# Session 4 Preparation: Plugin Migration & Hardcoded Removal

**Date Prepared**: 2026-01-10
**Status**: Ready for Session 4 start
**Estimated Work**: 2-3 hours

## Situation Summary

### What We Found
Phase 2 work has been **partially completed** in a prior session:
- ✅ All 11 plugin YAML files **already exist** in `AgentQMS/.agentqms/plugins/artifact_types/`
- ✅ Plugin system **already integrated** into `ArtifactTemplates` class
- ❌ **BUT** hardcoded templates take precedence over plugins
- ❌ Missing one plugin: `design` (hardcoded exists but plugin doesn't)
- ❌ Hardcoded dictionary still controls artifact creation

### Current State: Dual System
```
Hardcoded Templates (12):  assessment, audit, bug_report, change_request,
                           design, design_document, implementation_plan,
                           ocr_experiment_report, research, template,
                           vlm_report, walkthrough

Plugin Templates (11):      assessment, audit, bug_report, change_request,
                           design_document, implementation_plan,
                           ocr_experiment_report, research, template,
                           vlm_report, walkthrough

MISSING: design plugin (only hardcoded exists)
```

### Problem: Precedence Issue
In `ArtifactTemplates._load_plugin_templates()` (line 339):
```python
# Skip if already defined (builtin takes precedence)
if name in self.templates:
    continue
```

**Result**: Even though plugins exist, hardcoded versions are ALWAYS used.

## Session 4 Scope

### Task 1: Create Missing 'design' Plugin
**File**: `AgentQMS/.agentqms/plugins/artifact_types/design.yaml`
**Source**: Extract from hardcoded in `artifact_templates.py` lines 146-183
**Work**: Create plugin YAML with identical structure and content

### Task 2: Reverse Precedence (Critical Fix)
**File**: `AgentQMS/tools/core/artifact_templates.py` lines 316-323
**Change**: Plugins should OVERRIDE hardcoded, not vice versa
**Pattern**: Load hardcoded first, then let plugins override
```python
# Current (wrong):
1. Load hardcoded
2. Load plugins (skip if hardcoded already exists) ❌

# New (correct):
1. Load hardcoded as base layer
2. Merge plugins (plugins override hardcoded) ✅
```

### Task 3: Create Equivalence Tests
**File**: Create `tests/test_plugin_vs_hardcoded_equivalence.py`
**Purpose**: Verify plugin output matches hardcoded for all 12 types
**Tests**:
- For each type: compare filename patterns
- Compare frontmatter structure
- Compare content templates
- Verify all fields present in both

### Task 4: Validate Complete Migration
**Work**: Run artifact creation with each type, verify plugins are used
**Tests**:
- Create artifact with each type
- Inspect generated artifact metadata
- Confirm source is "plugin" (after precedence fix)

### Task 5: Document Migration Status
**Files**:
- Update session_4_completion.md
- Update Phase 2 status in current_session.yml

## Implementation Details

### 1. Create design.yaml Plugin

Extract from hardcoded definition (lines 146-183 in artifact_templates.py):
```python
"design": {
    "filename_pattern": "YYYY-MM-DD_HHMM_design_{name}.md",
    "directory": "designs/",
    "frontmatter": {
        "ads_version": "1.0",
        "type": "design",
        "category": "architecture",
        "status": "active",
        "version": "1.0",
        "tags": ["design", "architecture", "specification"],
    },
    "content_template": "# Design - {title}\n\n## Overview\n..."
}
```

Convert to YAML format matching existing plugins (assessment.yaml pattern).

### 2. Fix Precedence in artifact_templates.py

**Current code** (line 316-323):
```python
def _load_plugin_templates(self) -> None:
    """Load additional artifact templates from plugin registry."""
    if not PLUGINS_AVAILABLE:
        return
    try:
        registry = get_plugin_registry()
        artifact_types = registry.get_artifact_types()
        for name, plugin_def in artifact_types.items():
            # Skip if already defined (builtin takes precedence)
            if name in self.templates:
                continue  # ❌ WRONG: prevents plugin override
```

**Fixed code** should be:
```python
def _load_plugin_templates(self) -> None:
    """Load artifact templates from plugin registry.

    Plugins OVERRIDE hardcoded templates for the same type name.
    This allows extension and customization.
    """
    if not PLUGINS_AVAILABLE:
        return
    try:
        registry = get_plugin_registry()
        artifact_types = registry.get_artifact_types()
        for name, plugin_def in artifact_types.items():
            # Load and MERGE plugin (overrides hardcoded if present)
            template = self._convert_plugin_to_template(name, plugin_def)
            if template:
                self.templates[name] = template  # ✅ CORRECT: plugins override
```

### 3. Equivalence Test Structure

```python
class TestPluginVsHardcodedEquivalence:
    """Verify plugin artifacts are functionally identical to hardcoded."""

    def test_assessment_plugin_matches_hardcoded(self):
        """assessment should produce identical output via plugin vs hardcoded."""
        # Get both versions
        hardcoded = get_hardcoded_template('assessment')
        plugin = get_plugin_template('assessment')
        # Compare structure
        assert hardcoded['frontmatter'] == plugin['frontmatter']
        assert hardcoded['filename_pattern'] == plugin['filename_pattern']

    # Similar tests for all 12 types...
```

### 4. Validation Plan

After implementing changes:
1. Run equivalence tests - should pass
2. Create test artifact with each type
3. Verify artifact metadata shows plugin source
4. Run full artifact workflow tests
5. Check no regressions in artifact creation

## Files Involved

| File | Role | Status |
|------|------|--------|
| `AgentQMS/.agentqms/plugins/artifact_types/design.yaml` | New plugin | To create |
| `AgentQMS/tools/core/artifact_templates.py` | Core logic | Fix precedence |
| `tests/test_plugin_vs_hardcoded_equivalence.py` | New tests | To create |
| `AgentQMS/tools/core/artifact_templates.py` (hardcoded dict) | Source data | Keep for now |

## Why This Approach?

### Keep Hardcoded Initially
- Hardcoded serves as reference/fallback
- Easier to compare plugin vs hardcoded
- Can validate equivalence before removal
- Reduces risk of breaking artifact creation

### Plugin Override Pattern
- Standard plugin pattern: plugins extend/override framework
- Allows gradual testing
- Makes precedence explicit
- Enables A/B testing of hardcoded vs plugin

### Equivalence Tests
- Ensure no functional change
- Document expected behavior
- Catch regressions
- Enable confidence in later removal (Phase 4)

## Success Criteria

✅ **Functional**
- Design plugin created with complete definition
- Hardcoded templates no longer override plugins
- All 12 types available via plugins
- Artifact creation uses plugin templates

✅ **Quality**
- Equivalence tests comprehensive (12 types covered)
- All tests passing
- No regressions in artifact workflow
- Plugin output matches hardcoded exactly

✅ **Documentation**
- Design plugin documented
- Precedence change documented
- Phase 2 status updated

## Estimated Effort

| Task | Time |
|------|------|
| Create design.yaml plugin | 15-20 min |
| Fix precedence logic | 15-20 min |
| Create equivalence tests | 30-45 min |
| Manual validation | 15-20 min |
| Documentation | 10-15 min |
| **Total** | **85-120 min** |

## Session 4 Checklist

- [ ] Read artifact_templates.py line 146-183 (design template)
- [ ] Create design.yaml from hardcoded definition
- [ ] Update _load_plugin_templates() to enable precedence override
- [ ] Create test_plugin_vs_hardcoded_equivalence.py
- [ ] Write 12 equivalence tests (one per type)
- [ ] Run tests - verify all passing
- [ ] Manual test: create artifact of each type
- [ ] Verify plugins are being used
- [ ] Update session_4_completion.md
- [ ] Update Phase 2 status tracking

## Notes for Session 4

1. **Backup Approach**: If plugin override causes issues, can add feature flag
2. **Testing**: Run existing artifact workflow tests to catch regressions immediately
3. **Validation**: Check that created artifacts have correct metadata
4. **Documentation**: Record any edge cases or special handling needed

## Next Phase After Session 4

**Phase 3**: Validation & Naming Conflicts
- Centralize validation rules
- Resolve assessment/design conflicts
- Document canonical names

**Phase 4**: Hardcoded Removal
- Once equivalence proven, remove hardcoded dictionary
- Simplify ArtifactTemplates to pure plugin wrapper
- Archive legacy code

## Session Handover Info

**Phase 2 Status**: ~50% complete (plugins exist, but need precedence fix + design)
**Blocker**: None - can proceed with session 4
**Previous Docs**: session_3_completion.md, current_session.yml
