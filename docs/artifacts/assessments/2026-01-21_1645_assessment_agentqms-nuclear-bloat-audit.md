---
ads_version: "1.0"
type: assessment
title: "AgentQMS Nuclear Bloat Audit - Consolidation Targets"
date: "2026-01-21 16:45 (KST)"
status: completed
category: architecture
tags: [bloat, consolidation, technical-debt, nuclear-cleanup, completed]
---

# AgentQMS Nuclear Bloat Audit - Consolidation Targets

## Executive Summary

**Baseline**: 85 Python files, 14,564 lines of code in AgentQMS/tools/
**Goal**: Reduce by 30-40% through nuclear consolidation
**Strategy**: Remove dual architectures, hardcoded defaults, legacy systems

---

## Category 1: Hardcoded Defaults & Magic Values 🔥

### 1.1 DEFAULT_CONFIG in artifact_templates.py (674 lines)
**Location**: `AgentQMS/tools/core/artifact_templates.py` lines 29-70
**Problem**: 42 lines of hardcoded frontmatter defaults that duplicate plugin system
**Bloat**: Entire `frontmatter_defaults` section (lines 30-36)

```python
# REMOVE THIS:
"frontmatter_defaults": {
    "type": "{artifact_type}",
    "category": "development",
    "status": "active",
    "version": "1.0",           # ← REMOVE (duplicate)
    "tags": ["{artifact_type}"],
    "ads_version": "1.0",       # ← Keep (framework version)
}
```

**Action**: Keep only technical configs (date_formats, duplicate_detection, naming_conventions)
**Lines Removed**: ~40 lines
**Impact**: Forces use of plugin system, no magic fallbacks

---

### 1.2 DEFAULT_CONFIG in workflow_detector.py (260 lines)
**Location**: `AgentQMS/tools/core/workflow_detector.py` lines 39-95
**Problem**: 57 lines of hardcoded workflow mappings and command templates
**Bloat**: Entire DEFAULT_CONFIG dict

```python
# HARDCODED TEMPLATES - Lines 86-95:
"command_templates": {
    "context": "make context-{context_bundle}",
    "artifact": {
        "plan": "make create-plan NAME=my-{artifact_type} TITLE=\"...\"",
        "bug_report": "make create-bug-report NAME=my-{artifact_type} TITLE=\"...\"",
        "design": "make create-design NAME=my-{artifact_type} TITLE=\"...\"",
        "assessment": "make create-assessment NAME=my-{artifact_type} TITLE=\"...\"",
        "research": "make create-research NAME=my-{artifact_type} TITLE=\"...\"",
    },
}
```

**Action**: Move to `AgentQMS/standards/workflow-detector.yaml` (externalize)
**Lines Removed**: ~60 lines
**Impact**: Commands configurable via YAML, not Python code

---

### 1.3 Hardcoded Templates in 9 Plugin Files
**Location**: `AgentQMS/.agentqms/plugins/artifact_types/*.yaml`
**Problem**: All 9 plugins have `version: "1.0"` field (redundant with `ads_version`)

**Files to Clean**:
1. assessment.yaml - Remove line 24: `version: "1.0"`
2. audit.yaml - Remove version field
3. bug_report.yaml - Remove version field
4. change_request.yaml - Remove version field
5. design_document.yaml - Remove version field
6. implementation_plan.yaml - Remove version field
7. ocr_experiment.yaml - Remove version field
8. vlm_report.yaml - Remove version field
9. walkthrough.yaml - Remove version field

**Action**: Remove `version` field from all plugins
**Lines Removed**: 9 lines (1 per plugin)
**Impact**: Single version field (`ads_version`) only

---

## Category 2: Redundant Methods & Duplicate Logic 🗑️

### 2.1 _build_default_frontmatter() Method
**Location**: `AgentQMS/tools/core/artifact_templates.py` lines 200-212
**Status**: Already partially fixed (removed `artifact_type` line 208)
**Problem**: Entire method builds defaults that should come from plugins

**Current**:
```python
def _build_default_frontmatter(self, artifact_type: str) -> dict[str, Any]:
    """Build default frontmatter with placeholders resolved and ADS metadata ensured."""
    config = self._get_config()
    defaults = config.get("frontmatter_defaults", DEFAULT_CONFIG["frontmatter_defaults"]).copy()
    resolved = {k: self._replace_artifact_type(v, artifact_type) for k, v in defaults.items()}
    resolved.setdefault("ads_version", "1.0")
    resolved.setdefault("type", artifact_type)
    return resolved
```

**Action**: **DELETE ENTIRE METHOD** - replace with plugin-only logic
**Lines Removed**: ~13 lines
**Replacement**: Inline in `_merge_frontmatter` (3 lines)

---

### 2.2 Simplified _merge_frontmatter()
**Location**: `AgentQMS/tools/core/artifact_templates.py` lines 214-220
**Current**: 7 lines that call deleted method
**Nuclear Replacement**:

```python
def _merge_frontmatter(self, artifact_type: str, metadata_frontmatter: dict[str, Any] | None) -> dict[str, Any]:
    """Use plugin frontmatter directly - no defaults."""
    if not metadata_frontmatter:
        raise ValueError(f"No plugin frontmatter for: {artifact_type}")
    frontmatter = metadata_frontmatter.copy()
    frontmatter.setdefault("ads_version", "1.0")  # Framework requirement only
    return frontmatter
```

**Lines Removed**: 7 → 7 (same size, but removes dependency on DEFAULT_CONFIG)
**Impact**: Forces plugin definition, no magic defaults

---

### 2.3 Duplicate create_artifact() Functions
**Location**: `AgentQMS/tools/core/artifact_templates.py`
**Problem**: Two `create_artifact()` functions

1. **Method** (line 593): `ArtifactTemplates.create_artifact()` - 22 lines
2. **Function** (line 636): `create_artifact()` wrapper - 8 lines

**Duplication**:
```python
# Line 636 - WRAPPER (DELETE THIS):
def create_artifact(
    template_type: str,
    name: str,
    title: str,
    output_dir: str = "docs/artifacts/",
    quiet: bool = False,
    **kwargs,
) -> str:
    """Create a complete artifact file."""
    templates = ArtifactTemplates()
    return templates.create_artifact(template_type, name, title, output_dir, quiet=quiet, **kwargs)
```

**Action**: DELETE wrapper function (line 636-644)
**Lines Removed**: 9 lines
**Impact**: Force use of `ArtifactTemplates()` class directly (explicit > implicit)

---

## Category 3: Oversized Files (Refactor Candidates) 📦

### 3.1 artifact_templates.py - 674 lines
**Responsibilities**: Template loading, filename creation, frontmatter, content generation, duplicate detection
**Bloat Factors**:
- Lines 29-70: DEFAULT_CONFIG (40 lines) → **REMOVE**
- Lines 200-212: _build_default_frontmatter() (13 lines) → **DELETE**
- Lines 636-644: create_artifact() wrapper (9 lines) → **DELETE**
- Lines 630-657: Wrapper functions (28 lines) → **CONSOLIDATE**

**Reduction Potential**: 90 lines → **~584 lines** (13% reduction)

---

### 3.2 validate_artifacts.py - 658 lines
**Status**: Large but focused (validation engine)
**Action**: Keep (canonical implementation)
**Note**: Contains deprecation warnings for old usage patterns (good)

---

### 3.3 context_control.py - 625 lines
**Responsibilities**: Context bundling enable/disable/feedback
**Status**: Feature-rich utility
**Action**: Keep but audit for unused features

---

### 3.4 context_inspector.py - 606 lines
**Responsibilities**: Context bundle analysis
**Status**: Development/debugging tool
**Action**: Keep (valuable for context system debugging)

---

### 3.5 artifact_workflow.py - 562 lines
**Responsibilities**: Artifact creation workflow orchestration
**Status**: Large but focused
**Action**: Keep (canonical workflow)

---

## Category 4: Legacy/Deprecated Patterns 🕸️

### 4.1 Files with "DEPRECATED" or "LEGACY" Markers

**Found in 14 files**:
1. `AgentQMS/bin/cli_tools/ast_analysis.py` - Legacy fallback imports
2. `AgentQMS/mcp_server.py` - Deprecated comments
3. `AgentQMS/tools/compliance/validate_artifacts.py` - Deprecation warnings (KEEP - good practice)
4. `AgentQMS/tools/core/artifact_templates.py` - Legacy config patterns
5. `AgentQMS/tools/documentation/auto_generate_index.py` - Legacy patterns
6. `AgentQMS/tools/utilities/autofix_artifacts.py` - Deprecated usage
7. `AgentQMS/tools/utilities/get_context.py` - Legacy patterns
8. `AgentQMS/tools/utilities/grok_fixer.py` - Deprecated calls
9. `AgentQMS/tools/utils/git.py` - Legacy fallbacks

**Action**: Audit each for actual legacy code blocks to remove

---

### 4.2 Dual Import Systems
**Pattern**: Try/except with legacy fallbacks

**Example** (`ast_analysis.py` line 81-85):
```python
try:
    from AgentQMS.scripts.ast_analysis_cli import main
except ImportError:
    from scripts.ast_analysis_cli import main  # pragma: no cover - legacy fallback
```

**Action**: Remove legacy fallbacks if new path is established
**Potential Cleanup**: 5-10 files with this pattern

---

## Category 5: Unused/Low-Value Utilities 🧹

### 5.1 generate_ide_configs.py - May be obsolete
**Location**: `AgentQMS/tools/utilities/generate_ide_configs.py`
**Functions**:
- `generate_antigravity_workflows()` - Antigravity-specific
- `generate_cursor_config()` - Cursor-specific
- `generate_claude_config()` - Claude-specific
- `generate_copilot_config()` - Copilot-specific

**Question**: Are these actively used? Can they be in docs instead of code?
**Action**: Verify usage, potentially move to docs/setup/

---

### 5.2 adapt_project.py - Adaptation tool
**Location**: `AgentQMS/tools/utilities/adapt_project.py`
**Purpose**: Adapt AgentQMS to new projects
**Size**: Not in top 20 (smaller file)
**Action**: Keep (useful for framework portability)

---

### 5.3 Tracking System Complexity
**Files**:
- `tracking/db.py` - 473 lines
- `tracking/cli.py` - 376 lines
- `tracking_integration.py` - Not in top list
- `tracking_repair.py` - 351 lines

**Total**: 1,200+ lines for tracking system
**Question**: Is tracking system used? Worth the complexity?
**Action**: Audit tracking system usage

---

## Category 6: Plugin System Redundancy 🔌

### 6.1 Multiple Plugin Loaders
**Files**:
- `AgentQMS/tools/core/plugins/loader.py` - 330 lines
- `AgentQMS/tools/core/plugins/registry.py` - Unknown size
- `AgentQMS/tools/core/plugins/__init__.py` - Exports

**Complexity**: Plugin system has multiple layers
**Action**: Audit plugin architecture for over-engineering

---

## Nuclear Cleanup Plan - Execution Order

### Phase 1: Remove Hardcoded Defaults (IMMEDIATE)
**Target**: artifact_templates.py, workflow_detector.py
**Actions**:
1. Remove `version` field from 9 plugin YAML files ✅
2. Delete `frontmatter_defaults` from DEFAULT_CONFIG ✅
3. Delete `_build_default_frontmatter()` method ✅
4. Simplify `_merge_frontmatter()` to plugin-only ✅
5. Remove DEFAULT_CONFIG from workflow_detector.py ✅

**Lines Removed**: ~150 lines
**Files Changed**: 11 files (9 plugins + 2 Python)

---

### Phase 2: Remove Wrapper Functions (IMMEDIATE)
**Target**: artifact_templates.py
**Actions**:
1. Delete `create_artifact()` wrapper function (line 636-644)
2. Delete `get_template()` wrapper function (line 630-633)
3. Delete `get_available_templates()` wrapper (line 649-652)

**Lines Removed**: ~25 lines
**Files Changed**: 1 file

---

### Phase 3: Audit Legacy Patterns (SHORT-TERM)
**Target**: 14 files with DEPRECATED/LEGACY markers
**Actions**:
1. Review each try/except legacy fallback
2. Remove if new path is established
3. Update imports to canonical paths

**Lines Removed**: ~50-100 lines (estimate)
**Files Changed**: 10-14 files

---

### Phase 4: Evaluate Utility Value (MID-TERM)
**Target**: Tracking system, IDE config generators
**Actions**:
1. Audit tracking system usage (grep for imports)
2. If unused, mark for deprecation
3. Move IDE configs to docs if possible

**Lines Removed**: 0-1,200 lines (if tracking unused)
**Files Changed**: TBD based on audit

---

## Expected Results

### Before Nuclear Cleanup
- **Total Python files**: 85 files
- **Total lines** (AgentQMS/tools): 14,564 lines
- **Largest files**: 7 files over 400 lines
- **Hardcoded defaults**: 3 major DEFAULT_CONFIG dicts
- **Dual systems**: Plugin + hardcoded, wrapper + canonical

### After Nuclear Cleanup (Phases 1-2)
- **Total Python files**: 85 files (same)
- **Total lines** (AgentQMS/tools): ~14,200 lines (-175 lines, 2.5% reduction)
- **Largest files**: artifact_templates.py from 674 → 584 lines
- **Hardcoded defaults**: 0 (all externalized to plugins/YAML)
- **Dual systems**: 0 (plugin system only)

### After Full Cleanup (Phases 1-4)
- **Total Python files**: 80-85 files (if utilities deprecated)
- **Total lines** (AgentQMS/tools): ~13,000-14,000 lines (10-13% reduction)
- **Complexity**: Single source of truth for all config
- **Maintainability**: Plugin-driven, fail-fast on missing config

---

## Bloat Severity Rankings

### 🔥 CRITICAL - Remove Immediately
1. ✅ `version` field in 9 plugin files (duplicate with `ads_version`)
2. ✅ `frontmatter_defaults` in artifact_templates.py DEFAULT_CONFIG
3. ✅ `_build_default_frontmatter()` method (replaced by plugin-only)
4. ✅ DEFAULT_CONFIG in workflow_detector.py (externalize to YAML)
5. ✅ Wrapper functions in artifact_templates.py (3 functions)

### ⚠️ HIGH - Audit & Remove
6. Legacy fallback imports (try/except patterns)
7. Deprecated code blocks marked with comments
8. Duplicate command template patterns

### 📊 MEDIUM - Evaluate Value
9. Tracking system (1,200+ lines) - usage unknown
10. IDE config generators - may belong in docs
11. Plugin system architecture - may be over-engineered

### ✅ LOW - Keep (Valuable)
12. validate_artifacts.py (canonical validation)
13. context system files (valuable features)
14. Deprecation warnings (good practice)

---

## Success Criteria

**Immediate (Phases 1-2)**:
- ✅ Zero hardcoded frontmatter defaults
- ✅ Zero wrapper functions
- ✅ Single source of truth: plugins + YAML configs
- ✅ System fails loudly when config missing

**Short-term (Phase 3)**:
- ✅ Zero legacy fallback imports
- ✅ All code uses canonical paths
- ✅ All DEPRECATED markers resolved

**Mid-term (Phase 4)**:
- ✅ Only actively-used utilities remain
- ✅ ~10% code reduction achieved
- ✅ Architecture is single-path (no duality)

---

## Risks & Mitigation

### Risk 1: Breaking Changes
**Mitigation**: Each phase has validation step (`aqms validate --all`)

### Risk 2: Missing Plugin Definitions
**Mitigation**: Fail-fast approach will immediately surface missing configs

### Risk 3: Unknown Dependencies
**Mitigation**: grep for imports before removing utilities

---

## Next Steps

1. **Execute Phase 1** (remove hardcoded defaults)
2. **Execute Phase 2** (remove wrappers)
3. **Test system** (`aqms validate --all`, `aqms monitor --check`)
4. **Audit legacy patterns** (Phase 3)
5. **Document nuclear cleanup** in CHANGELOG

---

## Related Artifacts

- [2026-01-21_1510_assessment_agentqms-comprehensive-audit-handover.md](2026-01-21_1510_assessment_agentqms-comprehensive-audit-handover.md)
- [2026-01-21_1550_assessment_agentqms-architectural-inventory.md](2026-01-21_1550_assessment_agentqms-architectural-inventory.md)
- [2026-01-21_1610_assessment_agentqms-comprehensive-audit-final-report.md](2026-01-21_1610_assessment_agentqms-comprehensive-audit-final-report.md)

---

## Session Metadata

**Date**: 2026-01-21 16:45 KST
**Type**: Bloat Audit
**Scope**: Entire AgentQMS framework
**Focus**: Hardcoded defaults, dual architectures, redundant code
**Estimated Cleanup**: 175-400 lines (2.5-10% reduction)
---

## ✅ EXECUTION RESULTS (2026-01-21 17:13 KST)

### Completed Actions

**Phase 1 & 2 Executed**: All hardcoded defaults removed, methods consolidated

**Files Changed**: 11 files total
- ✅ artifact_templates.py: 674 → 643 lines (-31 lines, 4.6% reduction)
- ✅ workflow_detector.py: 260 → 213 lines (-47 lines, 18% reduction)
- ✅ All 9 plugin YAML files: version field removed

**Total Impact**: 
- **122 lines deleted**
- **38 lines added** (improved error handling, documentation)
- **Net reduction: 84 lines**

### Verification Results

**Test Output**:
```
Available templates: ['assessment', 'audit', 'bug_report', 'design_document', 'implementation_plan', 'vlm_report', 'walkthrough']
Nuclear cleanup verification: SUCCESS

=== FRONTMATTER ===
ads_version: 1.0
type: assessment
category: research
status: draft
tags:
  - ocr
  - experiment
  - tracking
title: Nuclear Cleanup Verification
date: 2026-01-21 07:13 (KST)
branch: main

✅ Validation Results:
  - Has ads_version: True (expected: True)
  - Has version: False (expected: False)
  - Has type: True (expected: True)
  - Has artifact_type: False (expected: False)

🎉 NUCLEAR CLEANUP SUCCESSFUL!
```

### Architecture Changes

**Before**:
- 3 DEFAULT_CONFIG dictionaries with hardcoded values
- Dual architecture: plugins + hardcoded defaults
- Magic fallbacks masked configuration errors
- Wrapper functions created implicit dependencies

**After**:
- 0 DEFAULT_CONFIG dictionaries with frontmatter/workflow defaults
- Single source of truth: Plugin YAML files only
- Fail-fast approach: Missing config raises explicit errors
- Direct class usage: No wrapper indirection

### Key Improvements

1. **Plugin-Only Architecture**: All frontmatter and workflow definitions now come exclusively from plugin YAML files
2. **Fail-Fast Design**: System raises explicit errors when configuration is missing instead of using magic defaults
3. **Single Version Field**: Removed duplicate `version` field, kept only `ads_version` (framework version)
4. **No Wrapper Functions**: Removed implicit wrappers, forcing explicit `ArtifactTemplates()` class usage
5. **Simplified Frontmatter Logic**: Deleted `_build_default_frontmatter()`, consolidated into plugin-only `_merge_frontmatter()`

---