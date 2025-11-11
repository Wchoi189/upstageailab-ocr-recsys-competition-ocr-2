# Phase 2: Agent Tools Comparison and Integration Plan

**Date:** 2025-01-XX
**Status:** In Progress
**Phase:** 2 - Agent Tools Integration

---

## Overview

This document compares the existing `scripts/agent_tools/` with QMF's `agent_tools/` to identify overlaps, complementary tools, and integration strategy.

---

## Structure Comparison

### Current Project Structure
```
scripts/agent_tools/
├── auto_generate_index.py
├── cleanup_remaining_checkpoints.py
├── context_log.py
├── declutter_root.py
├── delegate_to_qwen.py
├── generate_checkpoint_configs.py
├── generate_handbook_index.py
├── get_context.py
├── next_run_proposer.py
├── quick_fix_log.py
├── remove_low_hmean_checkpoints.py
├── remove_low_step_checkpoints.py
├── run_agent_demo.py
├── strip_doc_markers.py
├── summarize_run.py
├── test_validation.py
├── validate_manifest.py
└── verify_best_checkpoints.py
```
**Structure:** Flat (all tools in root)
**Total:** 18 Python files

### QMF Structure
```
agent_tools/
├── __init__.py
├── core/
│   ├── artifact_workflow.py
│   ├── context_bundle.py
│   └── discover.py
├── compliance/
│   ├── compliance_alert_system.py
│   ├── compliance_dashboard.py
│   ├── compliance_trend_tracker.py
│   ├── daily_compliance_monitor.py
│   ├── documentation_quality_monitor.py
│   ├── fix_artifacts.py
│   ├── monitor_artifacts.py
│   └── validate_artifacts.py
├── documentation/
│   ├── auto_generate_index.py
│   ├── check_freshness.py
│   ├── deprecate_docs.py
│   ├── generate_changelog_draft.py
│   ├── regenerate_docs.py
│   ├── update_artifact_indexes.py
│   ├── validate_coordinate_consistency.py
│   ├── validate_links.py
│   ├── validate_manifest.py
│   ├── validate_metadata.py
│   ├── validate_templates.py
│   └── validate_ui_schema.py
├── maintenance/
│   ├── add_frontmatter.py
│   ├── fix_categories.py
│   ├── fix_naming_conventions.py
│   ├── reorganize_files.py
│   └── update_docs.py
└── utilities/
    ├── adapt_project.py
    ├── agent_feedback.py
    ├── browser_extension_wrapper.py
    ├── clean_logs.py
    ├── export_framework.py
    ├── generate_tool_catalog.py
    ├── get_context.py
    ├── puppeteer_wrapper.py
    ├── quick_fix_log.py
    ├── tracking/
    │   ├── cli.py
    │   ├── db.py
    │   └── query.py
    └── view_logs.py
```
**Structure:** Organized by category (core, compliance, documentation, maintenance, utilities)
**Total:** 47 Python files

---

## Tool Comparison

### Direct Overlaps (Same Name, Similar Function)

| Tool | Current Project | QMF | Action |
|------|----------------|-----|--------|
| `auto_generate_index.py` | ✅ Present | ✅ Present | **Adopt QMF** (better UTC handling) |
| `validate_manifest.py` | ✅ Present | ✅ Present | **Adopt QMF** (better error handling) |
| `quick_fix_log.py` | ✅ Present | ✅ Present | **Adopt QMF** (better formatting) |
| `get_context.py` | ✅ Present (simple) | ✅ Present (advanced) | **Adopt QMF** (YAML bundles, bootstrap) |

### OCR-Specific Tools (Keep Current)

| Tool | Purpose | Action |
|------|---------|--------|
| `cleanup_remaining_checkpoints.py` | OCR checkpoint management | **Keep** |
| `generate_checkpoint_configs.py` | OCR checkpoint configs | **Keep** |
| `next_run_proposer.py` | OCR training run proposals | **Keep** |
| `remove_low_hmean_checkpoints.py` | OCR checkpoint filtering | **Keep** |
| `remove_low_step_checkpoints.py` | OCR checkpoint filtering | **Keep** |
| `verify_best_checkpoints.py` | OCR checkpoint verification | **Keep** |
| `summarize_run.py` | OCR run summarization | **Keep** |
| `delegate_to_qwen.py` | OCR-specific delegation | **Keep** |
| `run_agent_demo.py` | OCR-specific demo | **Keep** |
| `context_log.py` | OCR-specific context logging | **Keep** |
| `declutter_root.py` | OCR-specific cleanup | **Keep** |
| `strip_doc_markers.py` | OCR-specific doc processing | **Keep** |
| `test_validation.py` | OCR-specific testing | **Keep** |
| `generate_handbook_index.py` | OCR-specific index generation | **Keep** (or merge with auto_generate_index) |

### QMF Tools to Adopt (New Capabilities)

#### Core Tools
- `core/artifact_workflow.py` - **Adopt** (uses AgentQMS toolbelt)
- `core/context_bundle.py` - **Adopt** (context bundle generation)
- `core/discover.py` - **Adopt** (tool discovery)

#### Compliance Tools
- `compliance/documentation_quality_monitor.py` - **Adopt** (quality monitoring)
- `compliance/validate_artifacts.py` - **Adopt** (artifact validation)
- `compliance/monitor_artifacts.py` - **Adopt** (artifact monitoring)
- `compliance/fix_artifacts.py` - **Adopt** (artifact fixing)
- `compliance/compliance_alert_system.py` - **Evaluate** (may be overkill)
- `compliance/compliance_dashboard.py` - **Evaluate** (may be overkill)
- `compliance/compliance_trend_tracker.py` - **Evaluate** (may be overkill)
- `compliance/daily_compliance_monitor.py` - **Evaluate** (may be overkill)

#### Documentation Tools
- `documentation/check_freshness.py` - **Adopt** (doc freshness)
- `documentation/validate_links.py` - **Adopt** (link validation)
- `documentation/validate_templates.py` - **Adopt** (template validation)
- `documentation/validate_metadata.py` - **Adopt** (metadata validation)
- `documentation/regenerate_docs.py` - **Adopt** (doc regeneration)
- `documentation/update_artifact_indexes.py` - **Adopt** (artifact indexing)
- `documentation/deprecate_docs.py` - **Adopt** (doc deprecation)
- `documentation/generate_changelog_draft.py` - **Adopt** (changelog generation)
- `documentation/validate_ui_schema.py` - **Evaluate** (UI-specific)
- `documentation/validate_coordinate_consistency.py` - **Evaluate** (UI-specific)

#### Maintenance Tools
- `maintenance/add_frontmatter.py` - **Adopt** (frontmatter generation)
- `maintenance/fix_naming_conventions.py` - **Adopt** (naming fixes)
- `maintenance/fix_categories.py` - **Adopt** (category fixes)
- `maintenance/reorganize_files.py` - **Adopt** (file reorganization)
- `maintenance/update_docs.py` - **Adopt** (doc updates)

#### Utilities
- `utilities/clean_logs.py` - **Adopt** (log cleaning)
- `utilities/view_logs.py` - **Adopt** (log viewing)
- `utilities/generate_tool_catalog.py` - **Adopt** (tool catalog)
- `utilities/export_framework.py` - **Adopt** (framework export)
- `utilities/agent_feedback.py` - **Evaluate** (feedback system)
- `utilities/browser_extension_wrapper.py` - **Evaluate** (browser integration)
- `utilities/puppeteer_wrapper.py` - **Evaluate** (puppeteer integration)
- `utilities/adapt_project.py` - **Evaluate** (project adaptation)
- `utilities/tracking/` - **Evaluate** (tracking system - may be useful)

---

## Integration Strategy

### Step 1: Reorganize Current Tools

**Action:** Move OCR-specific tools to appropriate subdirectories

**New Structure:**
```
scripts/agent_tools/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── (QMF core tools)
├── compliance/
│   ├── __init__.py
│   └── (QMF compliance tools)
├── documentation/
│   ├── __init__.py
│   ├── auto_generate_index.py (QMF version)
│   ├── validate_manifest.py (QMF version)
│   └── (other QMF doc tools)
├── maintenance/
│   ├── __init__.py
│   └── (QMF maintenance tools)
├── ocr/
│   ├── __init__.py
│   ├── cleanup_remaining_checkpoints.py
│   ├── generate_checkpoint_configs.py
│   ├── next_run_proposer.py
│   ├── remove_low_hmean_checkpoints.py
│   ├── remove_low_step_checkpoints.py
│   ├── verify_best_checkpoints.py
│   ├── summarize_run.py
│   └── (other OCR-specific tools)
└── utilities/
    ├── __init__.py
    ├── get_context.py (QMF version)
    ├── quick_fix_log.py (QMF version)
    ├── context_log.py (OCR-specific)
    └── (other utilities)
```

### Step 2: Adopt QMF Tools

**Priority 1 (High Value, Low Risk):**
- Core tools (artifact_workflow, context_bundle, discover)
- Documentation tools (check_freshness, validate_links, validate_templates, etc.)
- Maintenance tools (all)
- Utilities (clean_logs, view_logs, generate_tool_catalog)

**Priority 2 (Medium Value, Medium Risk):**
- Compliance tools (documentation_quality_monitor, validate_artifacts, monitor_artifacts, fix_artifacts)
- Advanced compliance tools (evaluate case-by-case)

**Priority 3 (Low Priority, Evaluate):**
- Browser/puppeteer tools (if needed)
- Tracking system (if useful)
- Advanced compliance dashboard/trending

### Step 3: Update Imports

**Action:** Update all imports to reflect new structure

**Pattern:**
- Old: `from scripts.agent_tools.get_context import ...`
- New: `from scripts.agent_tools.utilities.get_context import ...`

### Step 4: Update References

**Action:** Update all references in:
- Documentation
- Makefiles
- Scripts
- Protocols

---

## Implementation Plan

### Phase 2.1: Reorganize Structure
1. Create subdirectories: `core/`, `compliance/`, `documentation/`, `maintenance/`, `ocr/`, `utilities/`
2. Move OCR-specific tools to `ocr/`
3. Create `__init__.py` files

### Phase 2.2: Adopt Priority 1 Tools
1. Copy QMF core tools
2. Copy QMF documentation tools (replace existing)
3. Copy QMF maintenance tools
4. Copy QMF utilities (replace existing)
5. Update imports

### Phase 2.3: Adopt Priority 2 Tools
1. Copy QMF compliance tools
2. Test integration
3. Update documentation

### Phase 2.4: Update References
1. Update all imports
2. Update documentation references
3. Update Makefile commands
4. Update protocol references

### Phase 2.5: Testing
1. Test all tools
2. Verify imports work
3. Test tool discovery
4. Update tool catalog

---

## Risk Assessment

### Low Risk
- Adopting QMF core tools (new functionality)
- Adopting QMF maintenance tools (new functionality)
- Adopting QMF utilities (replacements)

### Medium Risk
- Reorganizing structure (imports need updating)
- Replacing existing tools (need to verify compatibility)
- Adopting compliance tools (may need configuration)

### High Risk
- None identified

---

## Next Steps

1. ✅ Complete tool comparison (this document)
2. ⏳ Reorganize structure
3. ⏳ Adopt Priority 1 tools
4. ⏳ Update imports and references
5. ⏳ Test integration
6. ⏳ Update documentation

---

**Last Updated:** 2025-01-XX
**Status:** Ready for Implementation

