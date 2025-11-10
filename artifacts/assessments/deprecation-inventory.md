---
title: "Deprecation and Legacy Files Inventory"
author: "ai-agent"
date: "2025-11-09"
status: "draft"
tags: ["deprecation", "legacy", "cleanup", "documentation", "assessment"]
---

# Deprecation and Legacy Files Inventory

**Status:** Draft
**Date:** 2025-11-09
**Priority:** High
**Impact:** Critical - Eliminates confusion from duplicates and outdated files

## Executive Summary

This inventory identifies all deprecated, legacy, and duplicate files causing confusion in the codebase. The goal is to:
1. **Eliminate duplicates** - Remove conflicting information
2. **Deprecate outdated files** - Mark clearly and redirect to current sources
3. **Consolidate protocols** - Single source of truth for each protocol
4. **Clean up legacy code** - Remove or archive unused code

## Critical Issues

### 1. Duplicate Protocol Locations

**Problem:** Same protocols exist in multiple locations with conflicting information

| Protocol | Outdated Location | Current Location | Status | Action |
|----------|-------------------|------------------|--------|--------|
| Artifact Management | `docs/agents/docs_governance/01_artifact_management_protocol.md` | `docs/agents/protocols/governance.md` | ❌ Conflicting | **DEPRECATE** outdated |
| Implementation Plans | `docs/agents/docs_governance/02_implementation_plan_protocol.md` | `docs/agents/protocols/governance.md` | ❌ Conflicting | **DEPRECATE** outdated |
| Blueprint Template | `docs/agents/docs_governance/03_blueprint_protocol_template.md` | `docs/maintainers/protocols/governance/03_blueprint_protocol_template.md` | ⚠️ Duplicate | **CONSOLIDATE** |
| Bug Fix Protocol | `docs/agents/docs_governance/04_bug_fix_protocol.md` | `docs/agents/protocols/governance.md` | ❌ Conflicting | **DEPRECATE** outdated |

**Impact:** Agents may read outdated protocols and follow wrong standards.

**Solution:**
- Mark `docs/agents/docs_governance/` as deprecated
- Add clear deprecation notices with redirects
- Eventually remove or archive

### 2. Deprecated Documentation Directory

**Location:** `docs/_deprecated/`

**Contents:**
- `ai_handbook/` - Entire deprecated handbook (migrated to `docs/agents/` and `docs/maintainers/`)
- `documentation-refactor_plan.md` - Old refactor plan
- `framework_placement_suggestions.md` - Old suggestions
- `proposed structure.md` - Old structure proposal
- `qmf_adoption_assessment.md` - Old assessment

**Status:** ⚠️ Deprecated but still exists

**Action:**
- Keep for historical reference
- Add clear README explaining what's deprecated
- Consider archiving to separate archive directory

### 3. Legacy Code Files

**Problem:** Legacy code files still exist but are deprecated

| File | Status | Replacement | Action |
|------|--------|-------------|--------|
| `scripts/convert_legacy_checkpoints.py` | ⚠️ Legacy | V2 checkpoint system | **DEPRECATE** or remove |
| `ui/apps/inference/services/checkpoint_catalog.py` | ⚠️ Legacy V1 | `ui/apps/inference/services/checkpoint/` | **DEPRECATE** (already marked) |
| `scripts/agent_tools/core/artifact_templates.py` | ⚠️ No longer used | `agent_qms/toolbelt/` | **DEPRECATE** or remove |

**Impact:** Confusion about which code to use.

**Solution:**
- Add deprecation notices
- Update references
- Remove after migration period

### 4. Outdated Migration Documentation

**Location:** `docs/maintainers/migration/`

**Contents:**
- `ENTRY_POINTS_ANALYSIS.md` - Old analysis
- `ENTRY_POINTS_IMPLEMENTATION.md` - Old implementation
- `ENTRY_POINTS_SUMMARY.md` - Old summary
- `INDEX_STRATEGY.md` - Old strategy
- `MIGRATION_SUMMARY.md` - Old migration summary
- `documentation_audit.md` - Old audit
- `qmf_integration_progress.md` - Old progress
- `AGENTQMS_RENAME.md` - Old rename doc

**Status:** ⚠️ Historical migration docs

**Action:**
- Keep for historical reference
- Add README explaining these are historical
- Consider archiving

## Detailed Inventory

### Deprecated Directories

#### 1. `docs/_deprecated/`

**Purpose:** Contains deprecated documentation

**Contents:**
```
docs/_deprecated/
├── ai_handbook/                    # Entire deprecated handbook
│   ├── 01_onboarding/             # Migrated to docs/maintainers/onboarding/
│   ├── 02_protocols/              # Migrated to docs/agents/protocols/
│   ├── 03_references/             # Migrated to docs/agents/references/
│   ├── 04_experiments/            # Migrated to docs/maintainers/experiments/
│   ├── 05_changelog/              # Migrated to docs/maintainers/changelog/
│   ├── 06_concepts/               # Deprecated
│   ├── 07_planning/               # Migrated to docs/maintainers/planning/
│   ├── 08_planning/               # Migrated to docs/maintainers/planning/
│   └── README_DEPRECATED.md       # Deprecation notice
├── documentation-refactor_plan.md # Old refactor plan
├── framework_placement_suggestions.md # Old suggestions
├── proposed structure.md          # Old structure
└── qmf_adoption_assessment.md     # Old assessment
```

**Status:** ⚠️ Deprecated - Keep for historical reference

**Action:**
- Add README explaining what's deprecated
- Consider archiving to separate archive directory

#### 2. `docs/agents/docs_governance/`

**Purpose:** Outdated governance protocols (duplicates of `docs/agents/protocols/`)

**Contents:**
```
docs/agents/docs_governance/
├── 01_artifact_management_protocol.md    # ⚠️ DEPRECATED - Use docs/agents/protocols/governance.md
├── 02_implementation_plan_protocol.md    # ⚠️ DEPRECATED - Use docs/agents/protocols/governance.md
├── 03_blueprint_protocol_template.md    # ⚠️ DUPLICATE - Use docs/maintainers/protocols/governance/
└── 04_bug_fix_protocol.md              # ⚠️ DEPRECATED - Use docs/agents/protocols/governance.md
```

**Status:** ❌ **CRITICAL** - Causes confusion, should be deprecated

**Action:**
- Add deprecation notices to all files
- Add redirects to current locations
- Update `docs/agents/index.md` to mark as deprecated
- Eventually remove or archive

### Legacy Code Files

#### 1. `scripts/convert_legacy_checkpoints.py`

**Status:** ⚠️ Legacy checkpoint conversion tool

**Replacement:** V2 checkpoint system

**Action:**
- Add deprecation notice
- Document replacement
- Remove after migration period

#### 2. `ui/apps/inference/services/checkpoint_catalog.py`

**Status:** ⚠️ Legacy V1 catalog (already marked as deprecated in code)

**Replacement:** `ui/apps/inference/services/checkpoint/` (V2)

**Action:**
- Already has deprecation notice
- Remove after migration period

#### 3. `scripts/agent_tools/core/artifact_templates.py`

**Status:** ⚠️ No longer used by `artifact_workflow.py`

**Replacement:** `agent_qms/toolbelt/` (AgentQMS toolbelt)

**Action:**
- Add deprecation notice
- Document that `artifact_workflow.py` now uses AgentQMS
- Remove after confirming no other usage

### Outdated Migration Documentation

#### `docs/maintainers/migration/`

**Purpose:** Historical migration documentation

**Contents:**
- `ENTRY_POINTS_ANALYSIS.md` - Old analysis
- `ENTRY_POINTS_IMPLEMENTATION.md` - Old implementation
- `ENTRY_POINTS_SUMMARY.md` - Old summary
- `INDEX_STRATEGY.md` - Old strategy
- `MIGRATION_SUMMARY.md` - Old migration summary
- `documentation_audit.md` - Old audit
- `qmf_integration_progress.md` - Old progress
- `AGENTQMS_RENAME.md` - Old rename doc

**Status:** ⚠️ Historical - Keep for reference

**Action:**
- Add README explaining these are historical
- Consider archiving to separate archive directory

## Recommended Actions

### Phase 1: Immediate (High Priority)

1. **Deprecate `docs/agents/docs_governance/`**
   - Add deprecation notice to all files
   - Add redirects to current locations
   - Update `docs/agents/index.md` to mark as deprecated
   - Update `docs/agents/system.md` to clarify current location

2. **Add README to `docs/_deprecated/`**
   - Explain what's deprecated
   - Point to current locations
   - Explain why it's kept (historical reference)

3. **Add deprecation notices to legacy code**
   - `scripts/convert_legacy_checkpoints.py`
   - `scripts/agent_tools/core/artifact_templates.py`
   - `ui/apps/inference/services/checkpoint_catalog.py` (already has notice)

### Phase 2: Short-term (Medium Priority)

4. **Consolidate Blueprint Template**
   - Check if `docs/agents/docs_governance/03_blueprint_protocol_template.md` is duplicate
   - If duplicate, remove from `docs/agents/docs_governance/`
   - Keep only in `docs/maintainers/protocols/governance/`

5. **Archive migration documentation**
   - Move `docs/maintainers/migration/` to archive
   - Add README explaining historical nature

### Phase 3: Long-term (Low Priority)

6. **Remove deprecated files**
   - After migration period, remove deprecated files
   - Keep only in archive if needed for historical reference

7. **Clean up `docs/_deprecated/`**
   - Archive to separate archive directory
   - Keep only essential historical reference

## Deprecation Notice Template

For all deprecated files, add this notice at the top:

```markdown
---
title: "[Original Title]"
status: "deprecated"
deprecated_date: "2025-11-09"
replacement: "[Link to current location]"
---

# ⚠️ DEPRECATED

**This file is deprecated as of 2025-11-09.**

**Current Location:** [Link to current file/location]

**Reason:** [Reason for deprecation]

**Action Required:** Use [current location] instead.

---

[Original content below]
```

## Success Criteria

1. ✅ **No duplicate protocols**
   - Single source of truth for each protocol
   - Clear deprecation notices on outdated versions

2. ✅ **Clear deprecation notices**
   - All deprecated files have notices
   - All notices point to current locations

3. ✅ **Updated references**
   - All references point to current locations
   - No references to deprecated files

4. ✅ **Documented legacy code**
   - All legacy code has deprecation notices
   - Clear migration path documented

5. ✅ **Organized archives**
   - Historical docs clearly marked
   - Easy to find current vs historical

## References

- Current protocols: `docs/agents/protocols/`
- Single source of truth: `docs/agents/system.md`
- AgentQMS: `agent_qms/`
- Artifact workflow: `scripts/agent_tools/core/artifact_workflow.py`

---

*This inventory identifies all deprecated and legacy files to eliminate confusion from duplicates and outdated information.*




