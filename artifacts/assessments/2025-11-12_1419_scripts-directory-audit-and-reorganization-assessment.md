---
title: "Scripts Directory Audit and Reorganization Assessment"
author: "ai-agent"
date: "2025-11-12"
timestamp: "2025-11-12 14:19 KST"
status: "draft"
tags: ["scripts", "organization", "cleanup", "discoverability", "ai-tools"]
---

## Progress Tracker
*(Required for iterative assessments, debugging sessions, or incremental work)*

- **STATUS:** Not Started / In Progress / Completed
- **CURRENT STEP:** [Current phase or task being worked on]
- **LAST COMPLETED TASK:** [Description of last completed task]
- **NEXT TASK:** [Description of the immediate next task]

### Assessment Checklist
- [ ] Initial assessment complete
- [ ] Analysis phase complete
- [ ] Recommendations documented
- [ ] Review and validation complete

---

## 1. Summary

This assessment audits the `scripts/` directory for redundancy, obsolete scripts, organizational issues, and discoverability problems following a branch merge. The audit identifies significant structural issues that prevent scripts from being useful to AI agents, who are the primary consumers of these tools.

**Key Findings:**
- **Root-level script pollution**: 7+ scripts at root level that should be organized into subdirectories
- **Misplaced scripts in agent_tools/**: 15+ scripts directly in `agent_tools/` root that belong in subdirectories
- **Duplicate scripts**: Multiple instances of the same functionality (e.g., `process_manager.py`, `validate_metadata.py`, `generate_diagrams.py`)
- **Temporary/debug scripts**: `temp/` directory with 15+ files that may be obsolete
- **Legacy directories**: `migration_refactoring/`, `seroost/` may contain completed work
- **Discoverability gaps**: Many scripts not integrated into discovery mechanisms
- **Architectural issues**: Scripts not organized for AI-oriented workflows

## 2. Assessment

### 2.1 Root-Level Script Issues

**Problem**: Multiple scripts at `scripts/` root level that should be organized:

1. **`validate_metadata.py`** - Duplicates `agent_tools/documentation/validate_metadata.py`
   - Root version: 185 lines, validates checkpoint metadata
   - Agent tools version: 206 lines, same functionality with bootstrap loading
   - **Action**: Remove root version, keep agent_tools version

2. **`validate_coordinate_consistency.py`** - Should be in `validation/checkpoints/`
   - Currently at root, validates coordinate consistency for checkpoints
   - **Action**: Move to `validation/checkpoints/`

3. **`process_manager.py`** - Duplicates `utilities/process_manager.py`
   - Both files are identical (406 lines each)
   - Referenced in Makefile and documentation
   - **Action**: Keep root version for backward compatibility, remove `utilities/` version, or create shim

4. **`generate_diagrams.py`** - Duplicates `documentation/generate_diagrams.py`
   - Both files are identical (615 lines each)
   - Referenced in Makefile and CI scripts
   - **Action**: Keep root version for backward compatibility, remove `documentation/` version, or create shim

5. **`generate_checkpoint_metadata.py`** - Should be in `checkpoints/` or `agent_tools/ocr/`
   - Currently at root, generates checkpoint metadata
   - **Action**: Move to `checkpoints/` or integrate into `agent_tools/ocr/`

6. **`convert_legacy_checkpoints.py`** - Should be in `checkpoints/` or `migration_refactoring/`
   - Currently at root, converts legacy checkpoint formats
   - **Action**: Move to `checkpoints/` or `migration_refactoring/`

7. **`preprocess_data.py`** - Compatibility shim (correctly placed)
   - Delegates to `scripts/data/preprocess.py`
   - **Action**: Keep as-is (backward compatibility shim)

### 2.2 Agent Tools Root Directory Issues

**Problem**: 15+ scripts directly in `agent_tools/` root that should be in subdirectories:

**OCR-related scripts** (should be in `agent_tools/ocr/`):
- `cleanup_remaining_checkpoints.py` - Already exists in `ocr/` directory
- `generate_checkpoint_configs.py` - Duplicates `ocr/generate_checkpoint_configs.py`
- `next_run_proposer.py` - Already exists in `ocr/` directory
- `remove_low_hmean_checkpoints.py` - Already exists in `ocr/` directory
- `remove_low_step_checkpoints.py` - Already exists in `ocr/` directory
- `summarize_run.py` - Already exists in `ocr/` directory
- `verify_best_checkpoints.py` - Already exists in `ocr/` directory

**Utility scripts** (should be in `agent_tools/utilities/`):
- `context_log.py` - Already exists in `utilities/` directory
- `delegate_to_qwen.py` - Already exists in `utilities/` directory
- `quick_fix_log.py` - Already exists in `utilities/` directory
- `run_agent_demo.py` - Already exists in `utilities/` directory
- `strip_doc_markers.py` - Already exists in `utilities/` directory
- `test_validation.py` - Already exists in `utilities/` directory

**Documentation/maintenance scripts**:
- `generate_handbook_index.py` - Should be in `documentation/` or `maintenance/`
- `validate_manifest.py` - Should be in `documentation/`
- `declutter_root.py` - Should be in `maintenance/`

**Action Required**: Remove duplicate scripts from `agent_tools/` root, keep only the properly organized versions in subdirectories.

### 2.3 Temporary and Debug Scripts

**Problem**: `scripts/temp/` contains 15+ files that may be obsolete:

- `debug_app_startup.py`
- `test_app_minimal_v2.py`
- `test_comparison_integration.py`
- `test_inference_minimal.py`
- `test_streamlit_minimal.py`
- `test_streamlit_widgets.py`
- `test_streamlit_with_imports.py`
- Multiple markdown files (DEBUG_INSTRUCTIONS.md, DEBUGGING_TOOLKIT.md, etc.)

**Assessment**: These appear to be debugging artifacts from previous sessions. Need to:
1. Review each file for current relevance
2. Archive or delete obsolete files
3. Move any still-useful scripts to appropriate directories (`troubleshooting/`, `demos/`, etc.)

**`scripts/debug/`**: Empty directory - candidate for removal.

### 2.4 Legacy Directories

**`migration_refactoring/`**:
- `migrate_checkpoint_names.py` - May be obsolete if migration completed
- `refactor_ocr_pl.py` - May be obsolete if refactoring completed
- **Action**: Verify if work is complete, archive if done

**`seroost/`**:
- Contains semantic search indexing tools
- May be project-specific or completed work
- **Action**: Verify current relevance, archive if obsolete

### 2.5 Discoverability Issues

**Current Discovery Mechanisms**:
- `scripts/agent_tools/core/discover.py` - Lists tools by category
- `scripts/agent_tools/utilities/generate_tool_catalog.py` - Generates tool catalog
- `scripts/agent_tools/__main__.py` - Unified CLI (limited commands)

**Problems**:
1. Root-level scripts not discoverable through agent_tools discovery
2. Many scripts in `agent_tools/` root not categorized
3. No comprehensive index of all scripts
4. Scripts in `validation/`, `performance/`, `monitoring/` not integrated into discovery
5. Temporary scripts in `temp/` not filtered out

**Impact**: AI agents cannot easily discover available tools, leading to:
- Duplicate script creation
- Underutilization of existing tools
- Inconsistent tool usage

### 2.6 Architectural Issues

**Current Structure Problems**:
1. **Mixed organization patterns**: Some scripts organized by function (`data/`, `performance/`), others by audience (`agent_tools/`)
2. **No clear entry point**: Multiple ways to run scripts (direct Python, Makefile, unified CLI)
3. **Inconsistent bootstrap loading**: Some scripts use `_bootstrap.py`, others use manual path manipulation
4. **No script metadata**: Scripts lack standardized metadata (description, usage, dependencies)
5. **No deprecation mechanism**: No way to mark scripts as deprecated or obsolete

**AI-Oriented Requirements**:
- Scripts should be discoverable through semantic search
- Scripts should have clear, standardized documentation
- Scripts should be organized by use case, not just by function
- Scripts should have machine-readable metadata
- Scripts should integrate with AI agent workflows

## 3. Recommendations

### 3.1 Immediate Actions (High Priority)

1. **Remove duplicate scripts from `agent_tools/` root**:
   - Delete all scripts that exist in subdirectories
   - Keep only `__init__.py`, `__main__.py`, and `README.md` at root

2. **Consolidate root-level duplicates**:
   - Remove `scripts/validate_metadata.py` (keep agent_tools version)
   - Remove `scripts/utilities/process_manager.py` (keep root version for backward compatibility)
   - Remove `scripts/documentation/generate_diagrams.py` (keep root version for backward compatibility)

3. **Organize root-level scripts**:
   - Move `validate_coordinate_consistency.py` → `validation/checkpoints/`
   - Move `generate_checkpoint_metadata.py` → `checkpoints/`
   - Move `convert_legacy_checkpoints.py` → `checkpoints/` or `migration_refactoring/`

4. **Audit and clean `temp/` directory**:
   - Review each file for relevance
   - Archive useful scripts to appropriate directories
   - Delete obsolete files
   - Remove empty `debug/` directory

### 3.2 Medium-Term Improvements

1. **Enhance discovery mechanism**:
   - Extend `discover.py` to scan all script directories, not just `agent_tools/`
   - Add script metadata extraction (docstrings, usage examples)
   - Create comprehensive script index
   - Filter out temporary/debug scripts

2. **Standardize script structure**:
   - Require all scripts to use `_bootstrap.py` for path setup
   - Standardize docstring format with usage examples
   - Add script metadata (category, purpose, dependencies)

3. **Create script registry**:
   - Generate `scripts/INDEX.md` with all scripts categorized
   - Include usage examples and dependencies
   - Mark deprecated/obsolete scripts

4. **Integrate with AI workflows**:
   - Add semantic search support for scripts
   - Create script recommendation system based on task
   - Integrate scripts with agent toolbelt

### 3.3 Long-Term Architectural Changes

1. **Reorganize by use case**:
   - Create `scripts/ai/` for AI-specific tools
   - Create `scripts/development/` for development tools
   - Create `scripts/operations/` for operational tools
   - Maintain backward compatibility with current structure

2. **Unified script interface**:
   - Extend `scripts/agent_tools/__main__.py` to cover all scripts
   - Add script discovery and execution through unified CLI
   - Support script metadata queries

3. **Script lifecycle management**:
   - Add deprecation markers to obsolete scripts
   - Create migration path for deprecated scripts
   - Archive completed migration/refactoring scripts

4. **AI-oriented enhancements**:
   - Add script descriptions in machine-readable format (JSON/YAML)
   - Create script dependency graph
   - Generate script usage documentation automatically
   - Integrate with agent context bundles

### 3.4 Verification Steps

After cleanup:
1. Run `python scripts/agent_tools/core/discover.py --list` to verify discovery works
2. Check all Makefile targets still work
3. Verify CI scripts still function
4. Test that all documented scripts are accessible
5. Confirm no broken imports or references

## 4. Progress Tracker

- **STATUS**: Assessment Complete
- **CURRENT STEP**: Review and validation
- **LAST COMPLETED TASK**: Comprehensive audit of scripts directory
- **NEXT TASK**: Begin cleanup based on recommendations

### Assessment Checklist
- [x] Initial assessment complete
- [x] Analysis phase complete
- [x] Recommendations documented
- [ ] Review and validation complete
- [ ] Cleanup execution (future session)
