# AgentQMS Framework Bug Fixes - AI Implementation Guide

## Changelog Format Guidelines
- **Format**: `[YYYY-MM-DD HH:MM] - Brief description (max 80 chars)`
- **Placement**: Add new entries at the very top, below this guidelines section
- **Conciseness**: Keep entries ultra-concise - focus on what changed, not why
- **Categories**: Group related changes under appropriate section headers

## Centralized IDE Configuration (2025-12-11)

### [2025-12-11 14:35] - Added Centralized IDE Configuration Generator
**Scope**: Agent settings management
**Change**: Created `AgentQMS/agent_tools/utilities/generate_ide_configs.py` and `make ide-config` targeting Antigravity, Cursor, Claude, and Copilot
**Impact**: Solves settings divergence by generating all IDE-specific configuration files from a single source of truth
**Details**: Logic consolidation of 4 different IDE config formats into one generator script

## AgentQMS Manager Dashboard Integration Planning (2025-12-08)

### [2025-12-08 02:45] - Recovered Phase 1-2 Dashboard Documentation
**Scope**: AgentQMS Manager Dashboard integration planning
**Change**: Extracted and recovered 11 docs from Phase 1-2 archive to `docs/agentqms-manager-dashboard/`
**Files Recovered**: 3 session handovers (2024-05-22/23), 8 technical docs (442+ lines)
**Critical Finding**: Backend bridge (`AgentQMS/agent_tools/bridge/`) not implemented despite session handover claims
**Artifacts Created**: Assessment `2025-12-08_0229_assessment-dashboard-phase1-phase2-recovery.md`, Plan `2025-12-08_0231_implementation_plan_dashboard-integration-testing.md`
**Status**: Documentation recovery complete, backend bridge missing, repository sanity check pending
**Next Steps**: Implement bridge from scratch, create feature branch, build artifact management API

## Toolkit Deprecation Completion (2025-12-06)

### [2025-12-06 20:24] - Completed AgentQMS Toolkit Migration to agent_tools
**Scope**: AgentQMS framework architecture
**Change**: Fully migrated all toolkit modules to agent_tools, archived toolkit directory
**Modules Migrated**: audit (3), core (1), documentation (3), utilities (3), compliance (1), tracking (3)
**Impact**: Clean architecture with single canonical location for all tools
**Files**: All modules in `AgentQMS/agent_tools/`, CLI tools updated
**Archive**: `AgentQMS/toolkit` → `AgentQMS/.old_toolkit`
**Details**: Implementation plan `2025-12-06_2009_implementation_plan_toolkit-deprecation-completion.md`

## Artifact Naming Convention Standardization (2025-11-29)

### [2025-11-29 17:30] - Terminology Standardization Complete
**Scope**: Validation system, documentation, governance
**Change**: Standardized all references to use "ARTIFACT_TYPE" terminology
**Impact**: Eliminates confusion between "prefix", "type", "document_type" terms
**Files**: `validate_artifacts.py` (agent_tools & toolkit), `system.md`, governance docs
**Details**: See audit report `2025-11-29_1642_assessment-artifact-naming-terminology-conflicts.md`

### [2025-11-29 17:20] - Added Audit Artifact Type
**File**: `AgentQMS/agent_tools/compliance/validate_artifacts.py`, `AgentQMS/toolkit/compliance/validate_artifacts.py`
**Addition**: `"audit-": "audits/"` artifact type registered
**Directory**: Created `docs/artifacts/audits/` for audit artifacts
**Purpose**: Separate audits from assessments for better categorization

### [2025-11-29 17:25] - Enforced docs/artifacts/ Location
**File**: `AgentQMS/agent_tools/compliance/validate_artifacts.py`
**Addition**: `validate_artifacts_root()` method added
**Enforcement**: Root-level `/artifacts/` directory now forbidden
**Requirement**: All artifacts must be in `docs/artifacts/` hierarchy

### [2025-11-29 17:28] - Fixed Directory Validation Bug
**File**: `AgentQMS/agent_tools/compliance/validate_artifacts.py`, `AgentQMS/toolkit/compliance/validate_artifacts.py`
**Issue**: Directory validation checked `filename.startswith(artifact_type)` (prefix-first) but format is timestamp-first
**Fix**: Extract artifact type from timestamp-first format using regex, then match against registry
**Impact**: Directory placement validation now works correctly for all valid filenames

## Validation System Bugs

### [2025-11-28 19:25] - Added 'code_quality' category to artifact validation
**File**: `AgentQMS/agent_tools/compliance/validate_artifacts.py`, `AgentQMS/toolkit/compliance/validate_artifacts.py`
**Issue**: 'code_quality' category not recognized by validators
**Fix**: Added "code_quality" to _BUILTIN_CATEGORIES and valid_categories lists
**Implementation**: Distinguish code quality assessments from development/compliance categories

### Bug: Incorrect Naming Regex Pattern
**File**: `.qwen/manual_validate.sh`
**Issue**: Regex expected `[TYPE]` uppercase instead of `[PREFIX]` format
**Fix**: Changed from `^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4}_[A-Z]+_.+\.md$` to `^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4}_(.+)\.md$`
**Implementation**: Update validation regex to match AgentQMS prefix-based naming

### Bug: Artifacts Path Configuration Mismatch
**File**: `.agentqms/settings.yaml`
**Issue**: Path pointed to wrong artifacts directory
**Fix**: Set `paths.artifacts: docs/artifacts`
**Implementation**: Ensure artifacts path matches actual directory structure

### Bug: Missing Frontmatter Opening Marker
**File**: Artifact files
**Issue**: Frontmatter fields present but missing opening `---`
**Fix**: Add `---` at start of YAML frontmatter
**Implementation**: Ensure all frontmatter starts with `---` marker

### Enhancement: Bug Report Template Clarification
**File**: `AgentQMS/toolkit/core/artifact_templates.py`
**Issue**: Bug report template lacked clear distinction between required initial fields and optional resolution fields
**Enhancement**: Added severity field to frontmatter, HTML comments to distinguish REQUIRED vs OPTIONAL sections, clarified Priority vs Severity
**Implementation**: Updated bug_report template with clear section indicators and frontmatter severity field

## Qwen CLI Integration Bugs

### Bug: Checkpointing Git Detection Failure
**File**: `.qwen/settings.json`
**Issue**: Qwen CLI failed with "Checkpointing is enabled, but Git is not installed"
**Fix**: Set `"general": {"checkpointing": {"enabled": false}}`
**Implementation**: Add checkpointing disabled setting to prevent Git dependency issues

### Bug: Approval Mode Syntax Error
**File**: `.qwen/run.sh`
**Issue**: `--yolo` flag conflicted with `--approval-mode yolo`
**Fix**: Use `--approval-mode yolo` consistently in all scripts
**Implementation**: Update Qwen command syntax to use correct approval mode flags

### Bug: Memory Leak - Abort Signal Listeners
**File**: Qwen CLI operations
**Issue**: MaxListenersExceededWarning (11 abort listeners > 10 max)
**Fix**: Avoid running multiple simultaneous Qwen processes
**Implementation**: Add process management to prevent concurrent Qwen instances

## Documentation Structure Bugs

### Bug: Loose Docs in Project Root
**File**: Project root directory
**Issue**: DOCS_INDEX.md, HANDOVER.md violated "no loose docs" rule
**Fix**: Move to `docs/` and `docs/artifacts/` respectively
**Implementation**: Enforce root directory policy (only README.md, CHANGELOG.md allowed)

### Bug: Inconsistent Artifact Naming
**File**: Artifact files
**Issue**: Files used various naming patterns instead of timestamped format
**Fix**: Rename to `YYYY-MM-DD_HHMM_[PREFIX]descriptive-name.md`
**Implementation**: Standardize naming convention across all artifacts

### Bug: Wrong Directory Structure
**File**: Artifact organization
**Issue**: Artifacts not organized by type in subdirectories
**Fix**: Create `docs/artifacts/{assessments,bug_reports,plans,etc}/` structure
**Implementation**: Implement type-based directory organization

## Configuration Bugs

### Bug: Inconsistent Path References
**Files**: Multiple config files
**Issue**: Scripts and configs referenced different artifact paths
**Fix**: Standardize all references to `docs/artifacts/`
**Implementation**: Audit and update all path references for consistency

### Bug: Missing Workspace Exclusions
**File**: `.qwen/settings.json`
**Issue**: Framework files could be accidentally modified
**Fix**: Add exclude patterns for `AgentQMS/`, `.agentqms/`, etc.
**Implementation**: Add comprehensive workspace exclusion patterns

### Bug: Incorrect Tool Permissions
**File**: `.qwen/settings.json`
**Issue**: Tools had insufficient permissions for autonomous operation
**Fix**: Set approval mode to "auto-edit" and add allowed tools list
**Implementation**: Configure appropriate tool permissions for AI operations

## Script Implementation Bugs

### Bug: Run Script Command Syntax
**File**: `.qwen/run.sh`
**Issue**: Qwen commands used incorrect flag syntax
**Fix**: Replace `--yolo` with `--approval-mode yolo`
**Implementation**: Update all Qwen command invocations with correct syntax

### Bug: Validation Script Path Hardcoding
**File**: `.qwen/manual_validate.sh`
**Issue**: Script hardcoded wrong artifacts directory path
**Fix**: Update `ARTIFACTS_DIR` variable to correct path
**Implementation**: Make artifact paths configurable or dynamically resolved

### Bug: Frontmatter Generation Missing Opening Marker
**File**: Artifact creation scripts
**Issue**: Generated frontmatter missing opening `---`
**Fix**: Ensure YAML frontmatter always starts with `---`
**Implementation**: Update frontmatter generation templates

### Bug: Missing sys Import in Artifact Workflow
**File**: `AgentQMS/toolkit/core/artifact_workflow.py`
**Issue**: `sys.executable` used without importing `sys`, causing "name 'sys' is not defined" error in index updater
**Fix**: Add `import sys` to imports
**Implementation**: Import sys module at the top of the file
