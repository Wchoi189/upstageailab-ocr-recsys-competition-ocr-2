# Project Compass - Nuclear Cleanup Requirements

**Status**: Requirements Gathering Complete
**Approach**: Nuclear refactor, zero legacy support
**Risk**: HIGH - Breaking changes, no backward compatibility

---

## Executive Summary

### Problem Statement
Project Compass currently has **toxic dual architecture**:
1. **MCP Server** (`mcp_server.py`) with `compass_meta_pulse` tools
2. **CLI** (`cli.py`) with `compass pulse-*` commands
3. **Skills** (just created) wrapping MCP tools

**Result**: Confusing, redundant, unclear workflow.

### Solution
**Single architecture: Skills → Python Core (no MCP layer)**

Remove MCP server entirely, keep CLI for direct invocation, skills as primary interface.

---

## Architecture Analysis

### Current State (Toxic Dual Architecture)

```
User
  ↓
  ├─→ Skills (/compass-start)
  │      ↓
  │   MCP Tools (compass_meta_pulse)
  │      ↓
  │   Python Core (PulseManager)
  │
  ├─→ CLI (uv run compass pulse-init)
  │      ↓
  │   Python Core (PulseManager)
  │
  └─→ Direct MCP (claude calls mcp__unified__compass_meta_pulse)
         ↓
      Python Core (PulseManager)
```

**Problems**:
- 3 entry points to same functionality
- Skills call MCP tools (unnecessary indirection)
- MCP and CLI duplicate implementation
- Confusing for users and AI agents
- More code to maintain

### Target State (Single Architecture)

```
User
  ↓
  ├─→ Skills (/compass-start) [PRIMARY]
  │      ↓
  │   Python Core (PulseManager)
  │
  └─→ CLI (uv run compass pulse-init) [FALLBACK]
         ↓
      Python Core (PulseManager)
```

**Benefits**:
- Single, clear interface (Skills)
- CLI for scripts/automation only
- No MCP indirection
- Less code, less confusion
- Easier to maintain

---

## Detailed Findings

### 1. Dual Directory Structure 🔴 CRITICAL

**Issue**: Two `pulse_staging` directories exist

```
/workspaces/dev_tools/project_compass/
├── pulse_staging/                          # DUPLICATE (active?)
│   ├── archive/
│   └── artifacts/
├── project_compass/
│   └── pulse_staging/                      # EXPECTED by paths.py
│       └── artifacts/
```

**Investigation Needed**:
- Which is authoritative? (`core.py` expects `project_compass/pulse_staging`)
- What's in top-level `pulse_staging/artifacts/`?
- Is `pulse_staging/archive/` used? (not mentioned in code)

**Decision Required**:
- **Option A**: Delete top-level `pulse_staging/`, use only `project_compass/pulse_staging/`
- **Option B**: Move everything to top-level, update `core.py` paths

**Recommendation**: Option A (follow existing code structure)

### 2. Snapshots System ⚠️ REVIEW REQUIRED

**Current Implementation**:
- Directory: `/workspaces/dev_tools/project_compass/snapshots/`
- Function: `create_snapshot()` in `pulse_exporter.py`
- Usage: Called by `pulse-checkpoint` MCP tool
- Purpose: Mid-work checkpoints WITHOUT clearing pulse

**Difference from Export**:
- `pulse-export`: Archives pulse, clears active pulse, moves to `history/`
- `create_snapshot`: Saves checkpoint, pulse remains active, saves to `snapshots/`

**Questions**:
1. Is mid-work checkpointing actually needed?
2. Does it add value over git commits?
3. Used in practice or theoretical feature?

**Discovery**:
```bash
ls snapshots/
# Output: recognition-audit-post-refactor/
#   - 20260202_201246_test_script_snapshot/
#   - context-bundling-audit/
```

**Has been used** - contains real data.

**Decision Options**:
- **Option A**: Keep snapshots (provides mid-pulse checkpointing value)
- **Option B**: Remove snapshots (git commits sufficient, simplify system)
- **Option C**: Make snapshot creation explicit in skills (not automatic)

**Recommendation**: Option B (remove) - Git provides version control, pulse export provides archiving. Snapshots add complexity without clear value.

### 3. MCP Server Complete Removal 🔴 CRITICAL

**File to Delete**: `project_compass/mcp_server.py` (519 lines)

**MCP Tools to Remove**:
1. `compass_meta_pulse` (routes to: init, sync, export, status, checkpoint)
2. `compass_meta_spec` (routes to: constitution, specify, plan, tasks)

**MCP Resources to Remove**:
1. `vessel://state`
2. `vessel://rules`
3. `vessel://staging`

**Impact Analysis**:

#### Skills Impact
Current skills call MCP tools:
```python
# compass-start/SKILL.md
mcp__unified__compass_meta_pulse(kind="init", ...)
```

**Required Change**: Update skills to call Python core directly or via CLI:
```python
# Option A: Call CLI
uv run compass pulse-init --id X --obj Y --milestone Z

# Option B: Import Python (if skills can execute Python)
from project_compass.src.core import PulseManager
manager = PulseManager()
manager.init_pulse(...)
```

**Recommendation**: Option A (call CLI via Bash tool from skills)

#### External Dependencies
**Check**: Are MCP tools exposed to other systems?
- Unified MCP server (`mcp__unified__compass_meta_*`)
- Direct MCP calls in user workflows

**Action**: Search for external MCP usage before removal

### 4. Spec-Kit Integration Removal 🟡 MEDIUM

**User Decision**: "Using specify (Github spec-kit) seems like a good idea, but it might be overkill"

**Current Implementation**:
- CLI commands: `spec-constitution`, `spec-specify`, `spec-plan`, `spec-tasks`
- MCP tools: Same 4 operations via `compass_meta_spec`
- Files created: `constitution.md`, `specification.md`, `implementation_plan.md`, `tasks.md`

**Removal Scope**:
1. Remove 4 CLI commands from `cli.py` (lines 329-508)
2. Remove `compass_meta_spec` tool from `mcp_server.py`
3. Remove spec handlers (`handle_spec_*` functions)
4. Remove from router (`route_spec` function)
5. Update AGENTS.md (remove spec-kit section)
6. Update CHANGELOG.md (note removal)

**Artifact Type Impact**:
Current types: `design`, `research`, `walkthrough`, `implementation_plan`, `bug_report`, `audit`

Spec-kit added: `specification`, `requirements`, `architecture`

**Decision**: Keep artifact types? Or remove spec-specific types?

**Recommendation**: Remove spec-specific types, keep core 6 types

### 5. Documentation Overhaul 📝 REQUIRED

#### AGENTS.md Updates

**Current Issues**:
- Documents MCP tools as primary interface
- Shows CLI commands as secondary
- No mention of Skills
- Outdated workflow diagrams
- References removed features

**Required Changes**:
1. **Section 1**: Update "Core Concepts" - Skills are primary interface
2. **Section 3**: Remove MCP Resources section entirely
3. **Section 3**: Replace "CLI Commands" with "Skills Reference"
4. **Section 5**: Update "Pulse Lifecycle" diagram - show Skills → Core
5. **Section 8**: Remove "Spec Kit Integration" section entirely
6. **New Section**: Add "Skills vs CLI" guidance

**New Content Needed**:
```markdown
## Skills Interface (Primary)

Use `/compass-*` skills for all interactive work:
- `/compass-start` - Initialize pulse
- `/compass-status` - Check state
- `/compass-register` - Register artifacts
- `/compass-finish` - Export pulse

## CLI (Automation Only)

Use `compass` CLI for scripts/automation:
- `uv run compass pulse-init` - Script initialization
- `uv run compass pulse-export` - Automated export
```

#### AGENTS.yaml Updates

**Current State**:
```yaml
entry_points:
  compass_help: "uv run compass --help"
  check_env: "uv run compass check-env"
  pulse_init: "uv run compass pulse-init ..."
  # ... more CLI commands
```

**Issues**:
- Documents CLI as entry points
- No mention of Skills
- Outdated consumption estimate

**Required Changes**:
1. Add Skills as primary entry points
2. Keep CLI commands for reference
3. Update consumption token estimate
4. Add deprecation note for direct CLI use

**New Structure**:
```yaml
# PROJECT COMPASS AI-INTERFACE V3.0
# CONSUMPTION: ~40 Tokens (optimized)
# PROTOCOL: Use /compass-* skills. CLI for automation only.

primary_interface:
  skills:
    - "/compass-start - Initialize pulse"
    - "/compass-status - Check state"
    - "/compass-register - Register artifacts"
    - "/compass-finish - Export pulse"
    - "/compass-resume - Load context"

automation_interface:
  cli:
    - "uv run compass pulse-init"
    - "uv run compass pulse-export"
```

#### CHANGELOG.md Entry

**Required Entry**:

```markdown
## [3.0.0] - 2026-02-12 - BREAKING CHANGES

### BREAKING CHANGES
- **MCP Server Removed**: All MCP tools deleted
- **Skills Primary Interface**: `/compass-*` skills are now primary
- **Spec-Kit Removed**: All spec-* commands deleted
- **Single Architecture**: Skills → Python Core only

### Removed
- MCP Server (`mcp_server.py`)
- MCP Tools: `compass_meta_pulse`, `compass_meta_spec`
- MCP Resources: `vessel://state`, `vessel://rules`, `vessel://staging`
- CLI Commands: `spec-constitution`, `spec-specify`, `spec-plan`, `spec-tasks`
- Snapshots system (optional - TBD)
- Router module (`router.py`) - no longer needed

### Added
- Skills system (7 skills total)
  - `/compass-start`, `/compass-status`, `/compass-register`
  - `/compass-finish`, `/compass-resume`, `/compass-help`
  - `/audit-run`
- Skills documentation (`skills/README.md`, `QUICKSTART.md`)

### Changed
- Skills now call CLI directly (not MCP)
- Documentation updated for skills-first workflow
- Simplified architecture (2 layers instead of 3)

### Migration Guide

#### For AI Agents
**Before (v2.x)**:
```python
mcp__unified__compass_meta_pulse(kind="init", pulse_id="x", ...)
```

**After (v3.0)**:
```bash
/compass-start domain-action-target "objective" milestone-id
```

#### For Automation Scripts
CLI commands unchanged:
```bash
uv run compass pulse-init --id x --obj "y" --milestone z
```

#### Breaking: MCP Tools Gone
If you were using MCP tools directly, migrate to:
1. Skills (interactive): `/compass-*` commands
2. CLI (scripts): `uv run compass` commands
```

### 6. Python Modules to Delete/Update

**Delete Completely**:
1. `project_compass/mcp_server.py` (519 lines)
2. `project_compass/src/router.py` (if exists - routes MCP calls)

**Update (Remove Spec Functions)**:
1. `project_compass/cli.py`:
   - Remove lines 329-508 (spec-* commands)
   - Remove spec subparsers
   - Keep pulse-* commands

2. `project_compass/src/pulse_exporter.py`:
   - Remove `create_snapshot()` function (if removing snapshots)
   - Keep `export_pulse()`, `register_artifact()`, `audit_staging()`

**No Changes Needed**:
- `project_compass/src/core.py` ✅
- `project_compass/src/state_schema.py` ✅
- `project_compass/src/rule_injector.py` ✅

### 7. Skills Updates Required

**Current Issue**: Skills call MCP tools that will be deleted

**Files to Update** (7 skills):
1. `skills/compass-start/SKILL.md`
2. `skills/compass-status/SKILL.md`
3. `skills/compass-register/SKILL.md`
4. `skills/compass-finish/SKILL.md`
5. `skills/compass-resume/SKILL.md`
6. `skills/compass-help/SKILL.md`
7. `skills/audit-run/SKILL.md`

**Change Pattern**:

**Before**:
```yaml
Call MCP tool:
mcp__unified__compass_meta_pulse(
  kind="init",
  pulse_id="[id]",
  objective="[obj]",
  milestone_id="[milestone]"
)
```

**After**:
```yaml
Call CLI:
!`uv run compass pulse-init --id [id] --obj "[obj]" --milestone [milestone]`
```

OR

```yaml
Execute via Bash tool:
Use Bash tool to run:
uv run compass pulse-init --id [validated-id] --obj "[validated-objective]" --milestone [validated-milestone]
```

**Template for Skills**:
```yaml
---
name: compass-start
description: Initialize new pulse
---

# Instructions
1. Validate inputs
2. Execute command:

```bash
uv run compass pulse-init \
  --id [pulse-id] \
  --obj "[objective]" \
  --milestone [milestone-id]
```

3. Confirm success
```

### 8. Directory Cleanup

**Directories to Evaluate**:

1. **`/workspaces/dev_tools/project_compass/pulse_staging/`**
   - Decision: Delete (use `project_compass/pulse_staging/` only)
   - Action: Move any content to correct location first

2. **`/workspaces/dev_tools/project_compass/snapshots/`**
   - Decision: TBD (remove if not valuable)
   - Action: Archive existing snapshots first if removing

3. **`/workspaces/dev_tools/project_compass/project_compass/pulse_staging/artifacts/`**
   - Current state: Contains `constitution.md`, `specification.md`, `implementation_plan.md`, `tasks.md`
   - Decision: Clean out spec-kit generated files
   - Action: Review and delete if not needed

4. **`/workspaces/dev_tools/project_compass/design/`**
   - Contains: `spec-kit-integration.md`, old design docs
   - Decision: Archive or delete obsolete design docs
   - Action: Move to `history/design-archive/` or delete

---

## Implementation Plan (High-Level)

### Phase 1: Pre-Cleanup Audit (1 hour)
1. Search codebase for external MCP dependencies
2. Inventory `pulse_staging/` directories - what's where?
3. Evaluate snapshots value - keep or remove?
4. Backup current state (git commit + tag)

### Phase 2: Code Removal (2 hours)
1. Delete `mcp_server.py`
2. Delete `router.py` (if exists)
3. Remove spec-* commands from `cli.py`
4. Remove `create_snapshot()` if decided
5. Update `pyproject.toml` (remove MCP dependencies if any)

### Phase 3: Skills Migration (2 hours)
1. Update all 7 skills to call CLI instead of MCP
2. Test each skill end-to-end
3. Update skill documentation

### Phase 4: Documentation (2 hours)
1. Rewrite AGENTS.md (skills-first)
2. Update AGENTS.yaml (remove MCP, add skills)
3. Write CHANGELOG.md v3.0 entry
4. Update skills/README.md (remove MCP references)

### Phase 5: Directory Cleanup (1 hour)
1. Consolidate `pulse_staging/` directories
2. Remove/archive `snapshots/` if decided
3. Clean spec-kit artifacts from staging
4. Archive obsolete design docs

### Phase 6: Testing & Validation (2 hours)
1. Test complete workflow: start → register → finish
2. Verify directive injection works
3. Test all 7 skills
4. Confirm no broken references

**Total Estimated Time**: 10 hours

---

## Risk Assessment

### High Risk Items
1. **External MCP Usage**: If other systems call MCP tools, they break
   - Mitigation: Search for external dependencies first

2. **Data Loss**: Deleting wrong staging directory loses work
   - Mitigation: Full backup before cleanup

3. **Skills Broken**: Skills can't call CLI properly
   - Mitigation: Test one skill first, validate pattern

### Medium Risk Items
1. **Documentation Drift**: Docs don't match implementation
   - Mitigation: Update docs immediately after code changes

2. **Snapshots Removal**: Losing valuable feature
   - Mitigation: Research usage patterns first, keep if valuable

### Low Risk Items
1. **CLI Changes**: Breaking automation scripts
   - Mitigation: CLI commands unchanged, only additions removed

---

## Decision Points (Requires User Input)

### Decision 1: Snapshots System
**Question**: Keep or remove `create_snapshot()` and `snapshots/` directory?

**Context**:
- Snapshots allow mid-work checkpointing without exporting pulse
- Git commits provide similar functionality
- Adds complexity to system

**Options**:
- A: Keep snapshots (preserve feature)
- B: Remove snapshots (simplify system)

**Recommendation**: Remove (Option B)

### Decision 2: Staging Directory Consolidation
**Question**: Which `pulse_staging` directory is authoritative?

**Options**:
- A: Delete top-level, use `project_compass/pulse_staging/` (matches code)
- B: Delete nested, use top-level `pulse_staging/` (update code)

**Recommendation**: Option A (follow existing code structure)

### Decision 3: Spec-Kit Artifact Types
**Question**: Remove spec-specific artifact types (`specification`, `requirements`, `architecture`)?

**Options**:
- A: Remove types (simplify to 6 core types)
- B: Keep types (preserve flexibility)

**Recommendation**: Remove (Option A) - types not used without spec-kit

### Decision 4: Router Module
**Question**: Delete `router.py` or repurpose?

**Investigation Needed**: Does `router.py` exist? What does it do?

**If exists and only routes MCP**: Delete
**If has other functionality**: Review and decide

---

## Success Criteria

Cleanup is complete when:
- ✅ Zero MCP server code remains
- ✅ All skills call CLI (no MCP dependencies)
- ✅ Single `pulse_staging/` directory
- ✅ Documentation updated (AGENTS.md, AGENTS.yaml, CHANGELOG.md)
- ✅ All 7 skills tested and working
- ✅ No broken imports or references
- ✅ Complete workflow tested (init → work → export)

---

## Feedback Questions

Before proceeding, confirm:

1. **Snapshots**: Remove or keep?
2. **Staging**: Which directory to keep?
3. **Spec Types**: Remove artifact types?
4. **Timeline**: Is 10-hour estimate acceptable?
5. **Backup**: Git tag for rollback sufficient?

---

**Next Step**: User approval to proceed with nuclear cleanup.
