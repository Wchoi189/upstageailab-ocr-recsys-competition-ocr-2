# Implementation Plan

## Approach
# Implementation Plan: Project Compass 2.0 & Bundle Enhancements

## Phase 1: Core Pulse Workflow Improvements (Critical)

### 1. Auto-Reconciliation of Artifacts
**Goal**: Prevent "orphaned artifact" errors.
**Changes**:
- Modify `Compass.get_status()` in `src/core.py`.
- Add logic to iterate through `state.artifacts`.
- If file missing: remove from list, log warning (ephemeral), and auto-save state.
- Add `clean=True` default param to status checks.

### 2. Auto-Registration for Spec-Kit
**Goal**: Eliminate manual `pulse-sync`.
**Changes**:
- Update `mcp_server.py` handling of `compass_meta_spec`.
- After successful file creation, immediately call internal `pulse.sync_artifact(path)`.
- Ensure this is largely transparent to the user but confirmed in output.

### 3. Snapshot System
**Goal**: Enable intermediate checkpoints.
**Changes**:
- Add `snapshot()` method to `ProjectPulse` class.
- Logic:
    1. Create `snapshots/<timestamp>_<label>/` dir.
    2. Copy `vessel_state.json`.
    3. Copy all current artifacts to snapshot dir.
    4. Log snapshot metadata.
- Expose via new MCP tool or `compass_meta_pulse(kind="checkpoint")`.

### 4. Explicit Artifact Lifecycle & Better Errors
**Goal**: Clarity and actionability.
**Changes**:
- Enhance `pulse_status` output to show "Staged" vs "Untracked" (using `os.listdir` vs manifest).
- Refactor `PulseError` exceptions to include `recommendation` field.
- Update `cli.py` to print recommended actions on failure.

## Phase 2: Context Bundle Evolution

### 1. Hierarchical & Lazy Bundles
**Goal**: Efficiency and organization.
**Changes**:
- Refactor `ContextBundle` class (if exists, or list dict structure).
- Add `sub_bundles` support in schema.
- Implement lazy-loading: Read only YAML metadata on listing; read files only on `get_context`.

### 2. New Bundles
**Goal**: Address specific missing context.
**Changes**:
- Create `rec-audit-foundation` bundle definition.
- Create `rec-historical-context` bundle definition.
- Create `v5-domains-standard` bundle definition.

## Verification Plan
1. **Test Auto-Cleanup**: Delete a registered artifact, run `status`, check `vessel_state.json`.
2. **Test Snapshot**: Run `checkpoint`, verify directory structure and state copy.
3. **Test Auto-Reg**: Create spec, verify it appears in `status` immediately.
4. **Test Bundles**: Request new bundles, verify content and hierarchy.


## High-Level Steps
1. **Analysis Phase**
   - Requirements review
   - Architecture design
   - Risk assessment

2. **Development Phase**
   - Core implementation
   - Testing strategy
   - Integration planning

3. **Validation Phase**
   - Quality assurance
   - Performance testing
   - Deployment preparation

## Success Criteria
- All requirements met
- Code quality standards maintained
- Performance benchmarks achieved

## Timeline
TBD - To be determined based on scope and resources

## Status
- Created: 2026-02-02T20:07:17.775778
- Tool: Project Compass v2
- Status: Draft
