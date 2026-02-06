# Implementation Plan - Context Bundle Repair

## Goal
Restore the integrity of AgentQMS context bundles by resolving broken file references caused by the recent refactor.

## User Review Required
> [!WARNING]
> This plan relies on remapping `AgentQMS/standards/` references to `archive/legacy_standards_dump/`. This effectively "resurrects" deprecated standards for the sake of context validity. A future pulse should proceed with a proper migration to the new `AgentQMS/specs/` structure if that is the intended long-term state.

## Proposed Changes

### Context Bundles
Remap broken file paths in `AgentQMS/.agentqms/plugins/context_bundles/*.yaml`:
- **Pattern**: `AgentQMS/standards/*` -> `archive/legacy_standards_dump/*`
- **Scope**: All 14 affected bundles (e.g., `ocr-debugging`, `pipeline-development`, etc.)
- **Fallback**: If a file is not found in the archive, check `AgentQMS/specs/` or remove the reference.

### Standards Database
Update `AgentQMS/.agentqms/standards_db.json` to reflect the new physical locations of these files, ensuring the `audit_bundles.py` script passes.

## Verification Plan

### Automated Verification
1.  **Refine Audit Script**: Update `audit_bundles.py` to support the remapping check (optional, but good for debugging).
2.  **Run Audit**: Execute `uv run python audit_bundles.py` after changes.
    - **Success Criteria**: "Total Broken References" should drop to nearly 0. "STATUS" for bundles should become "OK".

### Manual Verification
1.  **Tool Test**: Run `python AgentQMS/tools/core/context/get_context.py --task "debug hydra config"` (or similar) to verify a repaired bundle loads without error.
