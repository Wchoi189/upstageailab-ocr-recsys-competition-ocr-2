# Phase 7 Migration Archive

This directory contains one-time migration tools that were used to transition AgentQMS from the legacy architecture to the Spec-Kit architecture.

## Archived Scripts

### `fix_migration_paths.py`
**Purpose**: Automated path reference updates
**Used**: Phase 7+ cleanup (2026-02-02)
**Function**: Updated 7 files, replacing references to deleted paths:
- `standards/schemas` → `.agentqms/schemas`
- `standards/registry.yaml` → `.agentqms/registry.yaml`
- `tier1-sst/` → `specs/tier1-contracts/`

**Status**: Completed successfully, no longer needed

### `check_migration_deps.sh`
**Purpose**: Comprehensive dependency scan for migration issues
**Used**: Phase 7+ cleanup (2026-02-02)
**Function**: Scanned 72 Python files for:
- Deleted path references
- Syntax errors
- Import issues
- System health checks

**Status**: Completed successfully, no longer needed

## Why Archived?

These were one-time migration tools. Archiving prevents:
- Confusion about which tools are current
- Accumulation of unmaintained scripts
- Bloat in `scripts/utils/`

## Restoration

If needed for rollback or reference:
```bash
cp .archive/phase7-migration/fix_migration_paths.py scripts/utils/
cp .archive/phase7-migration/check_migration_deps.sh scripts/utils/
```

## Related Documentation

- Phase 7 Walkthrough: `brain/.../walkthrough.md`
- Migration Impact Assessment: `brain/.../migration_impact_assessment.md`
- Cleanup Summary: `brain/.../phase_a_cleanup_summary.md`

---

**Archive Date**: 2026-02-02
**Retention**: Permanent (historical reference)
