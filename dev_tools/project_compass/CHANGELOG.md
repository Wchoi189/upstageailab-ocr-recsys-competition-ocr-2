# Project Compass Changelog

## [3.0.1] - 2026-02-12 - Architecture Fix

### Fixed
- **Critical Bug**: Path auto-detection now uses `pyproject.toml` instead of directory name
- **Follow-up Bug**: Fixed fallback logic creating `.vessel` at workspace root when called from external code
- **Stricter Validation**: Now requires BOTH `vault/` AND (`history/` OR `.vessel/`) to confirm compass project
- **Smart Search**: Added fallback search for `dev_tools/project_compass` in common locations
- **Directory Structure**: Migrated `.vessel/` and `pulse_staging/` to project root (Python best practices)
- **Path Resolution**: Robust detection works from ANY working directory (tested from workspace, dev_tools, and project)
- Python package (`project_compass/`) now contains ONLY code

### Changed
- All data directories (`.vessel/`, `vault/`, `history/`, `pulse_staging/`) moved to project root
- Skills updated to reference correct paths (compass-start, compass-resume, compass-finish)
- Documentation updated: AGENTS.md, AGENTS.yaml reflect new structure
- Version bumped to 3.0.1 to reflect bug fix

### Technical Details
- See [POST_NUCLEAR_AUDIT.md](POST_NUCLEAR_AUDIT.md) for detailed analysis
- See [ARCHITECTURE_AUDIT_COMPLETE.md](ARCHITECTURE_AUDIT_COMPLETE.md) for complete summary
- Root cause: Auto-detection confused package subdirectory with project root

---

## [3.0.0] - 2026-02-12 - BREAKING CHANGES ⚠️

### BREAKING CHANGES

This is a **nuclear refactor** with zero backward compatibility. All MCP tools have been removed.

#### Removed - MCP Architecture
- **MCP Server** (`mcp_server.py` - 519 lines)
- **MCP Tools**: `compass_meta_pulse`, `compass_meta_spec`
- **MCP Resources**: `vessel://state`, `vessel://rules`, `vessel://staging`
- **Router Module** (`router.py` - 91 lines)
- **Snapshot System** (`create_snapshot()` function + `snapshots/` directory)

#### Removed - Spec-Kit Integration
- **CLI Commands**: `spec-constitution`, `spec-specify`, `spec-plan`, `spec-tasks` (180 lines)
- **Artifact Types**: `specification`, `requirements`, `architecture`
- **Design Documentation**: `spec-kit-integration.md`

### Added - Skills System

**7 New Skills** (Primary Interface):
- `/compass-start` - Initialize pulse with vault directives
- `/compass-status` - Check pulse state + directive reminders
- `/compass-register` - Register artifacts (auto-type detection)
- `/compass-finish` - Export with pre-checks
- `/compass-resume` - Session context loader with directive re-injection
- `/compass-help` - Command reference
- `/audit-run` - Execute audit checklist systematically

**Skills Documentation**:
- `skills/README.md` - Comprehensive skill reference
- `skills/QUICKSTART.md` - Quick start guide
- `SKILLS_IMPLEMENTATION.md` - Implementation details

### Changed - Architecture Simplification

**Before (v2.x)**:
```
User → Skills/CLI/MCP → MCP Server → Router → Core
4 layers, ~2000 lines, 3 entry points
```

**After (v3.0)**:
```
User → Skills/CLI → Core
2 layers, ~800 lines, skills primary
```

**Path Consolidation**:
- Moved `.vessel/` to `project_compass/.vessel/` (consistent with code)
- Consolidated duplicate `pulse_staging/` directories
- Single, clean directory structure

**Skills Architecture**:
- Skills now call CLI directly (not MCP)
- Vault directives auto-injected via dynamic context (`!`command``)
- Session handover protection (directives persist)

### Migration Guide

#### For AI Agents

**Before (v2.x)**: MCP tools
```python
mcp__unified__compass_meta_pulse(
  kind="init",
  pulse_id="recognition-audit-plm",
  objective="Audit PLM implementation",
  milestone_id="v1.0-recognition-optimization"
)
```

**After (v3.0)**: Skills (primary)
```bash
/compass-start recognition-audit-plm "Audit PLM implementation" v1.0-recognition-optimization
```

**Directive Persistence**:
- Use `/compass-resume` at session start to reload directives
- All skills auto-inject vault directives
- No more lost instructions between sessions

#### For Automation Scripts

**CLI commands unchanged** (backward compatible):
```bash
uv run compass pulse-init --id recognition-audit-plm \
  --obj "Audit PLM implementation" \
  --milestone v1.0-recognition-optimization
```

#### Breaking: External MCP Usage

If you were calling MCP tools directly:
- **Migration**: Use skills (`/compass-*`) or CLI (`uv run compass`)
- **No MCP tools exist** in v3.0+

### Impact Summary

**Code Reduction**: ~60% (removed ~1200 lines)
- MCP server: 519 lines
- Router: 91 lines
- Spec-kit: 180 lines
- Snapshot: 65 lines
- Docs cleanup: ~345 lines

**Architecture Layers**: 4 → 2 (50% reduction)

**Clarity**: Single primary interface (skills), clear workflow

### Success Metrics

- ✅ Zero MCP code remains
- ✅ All skills call CLI (no MCP dependencies)
- ✅ Single directory structure (no duplicates)
- ✅ Complete documentation update
- ✅ 60% code reduction
- ✅ 2-layer architecture

---

## [2.1.0] - 2026-01-21

### Added
- **Global CLI:** `compass` command available system-wide (via `pyproject.toml`).
- **Vessel Architecture:** Full integration of Pydantic state (`vessel_state.json`), Vault, and Pulse system.
- **Pulse Commands:** `pulse-init`, `pulse-sync`, `pulse-export`, `pulse-status`, `pulse-checkpoint`.
- **Vault:** Directives and milestones now managed in `project_compass/vault/`.

### Changed
- **Unified State:** Replaced `compass.json`/`current_session.yml` with `.vessel/vessel_state.json`.
- **Artifact Management:** Strict staging in `pulse_staging/artifacts/` with manifesto-based export.

## [2.0.0] - 2026-01-17

### Added
- **New CLI Command:** `update-status` for manual compass.json updates
  - Supports `--phase`, `--health`, and `--note` parameters
  - Provides manual override for automated synchronization
- **Automated compass.json Sync:** `session-init` now automatically updates project status
  - Updates `current_phase` with pipeline value
  - Sets `overall_health` to "healthy"
  - Captures first 80 chars of objective as note
  - Always updates `last_updated` timestamp
- **CLI Reference Documentation:** Comprehensive `CLI_REFERENCE.md` with usage examples
- **Enhanced SessionManager:** New `_update_compass_status()` method with atomic writes

### Changed
- **AGENTS.md:** Updated with compass.json management section
- **Session Lifecycle:** Documented automatic compass.json updates in workflow

### Fixed
- **Import Errors:** Fixed obsolete `etk.compass` import in `env_check.py`
- **MCP Server:** Fixed 4 `etk.factory` references to use `etk.cli` in `experiment_manager/mcp_server.py`
- **Documentation:** Updated `uv_lock_state.yml` comment to reference correct CLI command

### Migration Guide

#### For AI Agents
- Use `session-init` as before - compass.json updates automatically
- Use `update-status` for manual corrections when needed
- No breaking changes to existing workflows

#### For Humans
- compass.json no longer requires manual editing
- Use `uv run python -m project_compass.cli update-status` for status updates
- Old `etk check-env` command replaced with `uv run python -m project_compass.cli check-env`

---

## [1.0.0] - 2026-01-13

### Added
- Initial Project Compass CLI with `session-init`, `session-export`, `check-env`
- Environment validation against lock state
- Session lifecycle management
- Separation from Experiment Manager (ETK)

### Changed
- Decoupled from ETK module structure
- Moved environment checker to `project_compass.src.core`

---

## Version Numbering

Project Compass follows semantic versioning:
- **Major:** Breaking changes to CLI or protocols
- **Minor:** New features, backward compatible
- **Patch:** Bug fixes, documentation updates
