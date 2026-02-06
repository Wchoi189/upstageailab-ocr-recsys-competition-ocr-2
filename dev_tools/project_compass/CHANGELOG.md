# Project Compass Changelog

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
