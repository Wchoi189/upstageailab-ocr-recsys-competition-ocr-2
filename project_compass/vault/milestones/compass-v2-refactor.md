# [Compass:Lock] Milestone: compass-v2-refactor

## Objective
Complete overhaul of Project Compass to Vessel V2 architecture.

## Technical Constraints

### 1. No Dual Architectures
- Delete ALL legacy functions that duplicate new functionality
- Remove `SessionManager` after `PulseManager` is complete
- Remove `compass.json` after `vessel_state.json` is verified

### 2. Pydantic Enforcement
- All state changes MUST go through `VesselState` model
- No direct JSON/YAML edits by AI agents
- Validation failures = hard errors, not warnings

### 3. Staging Isolation
- `pulse_staging/artifacts/` is the ONLY writable location during a pulse
- Export MUST audit disk-vs-manifest consistency
- Stray files = blocked export

## Success Criteria
1. `uv run python -c "from project_compass.src.state_schema import VesselState"` succeeds
2. `uv run python -m project_compass.cli pulse-init --help` shows new commands
3. Zero legacy session/compass.json references in codebase
