# Project Compass CLI Reference

**Version:** 2.0
**Last Updated:** 2026-01-17
**Module:** `project_compass.cli`

---

## Overview

The Project Compass CLI provides tools for session management, environment validation, and project state tracking. All commands must be run from the project root using `uv run`.

---

## Commands

### `check-env`

Validate current environment against Project Compass lock state.

**Usage:**
```bash
uv run python -m project_compass.cli check-env [--strict]
```

**Options:**
- `--strict` - Treat warnings as errors

**Checks:**
- UV binary path matches expected location
- Python version matches requirement
- PyTorch/CUDA configuration matches lock state

**Exit Codes:**
- `0` - Environment valid
- `1` - Validation failed

**Example:**
```bash
$ uv run python -m project_compass.cli check-env
🔒 Environment Guard: Checking against Project Compass lock state...
✅ Environment validated against Compass lock state
```

---

### `session-init`

Initialize or update the current session context.

**Usage:**
```bash
uv run python -m project_compass.cli session-init \
  --objective "Session objective" \
  [--pipeline PIPELINE]
```

**Options:**
- `--objective`, `-o` - Primary goal for this session (required)
- `--pipeline`, `-p` - Active pipeline (default: kie)
  - Choices: `text_detection`, `text_recognition`, `layout_analysis`, `kie`, `roadmap`

**Side Effects:**
- Creates/updates `active_context/current_session.yml`
- **Automatically updates `compass.json`** with:
  - `current_phase` = pipeline
  - `overall_health` = "healthy"
  - `note` = first 80 chars of objective
  - `last_updated` = current timestamp

**Example:**
```bash
uv run python -m project_compass.cli session-init \
  --objective "Execute Hydra configuration audit and refactor" \
  --pipeline kie
```

---

### `session-export`

Archive current session to history.

**Usage:**
```bash
uv run python -m project_compass.cli session-export \
  [--note "Export note"] \
  [--force]
```

**Options:**
- `--note`, `-n` - Note for the manifest
- `--force`, `-f` - Bypass stale session check

**Behavior:**
- Exports `active_context/` to `history/sessions/TIMESTAMP_SESSION_ID/`
- Creates timestamped session handover
- Clears active context
- Creates fresh session template

**Example:**
```bash
uv run python -m project_compass.cli session-export \
  --note "Completed Hydra refactor documentation"
```

---

### `update-status` ⭐ NEW

Manually update compass.json project status.

**Usage:**
```bash
uv run python -m project_compass.cli update-status \
  [--phase PHASE] \
  [--health HEALTH] \
  [--note NOTE]
```

**Options:**
- `--phase` - Current project phase (e.g., "hydra_config_refactor")
- `--health` - Overall health status
  - Choices: `healthy`, `degraded`, `blocked`
- `--note` - Human-readable status description

**Behavior:**
- Updates specified fields in `compass.json`
- Always updates `last_updated` timestamp
- Uses atomic JSON writes for safety

**When to Use:**
- compass.json becomes stale
- Need to manually correct project status
- Session-init doesn't capture current state accurately

**Examples:**
```bash
# Update phase only
uv run python -m project_compass.cli update-status \
  --phase "hydra_config_refactor"

# Update health status
uv run python -m project_compass.cli update-status \
  --health "degraded"

# Update all fields
uv run python -m project_compass.cli update-status \
  --phase "hydra_config_refactor" \
  --health "healthy" \
  --note "Executing Hydra configuration audit - 50% complete"
```

---

### `wizard`

Interactive Sprint Context setup (guided session initialization).

**Usage:**
```bash
uv run python -m project_compass.cli wizard
```

**Behavior:**
- Launches interactive prompt
- Guides through session setup
- Alternative to manual `session-init`

---

## compass.json Synchronization

### Automatic Updates

The following commands automatically update `compass.json`:

| Command        | Updates                           |
| -------------- | --------------------------------- |
| `session-init` | phase, health, note, last_updated |

### Manual Updates

Use `update-status` when:
- compass.json is stale
- Session state changed outside normal flow
- Need to correct inaccurate status

### Fields

| Field                           | Description        | Updated By                  |
| ------------------------------- | ------------------ | --------------------------- |
| `last_updated`                  | Timestamp (KST)    | All update commands         |
| `project_status.current_phase`  | Active work area   | session-init, update-status |
| `project_status.overall_health` | System health      | session-init, update-status |
| `project_status.note`           | Status description | session-init, update-status |

---

## Best Practices

### Session Lifecycle

1. **Start:** `session-init` with clear objective
2. **Work:** Execute tasks, update artifacts
3. **Update:** Modify `current_session.yml` before export
4. **Export:** `session-export` with completion note

### Environment Validation

- Run `check-env` before starting training
- Use `--strict` in CI/CD pipelines
- Fix environment mismatches immediately

### Status Management

- Let `session-init` handle compass.json automatically
- Use `update-status` sparingly for corrections
- Keep notes concise and descriptive

---

## Troubleshooting

### "Environment breach detected"

**Cause:** UV path or PyTorch version mismatch
**Fix:**
```bash
# Ensure correct UV binary
export PATH="/opt/uv/bin:$PATH"

# Resync environment
uv sync

# Verify
uv run python -c "import torch; print(torch.__version__)"
```

### "Stale session ID detected"

**Cause:** Trying to export without updating session_id
**Fix:**
1. Update `active_context/current_session.yml`
2. Change `session_id` to new value
3. Or use `--force` to bypass check

### compass.json not updating

**Cause:** Permission issues or file lock
**Fix:**
```bash
# Check file permissions
ls -l project_compass/compass.json

# Manually update if needed
uv run python -m project_compass.cli update-status --note "Manual update"
```

---

## Migration Notes

### From ETK Commands

Old ETK commands have been replaced:

| Old (ETK)                 | New (Project Compass)                                |
| ------------------------- | ---------------------------------------------------- |
| `etk check-env`           | `uv run python -m project_compass.cli check-env`     |
| Manual compass.json edits | `uv run python -m project_compass.cli update-status` |

### Breaking Changes

None - all commands are backward compatible.

---

## See Also

- **AGENTS.md** - AI agent protocols and constraints
- **compass.json** - Global project state
- **active_context/current_session.yml** - Current session context
