# Project Compass V3: Skills-First AI Interface

## 1. Core Concepts

| Term | Concept | Explanation | Usage Example |
| :--- | :------- | :---------- | :------------ |
| **Project Compass** | **The Product** | AI Interface for pulse-based work management | "Using Project Compass to manage work." |
| **Vessel** | **The Engine** | V3.0 architecture with strict state rules | "State stored in `.vessel/`." |
| **Pulse** | **The Work Unit** | A single session or cycle of work | "Starting a pulse to refactor OCR." |
| **Skills** | **Primary Interface** | Discoverable `/compass-*` commands | `/compass-start` to begin work |
| **CLI** | **Automation Interface** | `uv run compass` for scripts | `uv run compass pulse-export` |
| **Staging** | **The Workspace** | `pulse_staging/artifacts/` - ONLY writable location | "Drafting `design.md` in staging." |
| **Vault** | **The Library** | `vault/` - Read-only directives and rules | "Pulse loaded rules from vault." |

## 2. Directory Structure

```text
project_compass/              # Project root
├── .vessel/                  # State storage (vessel_state.json)
├── vault/                    # Read-only rule library
│   ├── directives/           # Core protocols
│   └── milestones/           # Star-chart definitions
├── pulse_staging/            # Staging workspace
│   └── artifacts/            # ACTIVE WORKSPACE (Write here)
├── history/                  # Archived pulses
│   ├── {milestone_id}/       # Grouped by goal
│   │   └── {timestamp}_{id}/ # Pulse snapshot
│   └── legacy/               # Old session archives
├── skills/                   # Skill definitions
│   ├── compass-start/
│   ├── compass-status/
│   └── ...
└── project_compass/          # Python package (CODE ONLY)
    ├── cli.py
    └── src/
        ├── core.py
        ├── state_schema.py
        └── ...
```

**Key Architecture Principles:**
- **Data directories** (`.vessel/`, `vault/`, `history/`, `pulse_staging/`) → Project root
- **Python package** (`project_compass/`) → Code only, no data
- **Path detection** → Simple: check standard location, then cwd, fail fast (no multi-tier fallbacks)

## 3. Skills Interface (Primary)

**Use `/compass-*` skills for all interactive work:**

| Skill | Usage | When |
|-------|-------|------|
| `/compass-start` | Initialize pulse | Starting new work |
| `/compass-status` | Check state | Resuming work, checking progress |
| `/compass-register` | Register artifact | After creating files |
| `/compass-finish` | Export pulse | Work complete |
| `/compass-resume` | Load context | Session start, after break |
| `/compass-help` | Show commands | Need guidance |
| `/audit-run` | Execute audit | Validating implementation |

**Example Workflow:**
```bash
# 1. Start new pulse
/compass-start recognition-audit-plm "Audit PLM implementation" v1.0-recognition-optimization

# 2. Create artifacts in pulse_staging/artifacts/

# 3. Register artifacts
/compass-register audit_plm_correctness.md

# 4. Check progress
/compass-status

# 5. Export when done
/compass-finish
```

## 4. CLI Interface (Automation Only)

**Use `uv run compass` for scripts and automation:**

```bash
# Check environment
uv run compass check-env

# Start pulse (scripted)
uv run compass pulse-init \
  --id domain-action-target \
  --obj "Objective (20-500 chars)" \
  --milestone milestone-id

# Check status
uv run compass pulse-status

# Register artifact
uv run compass pulse-sync \
  --path filename.md \
  --type design

# Export pulse
uv run compass pulse-export

# Update token burden
uv run compass pulse-checkpoint --burden high
```

## 5. Hard Constraints

- **ALL artifacts** must be in `pulse_staging/artifacts/`
- **ALL state changes** must go through Skills or CLI
- **NO manual YAML/JSON editing**
- **NO generic pulse IDs** (banned: "new", "session", "test", "tmp")
- **MAX 20 words** for any status note

## 6. Pulse Lifecycle

```
┌─────────────────────────────────────────────────┐
│       PULSE INIT (via /compass-start)           │
│  → Creates vessel_state.json                    │
│  → Injects rules from vault/                    │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│                     WORK                        │
│  - Create files in pulse_staging/artifacts/     │
│  - Register with /compass-register              │
│  - Check maturity with /compass-status          │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│      PULSE EXPORT (via /compass-finish)         │
│  → Audits staging vs manifest                   │
│  → BLOCKS if unregistered files exist           │
│  → Moves artifacts to history/                  │
└─────────────────────────────────────────────────┘
```

## 7. Vault Directives

Rules in `vault/directives/` are auto-injected on pulse-init:

- `00_meta_cognition.md` - [Compass:Reflection] protocol
- `01_naming_standards.md` - Pulse ID and artifact naming
- `02_artifact_purity.md` - Staging constraints, zero-narrative policy

Milestone-specific rules: `vault/milestones/{milestone_id}.md`

## 8. Skills vs CLI Guidance

### Use Skills When:
- **Interactive work** with AI agent
- **Session start** (use `/compass-resume`)
- **Quick commands** during development
- **Need directive injection** (skills auto-inject vault rules)

### Use CLI When:
- **Automation scripts** (CI/CD, hooks)
- **Headless execution** (no AI agent interaction)
- **Direct invocation** from shell scripts
- **Debugging** (lower-level access)

## 9. Session Handover Protection

**Problem**: Directives lost between sessions

**Solution**: Skills auto-inject vault directives

**Use `/compass-resume` at every session start:**
- Loads vessel state
- Re-injects ALL vault directives
- Shows current context
- Prevents instruction drift

**Directives that persist:**
- Max 20 words per status note
- All artifacts in staging only
- [Compass:Reflection] required before file creation
- AI-only documentation (not user-facing)
- No comprehensive docs (concise only)

## 10. Artifact Types

Valid artifact types for registration:
- `design` - Design documents
- `research` - Investigation, research findings
- `walkthrough` - Step-by-step guides
- `implementation_plan` - Implementation plans
- `bug_report` - Bug reports and fixes
- `audit` - Audit findings and analysis

## 11. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Export blocked | Unregistered files | `/compass-status` to find, then register or delete |
| File not in staging | Created outside staging | Move to `pulse_staging/artifacts/` |
| Invalid pulse ID | Format error or banned term | Use `domain-action-target`, lowercase |
| Skills not found | Skills not loaded | Restart Claude Code or wait ~1min |

## 12. Migration from V2.x

**V2.x (MCP Tools)**:
```python
mcp__unified__compass_meta_pulse(kind="init", pulse_id="x", ...)
```

**V3.0 (Skills)**:
```bash
/compass-start domain-action-target "objective" milestone-id
```

**Breaking Changes in V3.0**:
- ❌ MCP tools removed (`compass_meta_pulse`, `compass_meta_spec`)
- ❌ Spec-kit removed (no `spec-*` commands)
- ❌ Snapshots removed (use git commits)
- ✅ Skills added (primary interface)
- ✅ Directory consolidation (single staging, single vessel)

---

**For full skill documentation**: See `skills/README.md` and `skills/QUICKSTART.md`

**For implementation details**: See `SKILLS_IMPLEMENTATION.md`

**For changes**: See `CHANGELOG.md` v3.0.0
