# Project Compass Skills

Skills for improved discoverability and directive persistence across sessions.

## Problem Statement

**Before Skills**:
- MCP tools exist but not discoverable
- Manual repetitive prompting required
- Session handovers lose project directives
- Instructions like "keep docs concise, AI-only" don't carry over

**With Skills**:
- `/compass-start`, `/compass-status` etc. are discoverable
- Directives auto-injected every session via dynamic context
- Guided workflows with validation
- Consistent behavior across sessions

---

## Available Skills

### Core Workflow

| Skill | Invocation | Purpose |
|-------|-----------|---------|
| **compass-start** | `/compass-start [id] "[obj]" [milestone]` | Initialize pulse with directives |
| **compass-status** | `/compass-status` | Check pulse state + directive reminder |
| **compass-register** | `/compass-register [file] [type]` | Register artifact (auto-detect type) |
| **compass-finish** | `/compass-finish` | Export with pre-checks |
| **compass-resume** | `/compass-resume` | Load session context + directives |
| **compass-help** | `/compass-help` | Command reference |

### Validation & Audit

| Skill | Invocation | Purpose |
|-------|-----------|---------|
| **audit-run** | `/audit-run [category]` | Execute audit checklist systematically |

**Audit Categories**:
- `plm-correctness` - Permutation logic, loss computation
- `flash-attention` - Numerical equivalence
- `device-placement` - CUDA context validation
- `gradient-flow` - Backprop validation
- `configuration` - Hydra config consistency
- `performance` - Throughput benchmarks
- `all` - Full audit suite

---

## Key Features

### 1. Directive Auto-Injection

Every skill loads project directives using dynamic context:
```yaml
!`cat vault/directives/00_meta_cognition.md`
!`cat vault/directives/01_naming_standards.md`
!`cat vault/directives/02_artifact_purity.md`
```

**Result**: Directives persist across sessions without manual reminding.

### 2. Validation Before Execution

Skills validate inputs before calling MCP tools:
- Pulse ID format checking
- Banned term detection
- Artifact type validation
- File existence verification

### 3. Guided Workflows

Interactive prompts for complex operations:
- Pre-export audit (finds unregistered files)
- Auto-type detection for artifacts
- Error handling with corrective actions

### 4. Context Loading

`/compass-resume` loads:
- Vessel state
- Project directives
- Staging artifacts
- Recent git changes
- Previous pulse context

---

## Implementation Notes

### Nuclear Refactor Approach
- **No legacy support**: Direct implementation
- **Fail fast**: Validate early, surface errors immediately
- **Clean migration**: Skills wrap MCP tools, no duplication

### Skill Architecture
```
skills/
├── compass-start/
│   └── SKILL.md          # Dynamic directive injection
├── compass-status/
│   └── SKILL.md          # State + directive reminder
├── compass-register/
│   └── SKILL.md          # Auto-type detection
├── compass-finish/
│   └── SKILL.md          # Pre-export audit
├── compass-resume/
│   └── SKILL.md          # Session handover protection
├── compass-help/
│   └── SKILL.md          # Command reference
└── audit-run/
    └── SKILL.md          # Audit automation (forked context)
```

### MCP Tool Mapping

| Skill | MCP Tool Call |
|-------|---------------|
| compass-start | `compass_meta_pulse(kind="init")` |
| compass-status | `compass_meta_pulse(kind="status")` |
| compass-register | `compass_meta_pulse(kind="sync")` |
| compass-finish | `compass_meta_pulse(kind="export")` |

---

## Usage Examples

### Starting New Work
```bash
/compass-start recognition-audit-plm "Audit PLM implementation for correctness" v1.0-recognition-optimization
```

### Resuming After Break
```bash
/compass-resume
# Loads full context: vessel state, directives, artifacts, git status
```

### Creating and Registering Artifact
```bash
# Create file in pulse_staging/artifacts/
# Then:
/compass-register design_plm_validation.md
# Auto-detects type from filename
```

### Running Audit
```bash
/audit-run plm-correctness
# Generates: pulse_staging/artifacts/audit/plm-correctness_audit.md
# Auto-registers artifact
```

### Completing Work
```bash
/compass-finish
# Pre-export audit
# Lists unregistered files (if any)
# Exports to history/
```

---

## Testing Checklist

- [ ] `/compass-start` validates pulse ID format
- [ ] `/compass-start` rejects banned terms (new, session, test)
- [ ] `/compass-status` shows directive reminders
- [ ] `/compass-register` auto-detects types correctly
- [ ] `/compass-finish` finds unregistered files
- [ ] `/compass-resume` injects all directives
- [ ] `/audit-run` generates and registers artifacts
- [ ] Directives persist across multiple skill invocations

---

## Session Handover Solution

**Problem**: New sessions lose directives like:
- "Keep documentation concise"
- "AI-only documentation, not user-facing"
- "Max 20 words per note"

**Solution**:
1. Use `/compass-resume` at session start
2. All skills auto-inject directives via `!`command``
3. Skills explicitly remind: "THESE APPLY TO ENTIRE SESSION"

**Result**: No instruction drift, consistent behavior.

---

## Future Enhancements

Potential additions:
- `/compass-search` - Search across archived pulses
- `/compass-diff` - Compare current pulse with previous
- `/test-generate` - Generate test suite from audit findings
- `/compass-migrate` - Migrate artifacts between pulses

---

## Technical Details

### Dynamic Context Injection
Pattern: `!`command``

Executes command and injects output before Claude sees skill content.

Example:
```yaml
## Current State
!`cat .vessel/vessel_state.json`
```

Becomes:
```yaml
## Current State
{
  "version": "2.0.0",
  "active_pulse": {...}
}
```

### Forked Context for Audit
`audit-run` uses:
```yaml
context: fork
agent: Explore
```

Runs in isolated subagent with access to codebase exploration tools.

---

## Maintenance

### Adding New Skills
1. Create `skills/[name]/SKILL.md`
2. Include frontmatter with `name`, `description`
3. Add dynamic context injection if needed
4. Test with `/[skill-name]`
5. Update this README

### Updating Directives
Directives in `vault/directives/` are auto-loaded.
No skill changes needed when directives update.

---

## References

- Core Docs: `AGENTS.md`
- MCP Tools: `AGENTS.yaml`
- Vault Directives: `vault/directives/`
- Skill System: [Claude Code Skills Reference](https://claude.ai/docs/skills)
