# Project Compass Skills - Quick Start Guide

**TL;DR**: Use `/compass-*` commands instead of MCP tools. Directives auto-inject every session.

---

## First Time Setup

Skills are ready to use immediately. No installation needed.

To see available commands:
```
/compass-help
```

---

## Daily Workflow

### 1. Start Your Day
```bash
/compass-resume
```
**Result**: Loads context, injects directives, shows current state.

### 2. Start New Work
```bash
/compass-start domain-action-target "Brief objective" milestone-id
```

**Example**:
```bash
/compass-start recognition-audit-plm "Audit PLM implementation" v1.0-recognition-optimization
```

### 3. Create Artifacts
Create files in `pulse_staging/artifacts/`:
- Design docs
- Research findings
- Audit reports
- Implementation plans

### 4. Register Artifacts
```bash
/compass-register filename.md
```
Type auto-detected from filename/content.

### 5. Check Progress
```bash
/compass-status
```

### 6. Finish Work
```bash
/compass-finish
```
Pre-export audit runs automatically.

---

## Audit Automation

Run systematic code validation:
```bash
/audit-run plm-correctness
```

Available categories:
- `plm-correctness`
- `flash-attention`
- `device-placement`
- `gradient-flow`
- `configuration`
- `performance`
- `all`

**Result**: Generates artifact in `pulse_staging/artifacts/audit/`, auto-registers.

---

## Session Handover Protection

**Problem**: Directives lost between sessions.

**Solution**: `/compass-resume` at session start.

**Directives that persist**:
- Max 20 words per status note
- All artifacts in staging only
- [Compass:Reflection] required
- AI-only documentation (not user-facing)
- No comprehensive docs (concise only)

---

## Testing Skills

### Test compass-start
```bash
/compass-start test-pulse-demo "Testing pulse initialization" v1.0-test
```
Should reject "test" as banned term.

Try:
```bash
/compass-start validation-demo-parseq "Testing pulse initialization" v1.0-test
```
Should succeed.

### Test compass-status
```bash
/compass-status
```
Should show current pulse + directive reminders.

### Test compass-register
```bash
# Create test file
echo "# Test Design" > /workspaces/dev_tools/project_compass/pulse_staging/artifacts/test_design.md

# Register it
/compass-register test_design.md
```
Should auto-detect type as "design".

### Clean Up
```bash
rm /workspaces/dev_tools/project_compass/pulse_staging/artifacts/test_design.md
```

---

## Troubleshooting

### "Skill not found"
**Cause**: Skills not loaded by Claude Code yet.
**Fix**: Restart Claude Code or wait ~1min for skill reload.

### "Invalid pulse ID"
**Cause**: Format error or banned term.
**Fix**: Use `domain-action-target` format, lowercase, no banned terms (new, session, test, tmp).

### "File not in staging"
**Cause**: Created file outside `pulse_staging/artifacts/`.
**Fix**: Move file to staging directory.

### Export blocked
**Cause**: Unregistered files in staging.
**Fix**: Run `/compass-status` to find, then register or delete.

---

## Tips

1. **Always start with `/compass-resume`** at session start
2. **Use `/compass-status`** frequently to check state
3. **Let `/compass-register` auto-detect** types (faster)
4. **Run `/compass-finish`** to catch unregistered files early
5. **Use `/audit-run` for systematic validation** (better than manual)

---

## Need Help?

- Full reference: `/compass-help`
- Implementation docs: `SKILLS_IMPLEMENTATION.md`
- Project docs: `AGENTS.md`
