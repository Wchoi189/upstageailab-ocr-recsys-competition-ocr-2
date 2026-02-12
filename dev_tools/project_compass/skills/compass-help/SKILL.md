---
name: compass-help
description: Show Project Compass commands and workflows. Use when user asks about compass commands or needs guidance.
disable-model-invocation: false
---

# Project Compass - Command Reference

## Quick Reference

| Command | Usage | When |
|---------|-------|------|
| `/compass-start` | Initialize new pulse | Starting new work |
| `/compass-status` | Check pulse state | Resuming work, checking progress |
| `/compass-register` | Register artifact | After creating files |
| `/compass-finish` | Export pulse | Work complete |
| `/audit-run` | Execute audit checklist | Validating implementation |

---

## Workflow

### 1. Start New Work
```
/compass-start domain-action-target "Brief objective" milestone-id
```

**Example**:
```
/compass-start recognition-audit-plm "Audit PLM implementation" v1.0-recognition-optimization
```

**Validation**:
- Pulse ID: lowercase, hyphens, format `domain-action-target`
- No banned terms: new, session, test, tmp
- Objective: 20-500 characters

---

### 2. Create Artifacts
Create files in `pulse_staging/artifacts/` ONLY.

**Artifact Types**:
- `design` - Design documents
- `research` - Investigation, research findings
- `walkthrough` - Step-by-step guides
- `implementation_plan` - Implementation plans
- `bug_report` - Bug reports and fixes
- `audit` - Audit findings and analysis

---

### 3. Register Artifacts
```
/compass-register filename.md [type]
```

Type auto-detected from filename/content if not provided.

---

### 4. Check Progress
```
/compass-status
```

Shows:
- Active pulse info
- Registered artifacts
- Unregistered files (if any)
- Suggested next actions

---

### 5. Complete Work
```
/compass-finish
```

Pre-export audit:
- Lists all staging files
- Identifies unregistered files
- Prompts for action before export

---

## Project Directives (Always Active)

**These rules apply to ALL work in Project Compass:**

1. **Concise Notes**: Max 20 words per status note
2. **Staging Only**: All files in `pulse_staging/artifacts/`
3. **Reflection Protocol**: Use [Compass:Reflection] before new files
4. **Evidence-Based**: "Code shows..." not "I think..."
5. **AI-Only Docs**: Documentation for AI agents, not user-facing

### [Compass:Reflection] Template
```
[Compass:Reflection]
- Type: (design|research|walkthrough|implementation_plan|bug_report|audit)
- Justification: Why is this file necessary for THIS pulse?
- Redundancy: Does an existing artifact cover this? Why not update it?
- Lifecycle: Transitional (delete after) or Archived (keep in history)?
```

---

## File Structure

```
project_compass/
├── .vessel/
│   └── vessel_state.json       # State (read-only for users)
├── vault/
│   ├── directives/             # Project rules (read-only)
│   └── milestones/             # Milestone definitions
├── pulse_staging/
│   └── artifacts/              # YOUR WORKSPACE (write here)
└── history/
    └── {milestone}/            # Archived pulses (read-only)
```

---

## Advanced: Audit Automation

For complex validation tasks:
```
/audit-run [category]
```

Executes audit checklist systematically. See `/compass-help audit` for details.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Export blocked | Unregistered files | `/compass-status` to find, then register or delete |
| "File not in staging" | Created file outside staging | Move to `pulse_staging/artifacts/` |
| "Invalid pulse ID" | Format error or banned term | Use `domain-action-target` format, lowercase |

---

## Need More Help?

- Full docs: `dev_tools/project_compass/AGENTS.md`
- Vault directives: `dev_tools/project_compass/vault/directives/`
