---
name: compass-start
description: Initialize new pulse with vault directives injected. Use when user starts new work or explicitly requests pulse initialization.
disable-model-invocation: false
---

# Compass Start - Pulse Initialization

## Current Vessel State
!`cat /workspaces/dev_tools/project_compass/.vessel/vessel_state.json | head -50`

## Auto-Injected Project Directives

### Meta-Cognition Protocol
!`cat /workspaces/dev_tools/project_compass/vault/directives/00_meta_cognition.md`

### Naming Standards
!`cat /workspaces/dev_tools/project_compass/vault/directives/01_naming_standards.md`

### Artifact Purity Rules
!`cat /workspaces/dev_tools/project_compass/vault/directives/02_artifact_purity.md`

---

## CRITICAL SESSION DIRECTIVES

**These directives persist for the ENTIRE session. Apply to ALL work:**

1. **Concise Communication**: Max 20 words for any status note. No narrative summaries.
2. **Artifact Location**: ALL files in `pulse_staging/artifacts/` ONLY.
3. **Reflection Protocol**: Use [Compass:Reflection] before creating ANY artifact.
4. **Evidence-Based**: "The code shows..." not "I think..."
5. **AI-Only Docs**: Documentation is for AI agents only, NOT user-facing.

---

## Initialize Pulse

Parse arguments as: `pulse-id` `objective` `milestone-id`

Example: `/compass-start recognition-audit-plm "Audit PLM implementation" v1.0-recognition-optimization`

### Validation Checklist
- [ ] Pulse ID format: `{domain}-{action}-{target}` (lowercase, hyphens)
- [ ] Banned terms NOT used: new, session, test, tmp
- [ ] Objective: 20-500 characters
- [ ] Milestone ID exists or is valid format

### Execution
1. Validate inputs against naming standards
2. Call MCP tool: `mcp__unified__compass_meta_pulse`
   - kind: "init"
   - pulse_id: [validated-id]
   - objective: [validated-objective]
   - milestone_id: [validated-milestone]
3. Confirm initialization with pulse-status
4. Remind user of directive constraints

### Post-Init Reminder
After successful init, output:
```
✅ Pulse initialized: [pulse-id]
📋 Active Directives: Max 20 words/note, artifacts in staging only, [Compass:Reflection] required
```
