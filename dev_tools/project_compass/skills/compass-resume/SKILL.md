---
name: compass-resume
description: Resume work on existing pulse with full context injection. Use when starting new session or after break.
disable-model-invocation: false
---

# Compass Resume - Session Context Loader

## Purpose
Load full project context when resuming work, ensuring directives and state are re-injected.

---

## Auto-Injected Context

### 1. Current Vessel State
!`cat /workspaces/dev_tools/project_compass/.vessel/vessel_state.json`

### 2. Project Directives (Re-Injection)
!`cat /workspaces/dev_tools/project_compass/vault/directives/00_meta_cognition.md`
!`cat /workspaces/dev_tools/project_compass/vault/directives/01_naming_standards.md`
!`cat /workspaces/dev_tools/project_compass/vault/directives/02_artifact_purity.md`

### 3. Active Pulse Artifacts
!`ls -la /workspaces/dev_tools/project_compass/pulse_staging/artifacts/ 2>/dev/null | tail -20`

### 4. Recent Git Status
!`git -C /workspaces status --short 2>/dev/null | head -20`

---

## Critical Directive Reminders

**THESE APPLY TO ENTIRE SESSION:**

1. **Concise Communication**: Max 20 words per status note
2. **Staging Only**: All artifacts in `pulse_staging/artifacts/`
3. **[Compass:Reflection]**: Required before creating any file
4. **Evidence-Based**: "Code shows..." not "I think..."
5. **AI-Only Docs**: Not user-facing documentation
6. **No Comprehensive Docs**: Keep documentation concise, only what AI agents need

---

## Context Summary

Analyze loaded state and provide:

```
🎯 Active Pulse: [pulse-id]
📌 Objective: [objective]
⏱️  Duration: [time since start]
📊 Token Burden: [level]

📁 Artifacts ([count]):
- [list registered artifacts]

📝 Recent Changes:
- [git status highlights]

🚀 Suggested Next Actions:
1. [action based on pulse state]
2. [action based on artifacts]
3. [action based on git status]

⚠️  Active Directives Loaded:
✓ Max 20 words/note
✓ Staging only
✓ [Compass:Reflection] required
✓ AI-only docs
```

---

## Previous Pulse Context

If INDEX.md exists, also load:
!`cat /workspaces/dev_tools/project_compass/pulse_staging/artifacts/INDEX.md 2>/dev/null | head -50`

Summarize:
- Previous pulse ID and status
- Key artifacts from previous work
- Bugs fixed or issues resolved
- Known issues or blockers

---

## Usage

Run automatically when:
- User starts new session
- User types `/compass-resume`
- User asks "what was I working on?"
- User asks "what's the current state?"

No arguments needed - reads from vessel state.

---

## Handover Protection

This skill solves the "lost directives" problem by:
1. **Auto-injecting** vault directives every session
2. **Reminding** of workflow constraints
3. **Loading** previous context automatically
4. **Preventing** instruction drift across sessions

The directives are NOT suggestions - they are **hard constraints** that persist for the entire session.
