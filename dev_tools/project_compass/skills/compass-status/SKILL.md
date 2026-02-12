---
name: compass-status
description: Check current pulse state, artifacts, and next steps. Use when resuming work or checking progress.
disable-model-invocation: false
---

# Compass Status - Pulse State Check

## Quick Status Check

Get current pulse state via CLI:
```bash
uv run compass pulse-status
```

## Display Format

Show in structured format:
```
🎯 Active Pulse: [pulse-id]
📌 Milestone: [milestone-id]
⏱️  Started: [timestamp]
📊 Token Burden: [low|medium|high]

📁 Registered Artifacts ([count]):
- [artifact-1] (type)
- [artifact-2] (type)

📋 Unregistered Files (if any):
- [file-1]
- [file-2]

⚠️  Blockers (if any):
- [blocker-1]

🚀 Suggested Next Actions:
[Based on pulse state and artifacts]
```

## Context Loading

If resuming after session break, also show:
1. Last exported pulse location
2. Key artifacts from previous pulse
3. Continuation points from INDEX.md

## Auto-Inject Directives Reminder

Always conclude with:
```
📌 Active Directives:
- Max 20 words per note
- Artifacts in pulse_staging/artifacts/ only
- [Compass:Reflection] before new files
- AI-only docs (not user-facing)
```
