---
name: compass-register
description: Register artifact in staging with auto-type detection. Use after creating files in pulse_staging/artifacts/.
disable-model-invocation: false
---

# Compass Register - Artifact Registration

## Quick Registration

Register artifact: $ARGUMENTS

Parse as: `filename [type]`

## Auto-Type Detection

If type not provided, detect from:
- Filename patterns:
  - `*design*.md` → design
  - `*research*.md` → research
  - `*walkthrough*.md` → walkthrough
  - `*plan*.md` → implementation_plan
  - `*bug*.md` → bug_report
  - `*audit*.md` → audit
- File content keywords (first 100 lines):
  - Contains "## Design" → design
  - Contains "## Research" or "## Investigation" → research
  - Contains "## Implementation Steps" → implementation_plan

## Valid Types
- design
- research
- walkthrough
- implementation_plan
- bug_report
- audit

## Execution

1. Verify file exists in `pulse_staging/artifacts/`
2. Determine type (provided or auto-detected)
3. Execute CLI command via Bash tool:
   ```bash
   uv run compass pulse-sync \
     --path "[filename]" \
     --type "[type]"
   ```
4. Confirm registration

## Output

```
✅ Registered: [filename]
   Type: [type] (detected/provided)
   Milestone: [milestone-id]
```
