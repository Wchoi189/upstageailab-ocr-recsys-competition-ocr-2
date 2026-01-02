---
title: "Implementation Plan Protocol"
audience: agent
status: active
domain: governance
id: PROTO-GOV-002
---

## When This Applies

- You are planning major feature work, refactors, or architecture changes.
- The work spans multiple sessions, components, or files.

## Create a Plan (Command)

```bash
python -m AgentQMS.agent_tools.core.artifact_workflow create \
  --type implementation_plan \
  --name "my-feature" \
  --title "My Feature Implementation"
```

The workflow tool:

- Names the file correctly.
- Places it under `artifacts/implementation_plans/`.
- Fills the template (progress tracker, sections, frontmatter).

## Plan Structure (Checklist)

Ensure the generated plan includes and maintains:

- **Progress tracker** – STATUS, CURRENT STEP, LAST COMPLETED, NEXT TASK.
- **Phases and tasks** – clear breakdown into actionable steps.
- **Success criteria** – how completion is validated and tested.
- **Source attribution** – links to requirements, specs, and related artifacts.

## Maintain the Plan

- Update progress tracker whenever work state changes.
- Record decisions and deviations from the original plan.
- Link to created artifacts (bug reports, assessments, design docs).

## Validate

```bash
python -m AgentQMS.agent_tools.core.artifact_workflow validate --file <plan-path>
```

## Related

- `artifact_rules.md` – naming and placement rules.
- `bug_fix_protocol.md` – for bug-related implementation work.


