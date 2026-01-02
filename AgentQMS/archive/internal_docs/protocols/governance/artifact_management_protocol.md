---
title: "Artifact Management Protocol"
audience: agent
status: active
domain: governance
id: PROTO-GOV-001
---

## When This Applies

- You are creating or updating any quality-management artifact.
- You are unsure **which artifact type** to use for work.

## Always Use the Workflow Tool

```bash
python -m AgentQMS.agent_tools.core.artifact_workflow create \
  --type <artifact_type> \
  --name "<short-name>" \
  --title "<Human-readable title>"
```

The tool:

- Applies naming conventions.
- Places the file in the correct artifacts subdirectory.
- Fills the template and validates structure.
- Updates indexes and runs compliance checks.

## Choose the Right Artifact Type

- **implementation_plan** – major features or refactors.
- **assessment** – evaluations and assessments of system components.
- **audit** – framework audits, compliance checks, quality evaluations with automated plan generation.
- **design** – architecture and design decisions.
- **bug_report** – bug documentation and resolution tracking.
- **research** – investigations and technical research.

See `artifact_rules.md` for full type list and locations.

## Validate and Check Compliance

```bash
python -m AgentQMS.agent_tools.core.artifact_workflow validate --all
python -m AgentQMS.agent_tools.core.artifact_workflow check-compliance
```

## Related

- `artifact_rules.md` – canonical artifact naming and directory rules.
- `implementation_plan_protocol.md`.
- `bug_fix_protocol.md`.


