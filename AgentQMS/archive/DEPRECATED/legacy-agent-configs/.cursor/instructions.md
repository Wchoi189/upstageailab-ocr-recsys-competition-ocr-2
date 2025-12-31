---
title: AgentQMS – Cursor Instructions
generated_by: AgentQMS/agent_tools/utilities/generate_ide_configs.py
---

## Core Rules (AgentQMS)

1. **Source of Truth**: Read `AgentQMS/knowledge/agent/system.md` first.
2. **No Manual Artifacts**: Always use `cd AgentQMS/interface && make create-*`.
3. **Validation**: Run `make validate` after edits.
4. **Context**: Use `make context` to load relevant docs.

## Workflows
- Create Plan: `make create-plan NAME=foo TITLE="Bar"`
- Validate: `make validate`
- Status: `make status`

For full details, run `make help`.
