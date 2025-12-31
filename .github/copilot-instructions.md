# AgentQMS Copilot Instructions

You are working in an AgentQMS-enabled project.

## Critical Rules
1. **Discovery**: Read `.copilot/context/agentqms-overview.md` and `.copilot/context/tool-catalog.md`.
2. **Artifacts**: NEVER create `docs/artifacts/*` files manually.
   - Use: `cd AgentQMS/bin && make create-plan` (or similar)
3. **Safety**: Run `make validate` before asking the user to review.

## Context
- Tool Registry: `.copilot/context/tool-registry.json`
- Workflows: `.copilot/context/workflow-triggers.yaml`
