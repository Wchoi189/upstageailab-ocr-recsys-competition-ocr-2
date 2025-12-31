# AgentQMS Project Instructions

## Core Rules
1. **Source of Truth**: Read `AgentQMS/docs/agent/system.md` first.
2. **Tools First**: Do not create artifacts manually. Use the tools.
3. **Validation**: Always run validation commands after making changes.

## Primary Commands
- **Create Plan**: `cd AgentQMS/bin && make create-plan NAME=slug TITLE="Title"`
- **Validate**: `cd AgentQMS/bin && make validate`
- **Status**: `cd AgentQMS/bin && make status`

## Architecture
See `AgentQMS/state/architecture.yaml` for component maps.
