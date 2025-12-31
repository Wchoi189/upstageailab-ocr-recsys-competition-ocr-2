---
type: instruction
category: agent_guidance
status: active
version: "1.0"
title: "AgentQMS Project Instructions"
date: "2025-12-14"
branch: main
---

# AgentQMS Project Instructions

## Core Rules
1. **Source of Truth**: Read `AgentQMS/knowledge/agent/system.md` first.
2. **Tools First**: Do not create artifacts manually. Use the tools.
3. **Validation**: Always run validation commands after making changes.

## Primary Commands
- **Create Plan**: `cd AgentQMS/interface && make create-plan NAME=slug TITLE="Title"`
- **Validate**: `cd AgentQMS/interface && make validate`
- **Status**: `cd AgentQMS/interface && make status`

## Architecture
See `.agentqms/state/architecture.yaml` for component maps.
