
---
title: "AgentQMS Copilot Instructions"
doc_id: "agentqms-copilot-instructions"
status: "active"
last_reviewed: "2026-01-08"
sources:
  - "AgentQMS/standards/INDEX.yaml"
  - "AgentQMS/standards/tier2-framework/tool-catalog.yaml"
  - "AgentQMS/standards/tier2-framework/quickstart.yaml"
  - "AGENTS.yaml"
---

# AgentQMS Copilot Instructions

You are working in an AgentQMS-enabled project.

## Critical Rules
1. **Discovery**: Read `AgentQMS/standards/INDEX.yaml` and `AgentQMS/standards/tier2-framework/tool-catalog.yaml`.
2. **Artifacts**: NEVER create `docs/artifacts/*` files manually.
   - Use: `cd AgentQMS/bin && make create-plan` (or similar)
3. **Safety**: Run `cd AgentQMS/bin && make validate` before asking the user to review.

## Context
- **Standards Index**: `AgentQMS/standards/INDEX.yaml`
- **Tool Catalog**: `AgentQMS/standards/tier2-framework/tool-catalog.yaml`
- **Quickstart**: `AgentQMS/standards/tier2-framework/quickstart.yaml`
- **Workflow Triggers (generator)**: `AgentQMS/tools/core/workflow_detector.py` (generates `.copilot/context/workflow-triggers.yaml` if needed)
