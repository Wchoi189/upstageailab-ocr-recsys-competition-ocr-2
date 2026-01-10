
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

You are working in an environment governed by **AgentQMS**—a comprehensive Quality Management System that transforms ad-hoc development into engineering excellence through tiered architectural standards and automated compliance validation. This framework empowers you to deliver high-reliability software by enforcing structured planning, execution, and verification across the entire project lifecycle.

## Critical Rules
1. **Discovery**: Read `AgentQMS/standards/INDEX.yaml` and `AgentQMS/standards/tier2-framework/tool-catalog.yaml`.
2. **Artifacts**: NEVER create `docs/artifacts/*` files manually.
   - Use: `cd AgentQMS/bin && make create-plan` (or similar)
3. **Safety**: Run `cd AgentQMS/bin && make validate` before asking the user to review.

## Context
- **Standards Index**: `AgentQMS/standards/INDEX.yaml`
- **Tool Catalog**: `AgentQMS/standards/tier2-framework/tool-catalog.yaml`
- **Quickstart**: `AgentQMS/standards/tier2-framework/quickstart.yaml`
- **Project Compass**: `project_compass/AGENTS.md`
- **Experiment Manager**: `experiment_manager/agent_interface.yaml`
- **Agent Debug Toolkit**: `agent-debug-toolkit/AI_USAGE.yaml`
- **Workflow Triggers (generator)**: `AgentQMS/tools/core/workflow_detector.py` (generates `.copilot/context/workflow-triggers.yaml` if needed)
