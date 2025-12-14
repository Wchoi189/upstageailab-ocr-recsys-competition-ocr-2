---
type: instruction
category: agent_guidance
status: active
version: "1.0"
title: "Validate Artifacts Workflow"
date: "2025-12-14"
branch: main
description: Validate all artifacts using AgentQMS compliance tools
---
1. Run the validation suite:
   ```bash
   cd AgentQMS/interface && make validate
   ```
2. If errors are reported, fix them or run `make compliance` for more details.
