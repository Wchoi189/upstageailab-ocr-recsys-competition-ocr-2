---
type: instruction
category: agent_guidance
status: active
version: "1.0"
title: "AgentQMS Status Check Workflow"
date: "2025-12-14"
branch: main
description: Check the status of the AgentQMS framework and available tools
---
// turbo
1. Check framework status:
   ```bash
   cd AgentQMS/interface && make status
   ```
2. Discover available tools:
   ```bash
   cd AgentQMS/interface && make discover
   ```
