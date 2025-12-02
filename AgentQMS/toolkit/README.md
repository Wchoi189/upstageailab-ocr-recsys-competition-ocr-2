---
title: "AgentQMS Toolkit – Legacy Compatibility Layer"
date: "2025-11-25 00:00 (KST)"
type: "documentation"
category: "usage"
status: "deprecated"
version: "2.0"
tags: ["agent-tools", "legacy", "shim"]
---

# AgentQMS Toolkit (Legacy Shim)

> ⚠️ **DEPRECATED**: This directory is a **legacy compatibility layer**.
> All new code and documentation should target `AgentQMS/agent_tools/`.

## Purpose

`AgentQMS/toolkit/` exists only to provide backward compatibility for scripts
and workflows that were written before the canonical implementation layer was
renamed to `AgentQMS/agent_tools/`.

Most modules in this directory are thin shims that delegate to the corresponding
module in `AgentQMS/agent_tools/`.

## Canonical Implementation Layer

For all new work, use:

```bash
# Discover tools
PYTHONPATH=. python AgentQMS/agent_tools/core/discover.py

# Create artifact
PYTHONPATH=. python AgentQMS/agent_tools/core/artifact_workflow.py create \
    --type implementation_plan --name my-plan --title "My Plan"

# Validation
PYTHONPATH=. python AgentQMS/agent_tools/compliance/validate_artifacts.py --all
PYTHONPATH=. python AgentQMS/agent_tools/compliance/validate_boundaries.py --json

# Documentation link validation
PYTHONPATH=. python AgentQMS/agent_tools/documentation/validate_links.py AgentQMS/knowledge
```

## Migration

If you have scripts that import from `AgentQMS.toolkit.*`, update them to import
from `AgentQMS.agent_tools.*` instead. The API is identical.

## Related

- **Canonical layer**: `AgentQMS/agent_tools/`
- **Agent SST**: `AgentQMS/knowledge/agent/system.md`
- **Maintainers guide**: `AgentQMS/knowledge/meta/MAINTAINERS.md`
