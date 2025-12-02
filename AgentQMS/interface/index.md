---
title: "Agent Directory Architecture"
date: "2025-11-01T18:00:00Z"
type: "documentation"
category: "architecture"
status: "active"
version: "1.0"
tags: ["architecture", "agent", "documentation"]
---

# Agent Directory Architecture

## Overview

The `agent/` directory is the **Agent-Only Interface Layer** for AI agents. It provides convenience commands and wrappers that make it easy for AI agents to interact with the underlying tool implementations.

## Purpose

- **Agent-Friendly Interface**: Provides simple `make` commands for common tasks
- **Convenience Layer**: Wraps complex tool invocations into simple commands
- **Agent-Only Access**: Designed exclusively for AI agents (humans should use `AgentQMS/agent_tools/` directly)
- **Configuration Hub**: Centralizes agent configuration and tool mappings

## Architecture Relationship

```
agent/ (Interface Layer)
    │
    │ imports/calls
    │
    ▼
AgentQMS/agent_tools/ (Implementation Layer)
```

**Key Principle**: `agent/` is a thin wrapper layer. All actual implementations live in `AgentQMS/agent_tools/`.

## Directory Structure

```
agent/
├── index.md             # This file - explains architecture
├── README.md           # Agent usage guide
├── Makefile            # Agent command interface (main entry point)
├── tools/              # Thin wrapper scripts
│   ├── ast_analysis.py # Wraps AgentQMS agent_tools/automation CLIs
│   ├── discover.py     # Wraps AgentQMS/agent_tools/core/discover.py
│   ├── feedback.py     # Wraps AgentQMS/agent_tools/utilities/agent_feedback.py
│   └── quality.py      # Wraps AgentQMS/agent_tools/compliance/documentation_quality_monitor.py
├── workflows/          # Bash workflow wrappers
│   ├── create-artifact.sh
│   ├── validate.sh
│   └── compliance.sh
├── config/             # Agent configuration
│   ├── agent_config.yaml
│   └── tool_mappings.json
└── logs/               # Agent activity logs
    ├── feedback/
    └── quality/
```

## Usage Patterns

### For AI Agents

**Primary Access Method**: Use `make` commands from `agent/` directory

```bash
cd AgentQMS/agent_interface/
make help              # Show all available commands
make discover          # Discover available tools
make create-plan NAME=my-plan TITLE="My Plan"
make validate          # Validate artifacts
make compliance        # Check compliance
```

**Why use `agent/`?**
- Convenient `make` commands
- Pre-configured paths
- Agent-specific workflows
- Centralized configuration

### For Humans

**DO NOT** use `agent/` directory. Instead:
- Use `scripts/agent_tools/` directly
- Use main project Makefile
- Run Python scripts directly

## Component Responsibilities

### Makefile
- **Purpose**: Primary agent command interface
- **Usage**: `make <command>`
- **Examples**: `make validate`, `make create-plan`, `make compliance`

### tools/
- **Purpose**: Thin Python wrappers that import from `scripts/agent_tools/`
- **Why**: Provides agent-friendly entry points while keeping implementations in one place
- **Pattern**: Each wrapper imports and calls the actual implementation

### workflows/
- **Purpose**: Bash scripts for common workflows
- **Usage**: Called by Makefile or directly (but only from `agent/` directory)
- **Pattern**: Validates directory context, then calls Python scripts

### config/
- **Purpose**: Agent configuration and tool mappings
- **Files**:
  - `agent_config.yaml`: Agent settings and preferences
  - `tool_mappings.json`: Maps tool names to implementation paths

### logs/
- **Purpose**: Stores agent activity logs
- **Structure**: Organized by tool/feature (feedback/, quality/, etc.)

## Design Principles

1. **Thin Wrapper Layer**: `agent/` should be minimal - just convenience wrappers
2. **Single Source of Truth**: All implementations in `scripts/agent_tools/`
3. **Clear Separation**: Agent interface vs implementation
4. **Agent-Only**: Designed for AI agents, not humans

## Adding New Agent Commands

1. **Add Makefile target** in `agent/Makefile`
2. **Call implementation** from `scripts/agent_tools/`
3. **Update tool_mappings.json** if adding new tool mapping
4. **Document** in `agent/README.md`

## Troubleshooting

**Import Errors**: Check that wrapper scripts import from correct paths in `scripts/agent_tools/`

**Command Not Found**: Ensure you're in `agent/` directory when running `make` commands

**Broken References**: Verify paths in `agent/Makefile` and `agent/config/tool_mappings.json` point to correct locations

## Related Documentation

- `scripts/agent_tools/index.md` - Implementation layer architecture
- `agent/README.md` - Agent usage guide
- `scripts/agent_tools/README.md` - Tool implementation guide

---

*This architecture ensures clear separation between agent interface and tool implementations.*

