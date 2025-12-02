# Context Bundles Index

Auto-generated index of available context bundles for task-specific context loading.

## Overview

Context bundles provide task-specific file collections that help AI agents focus on relevant documentation and code. Bundles are automatically suggested based on task keywords.

## Available Bundles

### development

**Purpose**: Code implementation, features, and development tasks

**When to use**: 
- Implementing new features
- Writing code
- Refactoring
- Adding functionality

**Make command**: `cd AgentQMS/interface && make context-development`

**Files included**:
- Essential: `AgentQMS/knowledge/agent/system.md`, `README.md`, `pyproject.toml`
- Architecture: `.agentqms/state/architecture.yaml`, framework design docs
- Standards: Coding standards, import protocols, artifact rules

### documentation

**Purpose**: Writing and updating documentation

**When to use**:
- Writing docs
- Updating README
- Creating guides or tutorials

**Make command**: `cd AgentQMS/interface && make context-docs`

**Files included**:
- Documentation structure and templates
- Documentation protocols
- Index generation tools

### debugging

**Purpose**: Troubleshooting issues and fixing bugs

**When to use**:
- Debugging errors
- Investigating issues
- Fixing broken functionality

**Make command**: `cd AgentQMS/interface && make context-debug`

**Files included**:
- Error handling patterns
- Troubleshooting guides
- Log analysis tools

### planning

**Purpose**: Design and implementation planning

**When to use**:
- Creating implementation plans
- Designing architecture
- Strategic planning

**Make command**: `cd AgentQMS/interface && make context-plan`

**Files included**:
- Planning protocols
- Design templates
- Architecture references

### general

**Purpose**: Default fallback bundle for general tasks

**When to use**: When task type cannot be determined

**Files included**:
- Basic framework overview
- Essential system files

## Auto-Detection

Context bundles are automatically suggested based on task keywords:

- **Development keywords**: implement, code, develop, feature, refactor, build, create
- **Documentation keywords**: document, doc, write docs, readme, guide, tutorial
- **Debugging keywords**: debug, troubleshoot, error, fix, broken, issue, crash
- **Planning keywords**: plan, design, architecture, blueprint, strategy, assess

## Usage

### Via Makefile

```bash
cd AgentQMS/interface
make context-development    # Load development context
make context-docs          # Load documentation context
make context-debug         # Load debugging context
make context-plan          # Load planning context
```

### Via Python

```python
from AgentQMS.agent_tools.core.context_bundle import get_context_bundle, auto_suggest_context

# Auto-detect and get context
files = get_context_bundle("implement new feature")

# Get suggestions with workflows
suggestions = auto_suggest_context("implement new feature")
print(suggestions['context_bundle'])  # 'development'
print(suggestions['bundle_files'])    # List of file paths
```

### Via CLI

```bash
PYTHONPATH=. python AgentQMS/agent_tools/core/context_bundle.py --task "implement feature"
PYTHONPATH=. python AgentQMS/agent_tools/core/context_bundle.py --auto --task "fix bug"
```

## Bundle Structure

Each bundle is defined in `AgentQMS/knowledge/context_bundles/*.yaml` with:

- **tiers**: Organized file collections (tier1 = essential, tier2 = architecture, tier3 = standards)
- **max_files**: Limit on number of files per tier
- **priority**: File importance (critical, high, medium, low)
- **description**: Why each file is included

## Custom Bundles

Project-specific bundles can be added in `.agentqms/plugins/context_bundles/` and will be automatically discovered.

