# .qwen/ Directory - Qwen AI Agent Integration

**Last Updated**: 2026-01-18
**Status**: Active - Configuration for Qwen AI integration

## Purpose

This directory contains configuration files for Qwen AI agent integration with the project's AgentQMS system.

## Directory Schema

```
.qwen/
├── README.md              # This file - configuration guide
├── QWEN.md                # Agent context and quick reference
├── QWEN.yaml              # AI agent configuration manifest
├── settings.json          # Qwen CLI settings
└── archive/               # Archived/deprecated files
```

## Active Configuration

### settings.json
- **Purpose**: Qwen CLI configuration with context bundling integration
- **Key Feature**: Uses context bundling system from AgentQMS
- **Configuration**: Includes context-aware loading and file filtering

### QWEN.yaml
- **Purpose**: AI agent context manifest
- **Content**: Project context, directory structure, task classifications
- **Integration**: Works with context bundling system

### QWEN.md
- **Purpose**: Quick reference for AI agents
- **Content**: Current project structure and workflows

## Context Bundling System

The configuration now integrates with the AgentQMS context bundling system:
- Located at: `AgentQMS/.agentqms/plugins/context_bundles/`
- Contains task-specific file collections for efficient context loading
- Reduces token usage and improves AI agent performance
- Supports various task types: ocr_experiment, documentation_update, pipeline_development, etc.

## Key Features

- **Context-Aware Loading**: Automatically loads relevant files based on task type
- **File Filtering**: Respects .gitignore and .qwenignore patterns
- **MCP Integration**: Unified server combines Project Compass, AgentQMS, and Experiment Manager
- **Workspace Management**: Efficient file inclusion/exclusion patterns

## Notes for AI Agents

The system is configured to automatically detect and load appropriate context bundles based on your task description. The context bundling system will suggest relevant files to include in your working context.

---

**For more details**, see `AgentQMS/knowledge/agent/system.md` for full AgentQMS documentation.
