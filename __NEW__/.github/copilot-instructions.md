---
title: "AgentQMS Copilot Instructions"
updated: "2025-11-27 15:10 UTC"
---

# AgentQMS Framework - Copilot Instructions

This project uses **AgentQMS** (Quality Management Framework) for AI-assisted development workflows.

## Quick Start

**READ FIRST**: `AgentQMS/knowledge/agent/system.md` - This is the Single Source of Truth (SST) for all agent operations.

## Framework Overview

AgentQMS provides standardized workflows for:
- **Artifact Creation**: Implementation plans, assessments, designs, bug reports
- **Validation & Compliance**: Automated artifact validation and boundary checking
- **Context Loading**: Task-specific context bundles for focused work
- **Documentation Management**: Automated index generation and link validation

## Key Entry Points

1. **System Instructions**: `AgentQMS/knowledge/agent/system.md` (REQUIRED READING)
2. **Architecture Map**: `.agentqms/state/architecture.yaml` (component and capability map)
3. **Tool Catalog**: `.copilot/context/tool-catalog.md` (available tools)
4. **Workflow Triggers**: `.copilot/context/workflow-triggers.yaml` (task → workflow mapping)

## Essential Rules

- **Always use automation tools**; never create artifacts manually
- **Artifact naming**: `YYYY-MM-DD_HHMM_[type]_descriptive-name.md`
- **Validation**: Run `cd AgentQMS/interface && make validate` after changes
- **Context bundles**: Auto-detected based on task keywords (see `.copilot/context/context-bundles-index.md`)

## Common Workflows

### Creating Artifacts

```bash
cd AgentQMS/interface
make create-plan NAME=feature-name TITLE="Feature Title"
make create-assessment NAME=assessment-name TITLE="Assessment Title"
make create-bug-report NAME=bug-name TITLE="Bug Description"
```

### Validation

```bash
cd AgentQMS/interface
make validate          # Validate all artifacts
make compliance        # Full compliance check
make boundary          # Boundary validation
```

### Context Loading

Context bundles are automatically suggested based on task keywords. Available bundles:
- `development` - Code implementation tasks
- `documentation` - Writing/updating docs
- `debugging` - Troubleshooting issues
- `planning` - Design and planning tasks

See `.copilot/context/context-bundles-index.md` for details.

## Tool Discovery

All available tools are registered in:
- `.copilot/context/tool-registry.json` (machine-readable)
- `.copilot/context/tool-catalog.md` (human-readable)

Tools are organized by category: core, compliance, documentation, utilities, audit.

## Auto-Discovery

This framework supports automatic discovery:
- **Tool registry**: Automatically generated from `AgentQMS/agent_tools/`
- **Workflow suggestions**: Automatically triggered based on task keywords
- **Context bundles**: Automatically suggested based on task type
- **Auto-validation**: Runs automatically after artifact creation

## When to Use AgentQMS

- Creating implementation plans or design documents
- Documenting bugs or assessments
- Validating artifact compliance
- Loading task-specific context
- Managing documentation indexes

## Reference

- Framework README: `README.md`
- System SST: `AgentQMS/knowledge/agent/system.md`
- Architecture: `.agentqms/state/architecture.yaml`
- Tool Catalog: `AgentQMS/knowledge/agent/tool_catalog.md`

