# AgentQMS Scripts Directory

> ⚠️ **NOTE**: This directory contains project-specific scripts that are **not part of the reusable AgentQMS framework**.

## Contents

- `legacy/` - Legacy adaptation scripts (project-specific)
- `maintenance/` - Project-specific maintenance utilities

## For Framework Reuse

When exporting AgentQMS to a new project, this `scripts/` directory should be:
1. **Excluded** from the export, or
2. **Emptied** of project-specific content

The reusable framework components are in:
- `AgentQMS/agent_tools/` - Canonical implementation layer
- `AgentQMS/conventions/` - Artifact types, schemas, templates
- `AgentQMS/knowledge/` - Protocols and references
- `AgentQMS/interface/` - Agent interface layer

## Project-Specific Scripts

If your project needs maintenance scripts, place them at:
- `scripts/` (project root level, outside AgentQMS/)
- Or a project-specific location of your choice

Do not add project-specific code to the AgentQMS container.

