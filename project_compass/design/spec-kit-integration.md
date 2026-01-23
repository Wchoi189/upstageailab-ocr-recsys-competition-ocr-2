# Spec Kit Integration Guide

## Overview

Project Compass v2 now integrates with GitHub Spec Kit to provide spec-driven development capabilities. This integration allows you to create specifications, plans, and tasks as part of your pulse workflow.

## Architecture

The integration consists of:

1. **MCP Tools**: `spec_constitution`, `spec_specify`, `spec_plan`, `spec_tasks`
2. **CLI Commands**: `spec-constitution`, `spec-specify`, `spec-plan`, `spec-tasks`
3. **Artifact Types**: `specification`, `requirements`, `architecture`
4. **Workflow Integration**: Spec artifacts are managed through pulse staging

## Workflow Integration

### 1. Establish Constitution (Project Setup)

```bash
# CLI
uv run compass spec-constitution "Focus on code quality, testing standards, user experience consistency, and performance requirements"

# MCP Tool: spec_constitution
{
  "principles": "Focus on code quality, testing standards, user experience consistency, and performance requirements"
}
```

### 2. Create Specification (During Pulse)

```bash
# CLI
uv run compass spec-specify --scope "OCR text recognition pipeline" --requirements "accuracy >95%, latency <100ms"

# MCP Tool: spec_specify
{
  "scope": "OCR text recognition pipeline",
  "requirements": "accuracy >95%, latency <100ms"
}
```

### 3. Generate Implementation Plan

```bash
# CLI
uv run compass spec-plan --approach "modular architecture with ML pipeline"

# MCP Tool: spec_plan
{
  "approach": "modular architecture with ML pipeline"
}
```

### 4. Create Actionable Tasks

```bash
# CLI
uv run compass spec-tasks --focus "model training pipeline"

# MCP Tool: spec_tasks
{
  "focus_area": "model training pipeline"
}
```

## Pulse Integration

Spec artifacts are registered as pulse artifacts:

```bash
# Register spec as artifact
uv run compass pulse-sync --path specs/ocr-requirements.md --type specification

# Register architecture as artifact
uv run compass pulse-sync --path design/architecture.md --type architecture
```

## MCP Resources

The integration adds spec-related resources:

- `vessel://state` - Includes spec status in pulse state
- `vessel://rules` - Spec-driven rules injected into active pulses
- `vessel://staging` - Lists spec artifacts in staging

## Best Practices

1. **Constitution First**: Establish project principles before starting pulses
2. **Spec During Pulse**: Create specifications as part of pulse objectives
3. **Register Artifacts**: Always register spec artifacts with `pulse-sync`
4. **Token Management**: Specs help manage token burden by providing clear scope
5. **Export with Specs**: Include spec artifacts when exporting pulses

## Dependencies

Requires GitHub Spec Kit CLI:

```bash
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
```

## Troubleshooting

- **Command not found**: Ensure `specify` CLI is installed
- **Permission errors**: Check file permissions in pulse_staging/artifacts/
- **MCP errors**: Verify pulse is active before using spec tools
