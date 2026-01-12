---
ads_version: "1.0"
type: implementation_plan
category: "development"
status: active
version: "1.0"
title: "Refactor workflow_detector: config + command templates"
date: "2026-01-12 20:15 (KST)"
owner: copilot
---

## Goals
- Remove magic values from workflow detection.
- Centralize config in YAML with safe defaults.
- Keep command templates and suggestions data-driven.

## Scope
- File: AgentQMS/tools/core/workflow_detector.py
- New config: AgentQMS/config/workflow_detector_config.yaml
- No user docs; AI-facing only.

## Steps
1) Config schema
- Define task_types (context_bundle, workflows, tools), artifact_triggers keywords, command_templates.
- Add defaults to keep behavior unchanged.

2) Loader
- Add load_config using ConfigLoader with defaults and validation of required sections.

3) Refactor detection
- Use config task_types instead of WORKFLOW_SUGGESTIONS.
- Use config artifact_triggers instead of ARTIFACT_TRIGGERS.
- Use command_templates for make commands.

4) YAML generation (optional)
- Make generate_workflow_triggers_yaml consume current config and emit yaml without hardcoded paths.

5) Sanity pass
- Quick dry run: suggest_workflows on sample tasks (development, bug, plan) to confirm parity.

## Validation
- Run mypy/pyright if configured (optional).
- Manual: invoke suggest_workflows() on a few strings; ensure no KeyError and outputs include commands.
