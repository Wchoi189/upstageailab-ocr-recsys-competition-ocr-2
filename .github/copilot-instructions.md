---
ads_version: "1.0"
type: agent_instructions
agent: copilot
tier: 1
priority: critical
memory_footprint: 40
date: "2026-01-12"
status: active
---

> [!IMPORTANT]
> Read utilities://quick-reference first.

# Copilot Runbook (AI-only)

## Essentials
- Registry: AgentQMS/standards/registry.yaml
- Tool catalog: AgentQMS/standards/tier2-framework/tool-catalog.yaml
- No manual artifacts. Follow tier1-sst/constraints/workflow-requirements.yaml.

## Commands (AgentQMS/bin)
- ./aqms registry resolve --task <task>
- ./aqms registry resolve --path <path>
- ./aqms registry sync
- ./aqms plugin validate

## Rules
- AI-facing only.
- No hardcoded paths (use AgentQMS.tools.utils.paths).
- Load YAML via ConfigLoader.

## Workflow
1) Resolve standards via registry.
2) Execute with tooling.
3) Validate plugins when standards change.
