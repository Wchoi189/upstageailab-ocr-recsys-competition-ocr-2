# AGENTS.md — AI Entrypoint (Concise)

AI-only quick start. For machine-readable values use AGENTS.yaml.

## Read Order
1) AgentQMS/AGENTS.yaml
2) AgentQMS/standards/registry.yaml
3) AgentQMS/standards/tier2-framework/tool-catalog.yaml

## Commands (AgentQMS/bin)
- ./aqms registry resolve --task <task>
- ./aqms registry resolve --path <path>
- ./aqms registry sync
- ./aqms plugin validate

## Rules
- No manual artifacts. Follow tier1-sst/constraints/workflow-requirements.yaml.
- AI-facing only; keep outputs concise.
- Use AgentQMS.tools.utils.paths for paths.
