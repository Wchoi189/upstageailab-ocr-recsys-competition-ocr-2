# AGENTS.md — AI Entrypoint (Concise)

AI-only quick start. For machine-readable values use AGENTS.yaml.

## Read Order
1) AgentQMS/AGENTS.yaml
2) AgentQMS/.agentqms/registry.yaml (auto-generated from specs)
3) AgentQMS/specs/ (source of truth)

## Commands (AgentQMS/bin)
- ./aqms registry resolve --task <task>
- ./aqms registry resolve --path <path>
- ./aqms registry sync  # Generates registry from specs
- ./aqms plugin validate

## Middleware Observability (Phase C)
- make qms-middleware-health  # Check policy enforcement
- make qms-middleware-dashboard  # View stats
- make qms-middleware-logs  # Recent logs

## Rules
- No manual artifacts. Follow AgentQMS/specs/ specifications.
- AI-facing only; keep outputs concise.
- Use AgentQMS.tools.utils.paths for paths.
- Logs output to outputs/logs/middleware/
