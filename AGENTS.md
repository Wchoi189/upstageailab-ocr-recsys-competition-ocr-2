# AGENTS.md — AI Entrypoint (Concise)

AI-only quick start. For machine-readable values use AGENTS.yaml.

## Read Order
1) AgentQMS/AGENTS.yaml
2) AgentQMS/.agentqms/registry.yaml (auto-generated from specs)
3) AgentQMS/specs/ (source of truth)

## Commands (Canonical CLI)

### Global (when installed)
- `aqms init` — Initialize `.agentqms/` scaffolding in current directory
- `aqms status` — Show resolved project/framework roots

### Module (always available)
- `python -m AgentQMS.cli registry resolve --task <task>`
- `python -m AgentQMS.cli registry resolve --path <path>`
- `python -m AgentQMS.cli registry sync`  # Generates registry from specs
- `python -m AgentQMS.cli plugin validate`

### Environment Variables
- `AGENTQMS_PROJECT_ROOT` — Override project root detection (absolute or relative path)

## Middleware Observability (Phase C)
- make qms-middleware-health  # Check policy enforcement
- make qms-middleware-dashboard  # View stats
- make qms-middleware-logs  # Recent logs

## Rules
- No manual artifacts. Follow AgentQMS/specs/ specifications.
- AI-facing only; keep outputs concise.
- Use AgentQMS.tools.utils.paths for paths.
- Logs output to outputs/logs/middleware/
