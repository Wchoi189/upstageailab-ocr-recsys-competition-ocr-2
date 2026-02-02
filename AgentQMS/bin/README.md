# AgentQMS (AI-only)

Concise entrypoint for registry + plugin + context tooling.

## Read Order
1) AgentQMS/AGENTS.yaml
2) AgentQMS/.agentqms/registry.yaml (auto-generated from specs)

## Rules
- AI-facing only. No user docs.
- No manual artifacts. Follow workflow requirements in specs.

## Core Commands
- ./bin/aqms registry resolve --task <task>
- ./bin/aqms registry resolve --path <path>
- ./bin/aqms registry sync  # Generates registry from specs

## Plugin Snapshot
- uv run python -m AgentQMS.tools.core.plugins --validate --write-snapshot
- Snapshot output: AgentQMS/.agentqms/state/plugins.yaml

## Context Suggestion
- uv run python AgentQMS/tools/core/context/suggest_context.py "<task>"

## Graph Regeneration
- uv run python AgentQMS/tools/generate_mechanized_graph.py

## Middleware Observability (Phase C)
- make qms-middleware-health      # Check middleware health (policy enforcement)
- make qms-middleware-dashboard   # View policy enforcement stats
- make qms-middleware-stats       # Export metrics as JSON
- make qms-middleware-logs        # Show recent logs
- uv run python AgentQMS/tools/middleware/dashboard.py --health-only

## Key Locations
- Framework plugins: AgentQMS/.agentqms/plugins/
- Project plugins: .agentqms/plugins/
- Standards registry: AgentQMS/.agentqms/registry.yaml (auto-generated)
- Standards specs: AgentQMS/specs/ (source of truth)
- Architecture graph: AgentQMS/standards/architecture_map.dot
- Middleware logs: outputs/logs/middleware/
- Middleware metrics: outputs/metrics/
