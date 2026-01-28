# AgentQMS (AI-only)

Concise entrypoint for registry + plugin + context tooling.

## Read Order
1) AgentQMS/AGENTS.yaml
2) AgentQMS/standards/registry.yaml

## Rules
- AI-facing only. No user docs.
- No manual artifacts. Follow workflow requirements in tier1-sst/constraints/workflow-requirements.yaml.

## Core Commands
- ./bin/aqms registry resolve --task <task>
- ./bin/aqms registry resolve --path <path>
- ./bin/aqms registry sync

## Plugin Snapshot
- uv run python -m AgentQMS.tools.core.plugins --validate --write-snapshot
- Snapshot output: AgentQMS/.agentqms/state/plugins.yaml

## Context Suggestion
- uv run python AgentQMS/tools/core/context/suggest_context.py "<task>"

## Graph Regeneration
- uv run python AgentQMS/tools/generate_mechanized_graph.py

## Key Locations
- Framework plugins: AgentQMS/.agentqms/plugins/
- Project plugins: .agentqms/plugins/
- Standards registry: AgentQMS/standards/registry.yaml
- Architecture graph: AgentQMS/standards/architecture_map.dot
