---
ads_version: "2.0"
type: agent_instructions
agent: copilot
priority: critical
memory_footprint: 25
updated: "2026-02-12"
status: active
---

# Copilot Instructions (AI-only)

AgentQMS is an AI-native quality management framework that keeps work spec-driven, context-efficient, and continuously validated. It aims to reduce token waste, enforce consistent standards, and make large-codebase navigation reliable for agents.

## Read Order
1) AGENTS.yaml
2) AGENTS.md
3) AgentQMS/AGENTS.yaml
4) AgentQMS/specs/

## Non-Negotiables
- Use `aqms` from PATH. Never `./aqms`.
- No manual artifacts. Specs and registry are the source of truth.
- Use `uv run` for Python.
- Load only required context (no global dumps).
- Write documentation for ai only and keep it concise.

## Constitution Summary
- Specs are the source of truth; code follows specs.
- Keep components small and modular.
- Load only the context you need.
- Validate changes against tier1 contracts.

## AQMS CLI (Framework Ops)
- Purpose: Resolve specs, validate artifacts, and keep standards in sync.
- Docs: AgentQMS/AGENTS.yaml, AgentQMS/specs/
- Examples: `aqms registry resolve --task <task>`, `aqms artifact validate --all`

## Context Bundling (Engine 2.0)
- Purpose: discover and load only relevant files for a task.
- Docs: AgentQMS/AGENTS.yaml
- Examples: `uv run python AgentQMS/tools/utilities/suggest_context.py "<task>"`, `uv run python AgentQMS/tools/utilities/context_inspector.py --list`

## MCP Unified Server (Context Bundling)
- Purpose: access context bundles and standards via MCP resources/tools.
- Docs: AgentQMS/mcp_server.py, AgentQMS/mcp_schema.yaml
- Examples: resource `agentqms://context/bundles`, resource `agentqms://context/bundle/{name}`, tool `get_context_bundle`.

## Project Compass (Vessel V2)
- Purpose: manage work cycles (pulses) and artifacts.
- Docs: dev_tools/project_compass/AGENTS.md
- Examples: `compass pulse-init --id <domain-action-target> --obj <objective> --milestone <id>`, `compass pulse-sync --path <file> --type <type>`
- Write scope: `pulse_staging/artifacts/` only.

## Experiment Manager
- Purpose: initialize, reconcile, and validate experiment state.
- Docs: dev_tools/experiment_manager/.ai-instructions/
- Examples: `etk init`, `etk validate`

## Agent Debug Toolkit (ADT)
- Purpose: AST-based analysis for Hydra/OmegaConf configuration issues.
- Docs: dev_tools/agent_debug_toolkit/
- Examples: `uv run adt analyze-config <path>`, `uv run adt trace-merges <file> --output markdown`

## Spec-Kit Planning (GitHub)
- Purpose: structured spec, plan, and task generation for complex work.
- Use agents: `/speckit.specify`, `/speckit.clarify`, `/speckit.plan`, `/speckit.tasks`, `/speckit.implement`
- Docs: .github/agents/

## Documentation Architecture
- Tree: AgentQMS/specs/ (tier1-contracts, tier2-framework, tier3-implementation).
- Project docs: docs/architecture/, docs/guides/, docs/planning/, docs/reports/.
- Examples: configs/README.md, docker/README.md, AgentQMS/ARCHITECTURE.md.

## Artifacts and Plugins
- Artifacts: standardized outputs defined by specs; create/validate via `aqms artifact ...`.
- Plugins: framework in AgentQMS/.agentqms/plugins/ and project in .agentqms/plugins/ (e.g., context bundles).

## Feedback Loop
- Proactively flag redundant tools, broken context bundling, or workflow friction.
- Use `aqms feedback report --issue-type <type> --description "<desc>"` or tell the user directly.

## Quick Links
- Specs: AgentQMS/specs/
- Middleware logs: outputs/logs/middleware/
- Middleware metrics: outputs/metrics/
