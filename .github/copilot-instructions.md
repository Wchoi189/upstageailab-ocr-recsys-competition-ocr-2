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
> Read `utilities://quick-reference` first (cached loaders, standard paths).

# Copilot Runbook (Concise)

## Essentials
- System: AgentQMS. Artifacts live in docs/artifacts/.
- Standards map: AgentQMS/standards/INDEX.yaml.
- Prefer AgentQMS/bin/make; fallback `uv run` when needed.

## Commands
- Context: `make context TASK="..."`
- Create artifact: `make create-plan` / `create-assessment` / `create-design`
- Validate: `make validate`
- Compliance: `make compliance`
- Discover tools: `make discover`

## Rules
- Do not write docs/artifacts manually; use make create-*.
- No hardcoded paths; use AgentQMS.tools.utils.paths.
- No user docs; AI-facing only.
- Load YAML via ConfigLoader; use KST helpers from timestamps utils; git info via git utils.
- Run `make validate` before finishing; add `make compliance` when touching artifacts.

## Resource Pointers
- Tool catalog: AgentQMS/standards/tier2-framework/tool-catalog.yaml
- Naming/placement: AgentQMS/standards/tier1-sst/naming-conventions.yaml and file-placement-rules.yaml
- Context bundles: context/utility-scripts/utility-scripts-index.yaml; `make context-list`
- Project guide: AGENTS.md

## Package Hints
- AgentQMS: standards + artifact tooling; entrypoint commands in AgentQMS/bin/make.
- project_compass: workflow/session bundles; see project_compass/AGENTS.md.
- experiment_manager: image experimentation CLI at experiment_manager/etk.py.
- agent-debug-toolkit: AST/Hydra analysis via `uv run adt ...`.

## Workflow
1) Load context (`make context TASK="..."`).
2) Plan if needed (make create-plan for implementation plan).
3) Execute using utilities (paths, ConfigLoader, timestamps, git).
4) Validate (`make validate`; add `make compliance` when artifacts change).
5) Summarize changes briefly.
