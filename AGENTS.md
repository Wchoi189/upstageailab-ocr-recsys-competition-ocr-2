# AGENTS.md — AI Entrypoint (Human Brief)

This file gives humans and AI a concise starting point. Machine-parsable keys stay in AGENTS.yaml (keep it light for memory).

## Use This When
- You need a 90-second orientation before running an agent.
- You want links to standards, context bundles, and safe workflows.
- You need to know which commands to run first (without scanning YAML).

## Quick Start
1. Load context: `aqms context "<what you are doing>"` (or `make context-development` from AgentQMS/bin).
2. Follow standards: AgentQMS/standards/registry.yaml (unified registry) → tool catalog: AgentQMS/standards/tier2-framework/tool-catalog.yaml.
3. Package policy: use `uv` only (no raw `pip`); commands: `uv add`, `uv sync`, `uv run ...`.
4. Artifact workflow: NEVER hand-write docs/artifacts/*. Use `aqms artifact create --type <type> --name <name> --title "<title>"` (auto-validates). Then `aqms validate --all`.
5. Compliance/safety: `aqms monitor --check`; for boundary checks run `make boundary` from AgentQMS/bin.

## Key Entry Points
- Project standards: see INDEX.yaml for the map; naming/workflow rules live under tier1-sst/*.
- Tooling: tier2-framework/tool-catalog.yaml for commands; `cd AgentQMS/bin && make discover` for a live list.
- Context bundles: `make context-list` to inspect; prefer task-specific `make context TASK="..."`.
- Experiment manager (separate system): `python experiment_manager/etk.py <cmd>`; docs in experiment_manager/.ai-instructions/.
- Agent Debug Toolkit (AST/Hydra analysis): location agent-debug-toolkit/, entrypoint `uv run adt ...`; reference AI_USAGE.yaml for tasks.

## Core Packages (AI Hints)
Keep these ultra-concise to avoid context overload. Use links for depth.

- AgentQMS: Standards + artifact tooling to enforce format/placement/naming. Core commands: `cd AgentQMS/bin && make create-*`, `make validate`, `make compliance`. Read the root map in [AgentQMS/standards/INDEX.yaml](AgentQMS/standards/INDEX.yaml).
- project_compass: Sessionized feature workflows and artifact bundling; supports export/clear of active session state. See entrypoint [project_compass/AGENTS.md](project_compass/AGENTS.md).
- experiment_manager: Fast iteration runner for image-processing experiments with schemas/templates. CLI: [experiment_manager/etk.py](experiment_manager/etk.py) (`init`, `reconcile`, `validate`).
- agent-debug-toolkit: AST/Hydra analyzer for config precedence, merges, and instantiation discovery. CLI: `uv run adt ...`. Reference [agent-debug-toolkit/AI_USAGE.yaml](agent-debug-toolkit/AI_USAGE.yaml).

## Operational Rules
- Read INDEX.yaml before assuming standards; do not hallucinate policies.
- Keep AGENTS.yaml minimal and schema-only; update this markdown for human guidance.
- Avoid bulk or destructive changes without validation (`make validate`, `make docs-validate-links`).

If you need machine-readable values, go to AGENTS.yaml. Use this file to onboard fast, choose the right command, and stay within AgentQMS guardrails.
