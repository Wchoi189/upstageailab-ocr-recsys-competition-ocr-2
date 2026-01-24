# AGENTS.md — AI Entrypoint (Human Brief)

This file gives humans and AI a concise starting point. Machine-parsable keys stay in AGENTS.yaml (keep it light for memory).

## Use This When
- You need a 90-second orientation before running an agent.
- You want links to standards, context bundles, and safe workflows.
- You need to know which commands to run first (without scanning YAML).

## Quick Start
1. **Entry Point**: Use the `aqms` CLI for all operations. Ensure it's in your PATH or alias it.
   - Example: `aqms context "<task>"`
2. **Context**: `aqms context "<what you are doing>"`
3. **Standards**: [AgentQMS/standards/registry.yaml](AgentQMS/standards/registry.yaml) (Unified Registry) - Single source of truth.
4. **Artifacts**: `aqms artifact create --type <type> --name <name> --title "<title>"` (Auto-validates).
5. **Validation**: `aqms validate --all`.
6. **Infrastructure**:
   - **IACP**: Agents communicate via `IACPEnvelope` over RabbitMQ.
   - **Config**: distributed via Redis (`config:{path}`) with explicit fallbacks.
   - **LLM**: Local inference via `QwenClient` (Lazy Loaded) -> Ollama.

## Key Entry Points
- **CLI**: `aqms` is the unified tool. Run `aqms --help`.
- **Standards**: `registry.yaml` maps Tasks -> Bundles -> Standards.
- **Project Compass**: Vessel-based pulse lifecycle. Artifacts sync to `project_compass/pulse_staging/artifacts/`.
- **Infrastructure**:
  - `ocr/core/infrastructure/agents`: Base agents and capability definitions.
  - `ocr/core/infrastructure/communication`: IACP and Transport layers.

## Core Packages (AI Hints)
- **AgentQMS**: Standards + Tooling. CLI: `aqms`.
- **project_compass**: Lifecycle management. all artifacts sync here.
- **ocr.core**: Core infrastructure (IACP, Agents, Inference).
- **experiment_manager**: Fast iteration runner for image-processing.

## Operational Rules
- **No Legacy Scripts**: Do not use `scripts/aqms.py` or `Makefile` targets directly if `aqms` covers it.
- **IACP Strictness**: All agent communication must use `IACPEnvelope`.
- **Lazy Loading**: Import LLM clients from `ocr.core.infrastructure.agents.llm` to avoid heavy overhead.

If you need machine-readable values, go to AGENTS.yaml. Use this file to onboard fast, choose the right command, and stay within AgentQMS guardrails.
