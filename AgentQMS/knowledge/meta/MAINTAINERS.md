---
title: "AgentQMS Maintainers Guide"
audience: maintainer
status: active
version: "0.2"
---

## Purpose

- Provide a concise, container-focused guide for maintaining and exporting AgentQMS.
- Assume the framework is used as a reusable bundle: `.agentqms/` + `AgentQMS/`.

## Key Locations

- `.agentqms/` – runtime state and effective configuration.
- `.agentqms/state/architecture.yaml` – component and capability map (see below).
- `AgentQMS/agent_tools/` – canonical implementation layer (Python tools).
- `AgentQMS/toolkit/` – legacy compatibility shim (delegates to `agent_tools`).
- `AgentQMS/conventions/` – artifact types, schemas, templates, audit framework.
- `AgentQMS/knowledge/` – agent instructions, protocols, references, templates, meta docs.

## Architecture State (`architecture.yaml`)

The file `.agentqms/state/architecture.yaml` is the **authoritative index** for:

- **Paths**: where docs, protocols, references, templates, artifacts, and config live.
- **Knowledge domains**: structured map of `AgentQMS/knowledge/*` contents.
- **Components**: implementation-layer modules and their responsibilities.
- **Capabilities**: high-level features (e.g., plan_generation, quality_assessment) and the tools/artifacts they use.

When adding new protocols, references, or tools:

1. Place the file in the appropriate `AgentQMS/knowledge/*` or `AgentQMS/agent_tools/*` directory.
2. Update `architecture.yaml` to register the new entry under the correct domain or component.
3. Run `make validate` to ensure link and manifest validation passes.

## Export & Adaptation (High Level)

- Copy `.agentqms/` and `AgentQMS/` into the target project.
- Optionally create a project-level `config/` to override framework defaults.
- Run basic checks:
  - `cd AgentQMS/interface && make discover`
  - `make status`
  - `make validate`

## Legacy Project Docs

- Long-form export guides and historical assessments under `docs_deprecated/` are **project history**, not part of the reusable container.
- When packaging AgentQMS for reuse, exclude:
  - `docs_deprecated/` (entire folder)
  - Project-specific artifacts and RFCs.


