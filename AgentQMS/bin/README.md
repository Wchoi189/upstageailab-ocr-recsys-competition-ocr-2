# AgentQMS (AI-only)

Concise entrypoint for AgentQMS CLI usage and discovery.

## Read Order
1) AgentQMS/AGENTS.yaml
2) AgentQMS/standards/registry.yaml
3) AgentQMS/standards/tier2-framework/tool-catalog.yaml

## Rules
- AI-facing only. No user docs.
- No manual artifacts. Follow workflow requirements in tier1-sst/constraints/workflow-requirements.yaml.
- Use AgentQMS/bin/aqms for discovery + plugin validation.

## Commands (from AgentQMS/bin)
- ./aqms registry resolve --task <task>
- ./aqms registry resolve --path <path>
- ./aqms registry sync
- ./aqms plugin list
- ./aqms plugin validate

## Notes
- Artifact creation via aqms is not yet implemented; follow workflow requirements and project compass processes.
# Registry Generation (for auto-discovery)
make generate-registry  # Regenerate tool registries and context files

# IDE Configuration (Centralized Settings)
make ide-config         # Generate configuration files for Antigravity, Cursor, Claude, Copilot
```

---

## Framework Contents

### AgentQMS/ (Framework Container)

- `bin/` – Agent interface layer: Makefile commands, CLI entry points
- `tools/` – Implementation layer: core workflows, validation, compliance, documentation automation
- `standards/` – Standards index, tier definitions, tool catalogs, quickstart guides
- `.agentqms/` – Framework plugins and state:
  - `plugins/` – Built-in artifact types, validators, context bundles
  - `schemas/` – Validation schemas (artifact_type_validation.yaml)
- `mcp_server.py` – Model Context Protocol server for AI tool integration
- `AGENTS.yaml` – Quick-reference index for AI agents

### Project-Level Directories

- `.agentqms/` – Project state and configuration:
  - `schemas/` – Centralized validation schemas
  - `context_control/` – Context bundling system state
  - `context_feedback/` – AI feedback on context relevance
- `docs/artifacts/` – QMS artifacts (plans, assessments, audits, bug reports)
- `.github/copilot-instructions.md` – Primary Copilot entrypoint

---

## Installation

### Option A: Install as a Python Package (Recommended)

```bash
# Clone and install in editable mode
git clone https://github.com/your-org/agent_qms.git
cd agent_qms
uv pip install -e .

# Verify installation
uv run python -c "import AgentQMS; print(AgentQMS.__version__)"
```

### Option B: Copy into Your Project

```bash
cp -r AgentQMS/ your_project/
cp -r .agentqms your_project/
mkdir -p your_project/docs/artifacts
```

---

## Plugin System (Extensibility)

AgentQMS supports project-level extensions via plugins. Define custom:

- **Artifact Types** – New document types with custom templates
- **Validators** – Additional validation rules and prefixes
- **Context Bundles** – Task-specific context file collections

### Plugin Directory Structure

```
AgentQMS/.agentqms/plugins/
├── artifact_types/           # Built-in artifact type definitions
│   ├── assessment.yaml
│   ├── audit.yaml
│   ├── bug_report.yaml
│   ├── design_document.yaml
│   ├── implementation_plan.yaml
│   ├── vlm_report.yaml
│   └── walkthrough.yaml
├── validators.yaml           # Validator extensions
└── context_bundles/          # Task-specific context bundles
    ├── agent-configuration.yaml
    ├── compliance-check.yaml
    ├── hydra-configuration.yaml
    └── ocr-*.yaml            # OCR domain bundles
```

### Using Plugins

```bash
# List registered plugins
python -m AgentQMS.tools.core.plugins --list

# Validate plugin definitions
python -m AgentQMS.tools.core.plugins --validate

# Write snapshot (used by make validate)
python -m AgentQMS.tools.core.plugins --write-snapshot

# View specific plugin
python -m AgentQMS.tools.core.plugins --show assessment
```

See `.agentqms/schemas/artifact_type_validation.yaml` for validation rules and canonical types.

---

## High-Level Layout

```text
project_root/
├── AgentQMS/                  # Framework (lightweight container)
│   ├── bin/                   # Agent commands (Makefile)
│   ├── tools/                 # Implementation layer
│   ├── standards/             # Standards, catalogs, quickstart
│   ├── .agentqms/plugins/     # Built-in plugins
│   ├── mcp_server.py          # MCP server
│   └── AGENTS.yaml            # Agent entrypoint index
├── .agentqms/                 # Project state
│   ├── schemas/               # Validation schemas
│   └── context_control/       # Context system state
├── docs/artifacts/            # QMS artifacts
└── .github/copilot-instructions.md  # Copilot entrypoint
```

---

## Key Capabilities

- **Artifact workflows** – `AgentQMS/tools/core/artifact_workflow.py` creates,
  validates, and maintains QMS artifacts.
- **Validation & compliance** – `AgentQMS/tools/compliance/*` enforces naming,
  structure, and boundary rules; integrates with CI and pre-commit hooks.
- **Plugin extensibility** – Define custom artifact types, validators, and context
  bundles in `AgentQMS/.agentqms/plugins/`.
- **MCP integration** – `AgentQMS/mcp_server.py` provides tool access for Claude Desktop
  and compatible AI clients.
- **Context bundling** – Task-specific file collections reduce token usage and improve
  AI context relevance. Powered by **Context Engine 2.0** (caching, parallel I/O, token budgeting).

---

## Maintenance Timestamping (Recommendation)

To keep docs and interfaces fresh and machine-friendly:

- Add a visible `Last Updated: YYYY-MM-DD (KST)` line at the top of key docs (as above).
- In Makefile help, embed `LAST_UPDATED := $(shell TZ=Asia/Seoul date +%Y-%m-%dT%H:%M:%S%z)` and print it in `help` output.
- Include `generated_at` fields in machine-readable YAML (e.g., `.agentqms/state/plugins.yaml`).
- Prefer KST timestamps in ISO-8601 format with timezone (e.g., `2026-01-10T15:20:00+0900`).

## Versioning (Recommendation)

- Use `ads_version` in frontmatter only when backed by a schema definition.
- Schema location: `.agentqms/schemas/artifact_type_validation.yaml` defines canonical types and validation rules.
- Version numbers without schema backing are discouraged; prefer referencing the schema itself.

---

## Contributing

For framework development and contribution guidelines, see [CONTRIBUTING.md](../../CONTRIBUTING.md).

---

## License

This project is licensed under the MIT License – see [LICENSE](../../LICENSE).
