# AgentQMS Framework

Last Updated: 2026-01-10 (KST)

AgentQMS is a reusable Quality Management Framework that standardizes
artifact creation, documentation workflows, and automation for collaborative
AI coding. The framework is **containerized** so it can travel between projects
as a pair of directories: `.agentqms/` + `AgentQMS/`.

---

## 🤖 For AI Agents: Getting Started

### Auto-Discovery (GitHub Copilot Spaces)

**AgentQMS is automatically discoverable in GitHub Copilot Spaces!** No manual instructions needed.

The framework provides auto-discovery files that Copilot automatically reads:
- **`.github/copilot-instructions.md`** – Primary entry point directing Copilot to framework
- **`.copilot/context/`** – Auto-scanned context files:
  - `agentqms-overview.md` – Framework overview
  - `tool-registry.json` – Machine-readable tool registry
  - `tool-catalog.md` – Human-readable tool catalog
  - `workflow-triggers.yaml` – Task → workflow mapping
  - `context-bundles-index.md` – Available context bundles

**Features**:
- ✅ **Auto-discovery**: Tools and workflows automatically registered
- ✅ **Context-aware suggestions**: Detects task type and suggests relevant tools/context
- ✅ **Workflow automation**: Auto-executes validation after artifact creation
- ✅ **Proactive guidance**: Suggests next steps based on current task

### Cursor AI Instructions

Cursor doesn't automatically read `.copilot/context/`, so a dedicated ultra-short instruction file is provided for Cursor agents:

- File: `.cursor/instructions.md`
- Usage: pin or paste into Cursor's Custom Instructions so every session knows to follow the AgentQMS SST, use automation, run validation, and load context via `make context`.

This keeps Cursor aligned with the same rules Copilot uses while staying within Cursor's tighter instruction window.

### First Contact: What to Read

When an AI agent encounters a project using AgentQMS, these are the **entry points** in priority order:

| Priority | File                                                   | Purpose                                                 |
| -------- | ------------------------------------------------------ | ------------------------------------------------------- |
| 1️⃣        | `AgentQMS/AGENTS.yaml`                                 | **Single Source of Truth** – Agent entrypoint and index |
| 2️⃣        | `AgentQMS/.agentqms/state/plugins.yaml`                | Component map, capabilities, tool locations             |
| 3️⃣        | `AgentQMS/standards/tier2-framework/tool-catalog.yaml` | Complete tool catalog                                   |
| 4️⃣        | This README                                            | Framework overview and installation                     |

### Quick Onboarding Prompt

Copy this prompt to quickly orient an AI agent to this framework:

```
You are working in a project that uses AgentQMS for quality management.

FIRST: Read these files to understand the framework:
1. AgentQMS/AGENTS.yaml (Main Index - REQUIRED)
2. AgentQMS/.agentqms/state/plugins.yaml (Component Map)

KEY RULES:
- Use automation tools; never create artifacts manually
- Run `cd AgentQMS/bin && make help` to see available commands
- Artifacts go in docs/artifacts/ with proper naming: YYYY-MM-DD_HHMM_[type]_name.md
- Validate changes: `make validate` and `make compliance`

When creating implementation plans, assessments, audits, or bug reports, use:
  cd AgentQMS/bin && make create-plan NAME=my-plan TITLE="My Title"
```

### Encouraging Proactive Use

To make the AI agent **proactively** use AgentQMS, include these instructions in your system prompt or project rules:

```
QUALITY MANAGEMENT RULES:
1. Before starting any significant task, check if an implementation plan exists
2. For multi-step work, create an implementation plan first:
   cd AgentQMS/bin && make create-plan NAME=feature-name TITLE="Feature Title"
3. After completing work, run validation:
   cd AgentQMS/bin && make validate && make compliance
4. Document bugs using the bug report workflow, not ad-hoc notes
5. When stuck, run `make discover` to see available tools
```

### Agent Interface Commands

All agent commands are run from `AgentQMS/bin/`:

```bash
cd AgentQMS/bin

# Discovery & Status
make help              # Show all available commands
make discover          # List available tools
make status            # Framework status check

# Artifact Creation
make create-plan NAME=my-plan TITLE="My Plan"
make create-assessment NAME=my-assessment TITLE="My Assessment"
make create-audit NAME=my-audit TITLE="My Audit"
make create-bug-report NAME=my-bug TITLE="Bug Description"

# Validation
make validate          # Validate all artifacts
make compliance        # Full compliance check
make boundary          # Boundary validation

# Context Loading (for focused work)
make context TASK="task description"  # Generate context bundle for specific task
make context-development    # Load development context bundle
make context-docs           # Load documentation context bundle
make context-debug          # Load debugging context bundle
make context-plan           # Load planning context bundle

# Plugin Management
# Note: Plugin commands use Python module syntax (see Plugin System section below)

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
