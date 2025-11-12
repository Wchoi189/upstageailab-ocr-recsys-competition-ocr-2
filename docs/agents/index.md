# AI Agent Documentation Map

**Purpose:** Quick reference for finding documentation. For detailed index, see `index.json` (auto-generated).

**Project:** OCR Receipt Text Detection
**Status:** Active

## Core Rules

- `system.md` → Single source of truth for AI agent behaviour. **READ FIRST.**
- `index.md` (this file) → Doc map so agents know where to look.
- `index.json` → Auto-generated machine-readable index (use for tooling).

## AgentQMS

- **Artifact Creation**: Use AgentQMS toolbelt programmatically (see `system.md`)
- **Framework Location**: `agent_qms/` (root, for Python imports)
- **Artifact Storage**: `artifacts/` at project root (AgentQMS-managed)
- **Protocols**: See `protocols/governance.md` for artifact management
- **Guide Tool**: `python scripts/agent_tools/core/artifact_guide.py` for format, location, usage

## Protocols

- `protocols/development.md` - Development protocols (coding, debugging, bug fixes)
- `protocols/components.md` - Component protocols (training, Streamlit, preprocessing)
- `protocols/configuration.md` - Configuration protocols (Hydra, command builder)
- `protocols/governance.md` - Governance protocols (artifacts, documentation, bug fixes)

## References

- `references/architecture.md` - System architecture key facts
- `references/commands.md` - Command reference
- `references/tools.md` - Tool reference

## Tracking

- `tracking/cli_reference.md` → Commands, recipes, and troubleshooting for the tracking CLI.
- `tracking/db_api.md` → SQLite tracking DB schema & CRUD API.

## Automation

- `automation/tooling_overview.md` → Human-curated overview of automation scripts.
- `automation/tool_catalog.md` → Generated catalog of all agent tools (read-only).
- `automation/changelog_process.md` → Semi-automated CHANGELOG workflow.

## Coding Protocols

- `coding_protocols/streamlit.md` → Coding rules when modifying Streamlit app code.

## Reserved Domains

- `artifact_workflow/` → Reserve for blueprint/protocol deep dives (create when needed).

## OCR Project Specific

- **Training**: `runners/train.py` with Hydra configs
- **Testing**: `runners/test.py` with checkpoint paths
- **Prediction**: `runners/predict.py` with checkpoint paths
- **UI Tools**: `run_ui.py` for command_builder, inference, evaluation_viewer
- **Tools**: `scripts/agent_tools/` organized by category (core, compliance, documentation, maintenance, ocr, utilities)

## Auto-Generated Index

The `index.json` file is auto-generated from the directory structure:
- **Regenerate**: `python scripts/agent_tools/documentation/auto_generate_index.py`
- **Use for tooling**: Machine-readable index for documentation discovery
- **Human-readable**: This `index.md` file (manually maintained)

## Guidelines

- Add new AI agent docs only if they fill a gap and link them here.
- If content grows beyond a quick reference, move detailed material into `docs/maintainers/` and keep a short pointer here.
- Always use AgentQMS toolbelt for artifact creation (preferred method).
- Use `python scripts/agent_tools/core/artifact_guide.py` for artifact creation guidance.
