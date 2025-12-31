# AI Agent Entrypoint

> **Context Map Location**: `AgentQMS/standards/INDEX.yaml`

This project uses the **Agentic Documentation Standard (ADS) v1.0**.

## 🗺️ Context Map
The source of truth for all project rules, architecture, and standards is located in:
**`AgentQMS/standards/INDEX.yaml`**

## ⚡ Quick Links
- **Workflow Rules**: `AgentQMS/standards/tier1-sst/workflow-requirements.yaml`
- **Artifact Standards**: `AgentQMS/standards/tier1-sst/artifact-types.yaml`
- **System Architecture**: `AgentQMS/standards/tier1-sst/system-architecture.yaml`
- **Tools Catalog**: `AgentQMS/standards/tier2-framework/tool-catalog.yaml`
- **Experiment Manager (ETK)**: `experiment_manager/etk.py` (CLI Entrypoint)
    - **Guides**: `experiment_manager/.ai-instructions/`
    - **Key Commands**: `init`, `reconcile`, `validate`

## 🛠️ Environment & Tools
Before starting work, ensure the environment is healthy:
1. **Validate Environment**:
   ```bash
   python scripts/validate_environment.py
   ```
   (Run this after `uv sync`)

2. **AgentQMS Tools**:
   Located in `AgentQMS/`, these tools help with validation and planning.
   **Best Practice**: Use the Makefile in `AgentQMS/bin/`.
   ```bash
   cd AgentQMS/bin
   make help          # List all tools
   make validate      # Validate artifacts
   make create-plan NAME=my-plan TITLE="My Plan"
   ```

   **Direct Access**:
   - `python -m AgentQMS.tools.compliance.validate_artifacts`
   - `python -m AgentQMS.tools.compliance.validate_boundaries`

## 🤖 Tier 3 Agent Configs
- **Gemini**: `AgentQMS/standards/tier3-agents/gemini/config.yaml`
- **Claude**: `AgentQMS/standards/tier3-agents/claude/config.yaml`
- **Cursor**: `AgentQMS/standards/tier3-agents/cursor/config.yaml`
- **Copilot**: `AgentQMS/standards/tier3-agents/copilot/config.yaml`

Do not hallucinate project standards. Read the `INDEX.yaml` first.
