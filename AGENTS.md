# AI Agent Entrypoint

> **Context Map Location**: `.ai-instructions/INDEX.yaml`

This project uses the **Agentic Documentation Standard (ADS) v1.0**.

## 🗺️ Context Map
The source of truth for all project rules, architecture, and standards is located in:
**`.ai-instructions/INDEX.yaml`**

## ⚡ Quick Links
- **Workflow Rules**: `.ai-instructions/tier1-sst/workflow-requirements.yaml`
- **Artifact Standards**: `.ai-instructions/tier1-sst/artifact-types.yaml`
- **System Architecture**: `.ai-instructions/tier1-sst/system-architecture.yaml`
- **Tools Catalog**: `.ai-instructions/tier2-framework/tool-catalog.yaml`
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
   **Best Practice**: Use the Makefile in `AgentQMS/interface/`.
   ```bash
   cd AgentQMS/interface
   make help          # List all tools
   make validate      # Validate artifacts
   make create-plan NAME=my-plan TITLE="My Plan"
   ```

   **Direct Access**:
   - `python -m AgentQMS.agent_tools.compliance.validate_artifacts`
   - `python -m AgentQMS.agent_tools.compliance.validate_boundaries`

## 🤖 Tier 3 Agent Configs
- **Gemini**: `.ai-instructions/tier3-agents/gemini/config.yaml`
- **Claude**: `.ai-instructions/tier3-agents/claude/config.yaml`
- **Cursor**: `.ai-instructions/tier3-agents/cursor/config.yaml`
- **Copilot**: `.ai-instructions/tier3-agents/copilot/config.yaml`

Do not hallucinate project standards. Read the `INDEX.yaml` first.
