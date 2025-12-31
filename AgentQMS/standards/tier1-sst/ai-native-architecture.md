# AI-Native Architecture

## Core Philosophy
**AI agents should never have to "discover" how to work.**
The system must be **Schema-First** and **Method-Locked**.

### 1. Schema-First Configuration
- **No Markdown for Specs**: Logic, contracts, and configurations must be in YAML/JSON. Markdown is for humans; schemas are for agents.
- **Strict Typing**: All configuration files must adhere to a defined schema (e.g., in `.ai-instructions/schema/`).
- **Example**: `ocr-components` logic is defined in YAML, allowing agents to ingest rules programmatically (e.g., `inverse-mapping` rule in `coordinate-transforms.yaml`).

### 2. Method-Locked Execution
- **Glob Pattern Association**: Agents do not guess which instructions apply to a file.
- **INDEX.yaml**: The single source of truth. It maps glob patterns (`ocr/inference/**/*.py`) to specific instruction files (`inference-framework.yaml`).
- **Enforcement**: Agents MUST check `INDEX.yaml` before modifying any file to load the relevant context.

### 3. Usage Standards
- **Naming**: Kebab-case for all files (`configuration-standards.yaml`), avoiding arbitrary underscores.
- **Organization**:
    - **Tier 1 (SST)**: Single Source of Truth (Architecture, Naming).
    - **Tier 2 (Framework)**: Domain logic (OCR components, Data Contracts).
    - **Tier 3 (Agents)**: tool-specific configs (Claude, Gemini).
    - **Tier 4 (Workflows)**: Compliance, Scripts.

### Implementation Status
- [x] `ocr-components` converted to YAML.
- [x] Glob patterns defined in `INDEX.yaml`.
- [x] Misplaced configurations (`gap-analysis`, etc.) moved/renamed.
