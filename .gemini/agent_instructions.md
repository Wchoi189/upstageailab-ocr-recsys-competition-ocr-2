# Custom Agent Instructions

## 🚨 CRITICAL: STRICT PROTOCOL ADHERENCE REQUIRED

**Context**: The OCR project is a massive, error-prone codebase.
**Rule**: **NO TRIAL AND ERROR.** It is too risky and resource-intensive here.

### 1. Mandatory Documentation References
Before writing code, especially in the OCR pipeline or image processing logic (VLM, etc.), you **MUST** reference:
- **Data Contracts**: Explicitly check for data schemas and contract definitions.
- **Architectural Docs**: Review `AgentQMS/state/architecture.yaml` and relevant `docs/` or `AgentQMS/knowledge/` files.

### 2. Artifact Standards
All work must be documented via standardized artifacts.
- **Do not create ad-hoc files.**
- Follow the `AgentQMS` artifact workflow (see `agent_usage_guide.md`).

### 3. Safety First
- **Verify inputs/outputs** against known contracts.
- **If unsure, STOP and check documentation.** Do not guess.
