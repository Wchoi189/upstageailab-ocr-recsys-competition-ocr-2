
---
title: "AgentQMS Dashboard Development Roadmap"
type: plan
status: in-progress
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 14:30 (KST)
phase: 2
priority: high
tags: [plan, roadmap, phases, milestones, timeline]
---

<div align="center">

# AgentQMS Roadmap

**Created:** 2025-12-08 14:30 (KST) | **Updated:** 2025-12-08 14:30 (KST)

[**README**](../README.md) • [**Roadmap**](./agentqms-roadmap.md) • [**Architecture**](./agentqms-mermaid.md) • [**Features**](./agentqms-features.md) • [**API**](./agentqms-api.md)

</div>

---

## 1. Phase 1: Foundation (COMPLETED)
- [x] **Schema Enforcement**: `branch_name` and `timestamp` validation logic.
- [x] **Frontend Dashboard**: React App with Artifact Generator & Basic Auditor.
- [x] **Multi-Provider AI**: Configurable support for OpenAI/OpenRouter & Gemini.
- [x] **Settings & State**: Import/Export capabilities for Configuration and `.env`.
- [x] **Configuration Centralization**: Refactor hardcoded strings to `config/constants.ts`.

## 2. Phase 2: Integration (CURRENT PRIORITY)
**Goal:** Break the "Air Gap" between the Browser Dashboard and the Local File System.

- [ ] **Data Contracts**: Define JSON schemas for File/Tool APIs. (Completed in `docs/DATA_CONTRACTS.md`)
- [ ] **Python Bridge Backend**:
    - [ ] Create `server.py` using FastAPI.
    - [ ] Implement File System (FS) Readers/Writers.
    - [ ] Implement `subprocess` wrapper for executing `agent_tools/*.py`.
- [ ] **Dashboard Integration**:
    - [ ] Update `aiService` or create `bridgeService` to consume `localhost:8000`.
    - [ ] Replace "Mock Data" in `IntegrationHub` with real DB status.

## 3. Phase 3: Traceability (NEXT)
**Goal:** Automate the relationships between requirements and code.

- [ ] **Context Graph**: Visualizing artifact dependencies (Plan -> Code) using real data.
- [ ] **Git Hooks**: Pre-commit hooks to block non-compliant artifacts.
- [ ] **Auto-Indexing**: Script to auto-update `registry/global_index.json` on file change.

## 4. Phase 4: Automation (FUTURE)
- [ ] **Agent Auto-Pilot**: Agents capable of self-correcting Frontmatter.
- [ ] **CI/CD Integration**: Github Action for AgentQMS auditing.
- [ ] **Vector Database**: Embedding artifacts for semantic search.
