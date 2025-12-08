---
title: "AgentQMS Dashboard Features"
type: development
status: complete
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 14:30 (KST)
phase: 1
priority: medium
tags: [development, features, dashboard, capabilities]
---

<div align="center">

# AgentQMS Features

**Created:** 2025-12-08 14:30 (KST) | **Updated:** 2025-12-08 14:30 (KST)

[**README**](../README.md) • [**Roadmap**](./agentqms-roadmap.md) • [**Architecture**](./agentqms-mermaid.md) • [**Features**](./agentqms-features.md) • [**API**](./agentqms-api.md)

</div>

---

## Core Capabilities

### 1. Artifact Standardization
*   **Enforcement**: Strict YAML Frontmatter schema.
*   **Fields**: Mandatory `branch_name` and `timestamp`.
*   **Templates**: Pre-defined structures for Plans, Audits, Reports.

### 2. AI-Powered Auditing
*   **Compliance Check**: Verifies schema integrity.
*   **Quality Analysis**: Scores clarity and completeness.
*   **Multi-Model**: Supports Gemini, GPT-4, and OpenRouter models.

### 3. Traceability Management
*   **Context Explorer**: Visual graph of artifact relationships.
*   **Registry**: JSON-based indexing of all project documentation.
*   **Tracking DB**: Simulation of state tracking for large projects.

### 4. Integration Utilities
*   **Bootstrap**: One-click scaffold generation (`agent_tools/bootstrap.sh`).
*   **System Prompt**: Auto-generated instructions for AI Agents.
*   **Config Portability**: Import/Export JSON settings and `.env` parsing.
