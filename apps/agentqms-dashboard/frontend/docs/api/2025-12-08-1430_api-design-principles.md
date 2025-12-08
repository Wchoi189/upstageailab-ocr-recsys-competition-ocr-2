---
title: "AgentQMS API Design Principles"
type: api
status: complete
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 14:30 (KST)
phase: 1
priority: medium
tags: [api, design-principles, rest, documentation]
---

<div align="center">

# AgentQMS Internal API

**Created:** 2025-12-08 14:30 (KST) | **Updated:** 2025-12-08 14:30 (KST)

[**README**](../README.md) • [**Roadmap**](./agentqms-roadmap.md) • [**Architecture**](./agentqms-mermaid.md) • [**Features**](./agentqms-features.md) • [**API**](./agentqms-api.md)

</div>

---

## 1. Frontend Services (`aiService.ts`)

### `generateContent(prompt, config)`
*   **Purpose**: Main gateway to AI providers.
*   **Args**:
    *   `prompt` (string): Input text.
    *   `config` (obj): `{ systemInstruction, jsonMode }`.
*   **Returns**: `Promise<string>` (Raw text or JSON string).

### `auditDocumentation(content, type)`
*   **Purpose**: Runs compliance audit against content.
*   **Returns**: `AuditResponse` object (`score`, `issues[]`, `recommendations[]`).

## 2. CLI Tools (`agent_tools/`)

### `validate_frontmatter.py`
*   **Usage**: `python validate_frontmatter.py --file <path>`
*   **Exit Codes**:
    *   `0`: Valid.
    *   `1`: Missing Fields.
    *   `2`: Invalid Format (Timestamp/Branch).

### `structure_check.py`
*   **Usage**: `python structure_check.py --config .agentqms/config.json`
*   **Purpose**: Verifies folder hierarchy matches `config.json` definition.
