---
title: "AgentQMS System Architecture Diagrams"
type: architecture
status: complete
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 14:30 (KST)
phase: 1
priority: high
tags: [architecture, diagrams, mermaid, system-design]
---

<div align="center">

# AgentQMS Architecture

**Created:** 2025-12-08 14:30 (KST) | **Updated:** 2025-12-08 14:30 (KST)

[**README**](../README.md) • [**Roadmap**](./agentqms-roadmap.md) • [**Architecture**](./agentqms-mermaid.md) • [**Features**](./agentqms-features.md) • [**API**](./agentqms-api.md)

</div>

---

## 1. System Context

```mermaid
graph TD
    User[Developer / Manager] -->|Interacts via| UI[AgentQMS Dashboard]
    UI -->|Generates| Artifacts[Markdown Artifacts]
    UI -->|Configures| Settings[.agentqms/config.json]

    subgraph "AgentQMS Framework"
        Tools[agent_tools/*.py]
        Registry[registry/*.json]
        Templates[templates/*.md]
    end

    subgraph "AI Layer"
        Gemini[Google Gemini]
        OpenAI[OpenAI / OpenRouter]
    end

    UI -->|Audits via| AI Layer
    Tools -->|Validates| Artifacts
    Tools -->|Updates| Registry
```

## 2. Artifact Lifecycle

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant UI as Dashboard
    participant AI as AI Auditor
    participant FS as File System

    Dev->>UI: Input Requirements
    UI->>AI: Generate Schema-Compliant Artifact
    AI-->>UI: Return Frontmatter + Body
    UI->>FS: Save .md File (Draft)

    Dev->>UI: Request Audit
    UI->>AI: Analyze Consistency
    AI-->>UI: Return Score & Issues

    Dev->>FS: Commit to Git (Approved)
```
