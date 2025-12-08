<div align="center">

# AgentQMS Framework

**Created:** 2024-05-22 15:30 (KST) | **Updated:** 2024-05-22 15:30 (KST)

[**README**](./README.md) • [**Roadmap**](./docs/agentqms-roadmap.md) • [**Architecture**](./docs/agentqms-mermaid.md) • [**Features**](./docs/agentqms-features.md) • [**API**](./docs/agentqms-api.md)

</div>

---

## 1. Directive for AI Agents
**Role:** You are an AgentQMS-compliant engineer.
**Prime Directive:** NEVER write code without an approved Artifact (Plan/Spec).
**Constraint:** All documentation MUST adhere to strict Schema Validation.

## 2. Directory Structure (Canonical)
```text
PROJECT_ROOT/
├── .agentqms/                 # [LOCAL] Config & State (GitIgnored)
│   └── config.json            # Active Module, Timezone
├── AgentQMS/                  # [FRAMEWORK] Immutable Core
│   ├── agent_tools/           # Python Scripts (Audit, Gen)
│   ├── registry/              # JSON Indexes (Trace Matrix)
│   └── templates/             # Markdown Templates
└── src/                       # Source Code
```

## 3. Mandatory Frontmatter Schema
All Artifacts (`.md`) must begin with:
```yaml
---
title: "String"
type: "Assessment | ImplementationPlan | BugReport"
status: "draft | review | approved"
branch_name: "feature/your-branch-name"  # CRITICAL: Must match git branch
created_at: "YYYY-MM-DD HH:MM (TIMEZONE)"
last_updated: "YYYY-MM-DD HH:MM (TIMEZONE)"
tags: [array, of, strings]
---
```

## 4. Operational Protocols
1.  **Bootstrap**: Run `AgentQMS/agent_tools/bootstrap.sh` on new install.
2.  **Audit**: Run `python AgentQMS/agent_tools/audit/validate_frontmatter.py` before commit.
3.  **Trace**: Ensure every `ImplementationPlan` links to a `Requirement` or `BugReport`.
