# Validation & Constraints Specification

**Tier**: 1 (Contracts)
**Scope**: System Policies, Validation Logic, and Frontmatter Schemas.

## 1. Agent Policy
*   **No "I cannot"**: Agents must always attempt tasks or pivot to alternatives.
*   **Strict Mode**: Adhere to `validation_runbook.yaml`.
*   **Autonomy**: Create plans (`project_compass`) before complex execution.

## 2. Validation Policies
| Policy | Constraint |
| :--- | :--- |
| **Strict Schema** | All artifacts MUST pass `validate_artifacts.py`. |
| **No Bloat** | Files > 2000 tokens must be split. |
| **Tier Integrity** | Do not import Tier 4 code into Tier 2. |

## 3. Workflow Requirements
1.  **Artifact Workflow**: `create` -> `validate` -> `commit`.
2.  **Experiment Workflow**: Must use `checkpoints`.
3.  **Validation**: Must pass `pre-commit` hooks.

## 4. Frontmatter Master Schema

### Required Fields
```yaml
title: String
date: YYYY-MM-DD HH:MM (KST)
type: [implementation_plan, assessment, audit, design_document, research, template, bug_report, session_note, vlm_report, guide, walkthrough]
category: [development, architecture, evaluation, compliance, code_quality, reference, planning, research, troubleshooting]
status: [active, draft, completed, archived, deprecated]
version: '1.0'
ads_version: '2.0'
```

### Type Mappings
| Frontmatter Type | Directory Prefix | Description |
| :--- | :--- | :--- |
| `implementation_plan` | `implementation_plans/` | Development roadmaps |
| `assessment` | `assessments/` | Technical evaluation |
| `design_document` | `design_documents/` | Architecture decision |
| `session_note` | `completion_summaries/session_notes/` | Handoff notes |

> [!NOTE]
> See `compliance.spec.md` for naming rules.
