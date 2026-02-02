# Artifact Compliance Specification

**Tier**: 1 (Contracts)
**Scope**: Naming, Placement, and Artifact Definitions.
**Enforcement**: Strict (`validate_artifacts.py`).

## 1. Naming & Placement Rules

All artifacts must reside in `docs/artifacts/{TYPE}/` and follow strict naming.

| Rule | Requirement | Example |
| :--- | :--- | :--- |
| **Format** | `YYYY-MM-DD_HHMM_{TYPE}_{slug}.md` | `2025-11-29_1800_audit_security.md` |
| **Case** | Kebab-case (lowercase, hyphens) | `my-feature-plan` (Not `MyFeature`) |
| **Root** | **PROHIBITED** at `docs/` root. | Must be in `docs/artifacts/templates/` etc. |
| **Exceptions** | `README.md`, `CHANGELOG.md` | Allowed at root. |

## 2. Artifact Types

| TypeKey | Prefix | Directory | Description |
| :--- | :--- | :--- | :--- |
| **implementation_plan** | `implementation_plan_` | `implementation_plans/` | Dev roadmaps |
| **assessment** | `assessment_` | `assessments/` | Technical eval |
| **audit** | `audit_` | `audits/` | Compliance review |
| **design_document** | `design_document_` | `design_documents/` | Architecture decision |
| **research** | `research_` | `research/` | Findings & notes |
| **template** | `template_` | `templates/` | Reusable forms |
| **bug_report** | `bug_` | `bug_reports/` | Issue tracking |
| **session_note** | `SESSION_` | `.../session_notes/` | Handoff summaries |
| **vlm_report** | `vlm_report_` | `vlm_reports/` | Vision model analysis |


## 3. Frontmatter Requirements

All artifacts must start with a YAML frontmatter block.

**Required Fields**:
*   `title`: (String) Human readable title
*   `date`: `YYYY-MM-DD HH:MM (KST)`
*   `type`: Must match artifact type (e.g., `implementation_plan`)
*   `category`: `development`, `architecture`, `evaluation`, `compliance`, `planning`
*   `status`: `active`, `draft`, `completed`, `deprecated`
*   `version`: `1.0`
*   `ads_version`: `2.0` (Required for new artifacts)

## 4. Prohibited Actions

> [!WARNING]
> Violations trigger CI failures.

*   **Manual Creation**: Do not use `touch` or `echo`. Use `make create-{TYPE}`.
*   **Root Placement**: Never put `.md` files in `docs/` root.
*   **Sys.path Hacks**: Do not manually manipulate `sys.path` in scripts.
*   **Verbose Docs**: AI instructions must be machine-readable, not tutorial-style.
