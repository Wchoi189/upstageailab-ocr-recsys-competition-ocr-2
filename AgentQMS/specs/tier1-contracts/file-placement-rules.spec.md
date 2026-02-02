---
ads_version: '2.0'
id: 'SC-004'
type: 'rule_set'
tier: 1
priority: 'critical'
updated: '2026-02-03'
dependencies:
---

# File Placement Rules

## Specification

```yaml
agent: all
depends_on:
- naming-conventions.yaml
placement_rules:
  artifacts_directory:
    base: docs/artifacts/
    rule: ALL artifacts MUST be in type-specific subdirectories
    prohibited: docs/*.md (except README.md)
    enforcement: make validate + pre-commit hooks
  type_locations:
    implementation_plan: docs/artifacts/implementation_plans/
    assessment: docs/artifacts/assessments/
    audit: docs/artifacts/audits/
    design: docs/artifacts/design_documents/
    research: docs/artifacts/research/
    template: docs/artifacts/templates/
    bug_report: docs/artifacts/bug_reports/
    session_note: docs/artifacts/completed_plans/completion_summaries/session_notes/
  root_exceptions:
    allowed_at_root:
    - README.md
    - CHANGELOG.md
    - CONTRIBUTING.md
    - LICENSE
    all_others: PROHIBITED at docs/ root
  knowledge_base:
    location: AgentQMS/standards/
    purpose: Long-form guidance (not for agents)
    structure:
    - agent/ - Agent instruction references
    - protocols/ - Development protocols
    - references/ - Technical references
    - templates/ - Document templates
prohibited_actions:
- action: Creating files at docs/ root
  severity: critical
  instead: Use docs/artifacts/{TYPE}/
- action: Manual artifact creation
  severity: critical
  instead: 'Use: cd AgentQMS/bin && make create-{TYPE}'
validation_command: make validate
dependencies: []

```