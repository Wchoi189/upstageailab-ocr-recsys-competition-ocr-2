---
ads_version: '2.0'
id: 'SC-010'
type: 'rule_set'
tier: 1
priority: 'critical'
updated: '2026-02-03'
description: 'Required workflow policies for artifact creation and validation'
dependencies:
  - naming-conventions.yaml
  - file-placement-rules.yaml
---

# Workflow Requirements

> Required workflow policies for artifact creation and validation

## Specification

```yaml
agent: all
dependencies:
- naming-conventions.yaml
- file-placement-rules.yaml
artifact_creation_policy:
  command_base: cd AgentQMS/bin && make create-{TYPE}
  args_required:
  - NAME: lowercase-slug-format
  - TITLE: quoted-string
  available_types:
  - create-plan
  - create-assessment
  - create-design
  - create-research
  - create-audit
  - create-bug-report
  enforcement: Manual creation is prohibited
frontmatter_requirements:
  required_fields:
  - type
  - category
  - status
  - version
  - tags
  - title
  - date
  - branch
  date_format: YYYY-MM-DD HH:MM (KST)
  status_values:
  - active
  - completed
  - archived
  - deprecated
  auto_generated: Via make create-* commands
blueprint_protocol:
  applies_to: implementation_plan artifacts
  template: AgentQMS/knowledge/templates/blueprint_protocol_template.md
  components:
  - Master Prompt
  - Living Implementation Blueprint
  - Progress Tracker
  - Goal-Execute-Update Loop
  auto_applied: true
runbook_reference: AgentQMS/specs/tier4-workflows/validation.spec.md

```
