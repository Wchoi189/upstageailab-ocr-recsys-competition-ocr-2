---
ads_version: "1.0"
type: "walkthrough"
category: "documentation"
status: "active"
version: "1.0"
tags: ['walkthrough', 'documentation', 'guide']
title: "AgentQMS Template Simplification"
date: "2026-01-03 17:11 (KST)"
branch: "main"
description: "Walkthrough of the simplified artifact templates."
---

# Walkthrough - AgentQMS Template Simplification

## Goal
Validate the simplification of AgentQMS artifact templates and the addition of the new `walkthrough` type.

## Steps
1.  **Refactor Templates**: Replaced verbose "Master Prompt" templates in `AgentQMS/tools/core/artifact_templates.py` with concise Markdown.
    -   Types: `assessment`, `design`, `research`, `bug_report`, `template`, `vlm_report`.
2.  **Add Subtypes**: Added `implementation_plan_walkthrough`, `implementation_plan_retroactive`, and `walkthrough`.
3.  **Sync Standards**: Updated `AgentQMS/standards/templates/*.md` to match the python definitions.
4.  **Verify Walkthrough**: Generated this artifact using `create_artifact('walkthrough', ...)`.

## Verification
- [x] Template generation succeeds (this file exists).
- [x] Content is concise and readable.
- [x] Frontmatter contains correct `type: walkthrough`.
