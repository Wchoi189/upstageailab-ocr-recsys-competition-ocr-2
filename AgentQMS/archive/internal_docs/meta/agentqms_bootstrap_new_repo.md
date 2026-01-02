---
title: "AgentQMS Bootstrap Guide – New Repository Integration"
status: "draft"
version: "0.1.0"
updated: "2025-12-02"
---

## Goal

Provide a **repeatable recipe** for integrating AgentQMS into a new repository so that AI agents and humans get the same quality management workflows everywhere.

---

## 1. Required directories and files

From the source project (this repository or a template), copy the following into the new repo:

- `AgentQMS/` – Framework container (interface, agent_tools, conventions, knowledge).
- `.agentqms/` – Project-level state and plugin directory.
- `.copilot/context/` – Auto-discovery context files:
  - `agentqms-overview.md`
  - `tool-registry.json`
  - `tool-catalog.md`
  - `workflow-triggers.yaml`
  - `context-bundles-index.md`
- `.cursor/instructions.md` – Ultra-short instructions for Cursor / IDE agents.

Also ensure the project has:

- `docs/artifacts/` – QMS artifacts (implementation plans, assessments, bug reports, audits).

> Recommended: create the directory even if initially empty so CI and tools behave consistently.

---

## 2. Minimal `.agentqms` configuration

In the new repo, update `.agentqms/settings.yaml` to reflect the project:

- Project name and description.
- Primary components (backend, frontend, data pipeline, etc.).
- Any CI toggles or integration flags specific to the new project.

Verify `.agentqms/state/architecture.yaml` still matches:

- `paths.artifacts_root` → usually `docs/artifacts`.
- `paths.runtime_state` → `.agentqms`.
- `paths.context_bundles` → `AgentQMS/knowledge/context_bundles`.

You can keep the core component map as-is if the AgentQMS layout is unchanged.

---

## 3. GitHub Actions integration

Copy or reuse the AgentQMS workflows:

- `.github/workflows/agentqms-ci.yml`
- `.github/workflows/agentqms-validation.yml`
- `.github/workflows/agentqms-autofix.yml` (optional but recommended)

Adjust only:

- Branch names (e.g., `main` vs `master`) if necessary.
- Python version, if the target repo standard differs.

Do **not** change the core steps that call into `AgentQMS/interface` unless you are deliberately extending the framework.

---

## 4. Root-level Makefile shortcuts (or task runner)

If the new repo uses a `Makefile`, add wrappers similar to:

```make
qms-plan:
	@cd AgentQMS/interface && make create-plan NAME=$(if $(NAME),$(NAME),my-plan) TITLE="$(if $(TITLE),$(TITLE),Implementation Plan)"

qms-bug:
	@cd AgentQMS/interface && make create-bug-report NAME=$(if $(NAME),$(NAME),my-bug) TITLE="$(if $(TITLE),$(TITLE),Bug Report)"

qms-validate:
	@cd AgentQMS/interface && make validate

qms-compliance:
	@cd AgentQMS/interface && make compliance
```

For repositories that use a different task runner (e.g., `just`, `invoke`, `npm scripts`), provide equivalent thin wrappers that still delegate to `AgentQMS/interface`.

---

## 5. Documentation hooks

Update the new repository’s documentation to point at AgentQMS:

- In `README.md`, add a short **“Quality Management with AgentQMS”** section that:
  - Explains that artifacts live in `docs/artifacts/`.
  - Shows the most common `qms-*` commands.
  - Links to `AgentQMS/knowledge/agent/system.md` and `.agentqms/state/architecture.yaml`.

- In `CONTRIBUTING.md`, add a brief subsection under documentation or process:
  - When to create implementation plans vs bug reports vs assessments.
  - How to run `qms-validate` / `qms-compliance` before opening a PR.

---

## 6. Readiness checklist for new repos

Use this checklist when enabling AgentQMS in a new repository:

- [ ] `AgentQMS/` directory present at repo root.
- [ ] `.agentqms/` directory present with `settings.yaml` and `state/architecture.yaml`.
- [ ] `docs/artifacts/` directory exists.
- [ ] `.copilot/context/` contains AgentQMS context files.
- [ ] `.cursor/instructions.md` exists and references AgentQMS SST + architecture.
- [ ] Root task runner (e.g., `Makefile`) exposes `qms-*` shortcuts or equivalents.
- [ ] GitHub Action workflows `agentqms-ci.yml` and `agentqms-validation.yml` enabled.
- [ ] `README.md` documents how to use AgentQMS locally.
- [ ] `CONTRIBUTING.md` references AgentQMS workflows for plans and bug reports.

When all boxes are checked, the repository is **AgentQMS-ready** and should provide a consistent experience for AI agents and human contributors.

