# Documentation Field Guide

Use this guide as the single entry point for all project documentation. It groups every maintained file by intent, shows where to find historical notes, and highlights the fastest command to surface the right doc when you are in a hurry.

## Quick Reference Matrix

| Category | When to Open | Primary Files | Fast Command |
| --- | --- | --- | --- |
| **Artifacts** | Reviewing assessments, audits, and plans. | `docs/artifacts/` | `ls -R docs/artifacts` |
| **Changelog** | Tracking project changes and version history. | `docs/changelog/` | `ls docs/changelog` |
| **Schemas** | Understanding data contracts and validation rules. | `docs/schemas/` | `ls docs/schemas` |
| **VLM Reports** | Reviewing Vision Language Model evaluation results. | `docs/vlm_reports/` | `ls docs/vlm_reports` |
| **Archive** | Accessing historical code, docs, and legacy UI. | `docs/archive/` | `ls -R docs/archive` |
| **System Source of Truth** | Core instructions and framework guidance. | `.ai-instructions/` | `ls -R .ai-instructions` |

## Artifact Categories

| Category | Description |
| --- | --- |
| **Assessments** | Deep dives into specific system components or issues. |
| **Audits** | Periodic health checks of documentation and code. |
| **Bug Reports** | Detailed logs of identified issues and their status. |
| **Completed Plans** | Historical record of executed implementation plans. |
| **Design Documents** | Architectural specs and feature designs. |
| **Implementation Plans** | Active and pending work roadmaps. |
| **Research** | Exploration of new techniques, models, or paradigms. |

## How to Keep This Page Current

1. **Add new docs here immediately.** Include a one-line "When to open" blurb and the preferred quick command.
2. **Retire stale docs.** Move them to `docs/archive/` or delete if redundant.
3. **Link from tickets or PRs.** Whenever a discussion references documentation, paste the relevant table row or file path for instant context.

## Search Patterns That Save Time

- List everything at depth ≤ 2:
  ```bash
  find docs -maxdepth 2 -type d | sort
  ```
- Jump straight to completed plans:
  ```bash
  ls docs/artifacts/completed_plans
  ```
- Search through all active instructions:
  ```bash
  rg "TODO" .ai-instructions
  ```

## Naming Conventions

- Date-stamp detailed logs as `YYYY-MM-DD_topic.md` (`2025-10-08_convex_hull_debugging.md`).
- Keep living references short and topic-focused (`project-overview.md`, `SETUP.md`).
- Archive experiments or abandoned concepts under `_deprecated/`.

## Archive & Deprecated Content

| Location | Status | Contents |
|----------|--------|----------|
| `docs/archive/` | ⚠️ ARCHIVED | Historical code/docs, do not update |
| `docs/archive/user_docs/` | ⚠️ LEGACY | Former user-facing documentation |
| `.ai-instructions/DEPRECATED/` | ⚠️ DEPRECATED | Legacy agent configs |

> **Note**: Stale references in archived files are NOT corrected.

Keeping this guide up to date lets every agent skip the guesswork and land on the right documentation in a few keystrokes.
