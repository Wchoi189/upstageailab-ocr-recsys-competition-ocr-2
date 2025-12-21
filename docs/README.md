# Documentation Field Guide

Use this guide as the single entry point for all project documentation. It groups every maintained file by intent, shows where to find historical notes, and highlights the fastest command to surface the right doc when you are in a hurry.

## Quick Reference Matrix

| Category | When to Open | Primary Files | Fast Command |
| --- | --- | --- | --- |
| **Project overview** | Need the competition story, deliverables, or high-level architecture. | `docs/project/project-overview.md` | `sed -n '1,80p' docs/project/project-overview.md` |
| **Performance logs** | Reviewing historical runs, regression investigations, or debugging diary notes. | `docs/performance/2025-10-08_convex_hull_debugging.md`, `docs/performance/baseline_2025-10-07*.md`, `docs/performance/baseline_2025-10-08_cache_optimized.md` | `ls docs/performance` |
| **Pipeline contracts** | Understanding data shapes, debugging type errors, or preventing shape mismatches. | `docs/pipeline/data_contracts.md`, `docs/testing/pipeline_validation.md`, `docs/troubleshooting/shape_issues.md` | `python scripts/validate_pipeline_contracts.py` |
| **AI agents** | Need agent operating procedures and concise instructions. | `AGENT_ENTRY.md` (root), `AgentQMS/knowledge/agent/system.md`, `AgentQMS/knowledge/protocols/`, `AgentQMS/knowledge/references/` | `rg --files AgentQMS/knowledge/agent` |
| **Experiment tracking** | Managing experiments, artifacts, and state files safely. | `experiment-tracker/docs/`, `experiment-tracker/README.md` | `ls experiment-tracker/docs` |
| **Maintainers** | Need detailed documentation, architecture, experiments. | `docs/maintainers/` | `rg --files docs/maintainers` |
| **Setup** | Provisioning a new environment or sharing shell helpers. | `docs/setup/SETUP.md`, `docs/setup/setup-uv-env.sh`, `docs/setup/BASH_ALIASES_KO.md` | `ls docs/setup` |
| **Deprecated ideas** | Searching for older proposals that were parked or replaced. | Historical docs removed - see current docs in `AgentQMS/knowledge/` and `docs/maintainers/` | N/A |

## How to Keep This Page Current

1. **Add new docs here immediately.** Include a one-line "When to open" blurb and the preferred quick command.
2. **Retire stale docs.** Remove them completely - no deprecated directories.
3. **Link from tickets or PRs.** Whenever a discussion references documentation, paste the relevant table row or file path for instant context.

## Search Patterns That Save Time

- List everything at depth ≤ 2 (default morning ritual):
  ```bash
  find docs -maxdepth 2 -type f | sort
  ```
- Jump straight to performance notes from the command line:
  ```bash
  rg "convex" docs/performance
  ```
- Preview the first section of any doc without opening an editor:
  ```bash
  sed -n '1,40p' AgentQMS/knowledge/agent/system.md
  ```

## Naming Conventions

- Date-stamp detailed logs as `YYYY-MM-DD_topic.md` (`2025-10-08_convex_hull_debugging.md`).
- Keep living references short and topic-focused (`project-overview.md`, `SETUP.md`).
- Archive experiments or abandoned concepts under `_deprecated/`.

## Archive & Deprecated Content

| Location | Status | Contents |
|----------|--------|----------|
| `docs/archive/` | ⚠️ DEPRECATED | 1,040 files - historical code/docs, do not update |
| `.ai-instructions/DEPRECATED/` | ⚠️ DEPRECATED | Legacy agent configs, superseded by ADS v1.0 |
| `.ai-instructions/tier1-sst/` | ✅ ACTIVE | System source of truth (ADS v1.0) |
| `.ai-instructions/tier2-framework/` | ✅ ACTIVE | Framework guidance (ADS v1.0) |

> **Note**: Stale references (port 8000, old paths) in archived files are NOT corrected.
> See [archive/README.md](archive/README.md) for migration pointers.

Keeping this guide up to date lets every agent skip the guesswork and land on the right documentation in a few keystrokes.
