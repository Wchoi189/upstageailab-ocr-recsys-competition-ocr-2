---
title: "AI Documentation and Scripts Cleanup"
author: "ai-agent"
date: "2025-11-11"
timestamp: "2025-11-11 23:43 KST"
status: "draft"
tags: []
---

## Progress Tracker
*(Required for iterative assessments, debugging sessions, or incremental work)*

- **STATUS:** Not Started / In Progress / Completed
- **CURRENT STEP:** [Current phase or task being worked on]
- **LAST COMPLETED TASK:** [Description of last completed task]
- **NEXT TASK:** [Description of the immediate next task]

### Assessment Checklist
- [ ] Initial assessment complete
- [ ] Analysis phase complete
- [ ] Recommendations documented
- [ ] Review and validation complete

---

## 1. Summary

## 2. Assessment

## 3. Recommendations
## Overview

The documentation and scripting layout still carries multiple legacy surfaces alongside the newer AgentQMS-driven structure. This assessment highlights key friction points for AI agents and enumerates candidates for pruning or re-architecture.

## Documentation Findings

### Split knowledge bases
- Legacy `docs/ai_handbook/**` remains partially populated (e.g. onboarding, experiments, planning), while rules now live under `docs/agents/**`. The duplicated onboarding and planning material produces conflicting guidance.
- `docs/_deprecated/proposed structure.md` is the only file in `_deprecated/`, signalling prior cleanup efforts but leaving the directory as noise.
- `docs/maintainers/**` mirrors many planning and changelog entries that overlap with AgentQMS artifacts. Several files (session handovers, logs) should either graduate into curated artifacts or move to `docs/archive/`.

### Candidate removals / archives
- Archive remaining `docs/ai_handbook/**` content or convert essential pieces into AgentQMS artifacts (e.g. onboarding quick reference) to remove the mixed terminology.
- Drop the lone `_deprecated` file or merge its guidance into `docs/agents/protocols/governance.md`.
- Review `docs/maintainers/experiments/**` and `docs/sessions/2025-10-21/**`; many are historical snapshots better suited for `docs/archive/` once distilled into assessments.
- `docs/performance/` and `docs/project/` contain point-in-time writeups that predate the new tooling; consider migrating to artifacts or tagging as archived resources.

## Scripts Findings

### Fragmented directories
- New tooling resides under `scripts/agent_tools/**`, but legacy directories such as `scripts/data_processing/`, `scripts/analysis_validation/`, `scripts/performance_benchmarking/`, `scripts/migration_refactoring/`, and `scripts/seroost/**` still exist. They often duplicate functionality now covered by the agent_tool suite.
- Root-level helper scripts (`scripts/validate_links.py`, `scripts/standardize_content.py`, `scripts/check_freshness.py`) overlap with the new documentation utilities providing the same behaviors.
- The `scripts/validation/**` tree duplicates compliance checks already encapsulated in `scripts/agent_tools/validation` modules.

### Candidate actions
- Consolidate all documentation-quality commands into `scripts/agent_tools/documentation/` and either delete or wrap the root-level duplicates.
- Evaluate `scripts/data_processing/**` versus `scripts/data/**`; the new layout suggests the old tree can be archived after verifying no remaining call sites.
- Review `scripts/migration_refactoring/` and `scripts/seroost/**` for current relevance; if tied to completed projects, relocate them to an `scripts/archive/` namespace with a README.
- Fold bespoke validation scripts into `agent_tools` to present a single command surface for AI workflows.

## Architectural Recommendations

1. **Single Source Documentation**: Retire `docs/ai_handbook` by converting any must-keep references into AgentQMS assessments or guides. Update navigation in `docs/agents/index.md` once finalized.
2. **Archive Historical Logs**: Create `docs/archive/2025/` (or similar) and move session handovers, raw experiment logs, and migration diaries there after summarizing key learnings in artifacts.
3. **Normalize Script Entry Points**: Adopt an `scripts/agent_tools/__main__.py` or CLI wrapper that enumerates supported commands, guiding AI agents away from deprecated paths.
4. **Introduce Deprecation Map**: Ship a markdown matrix (e.g. `docs/agents/tracking/deprecations.md`) documenting legacy -> replacement scripts/docs so future cleanups become mechanical.

## Next Steps

- Decide which remaining ai_handbook pages warrant preservation and schedule their conversion to AgentQMS artifacts.
- Run usage scans (ripgrep for import/call references) to confirm legacy directories are unused before deletion.
- After pruning, regenerate indexes with `python scripts/agent_tools/documentation/update_artifact_indexes.py` and re-run link validation.
- Communicate the new command surface to collaborators, updating `AGENT_ENTRY.md` and any setup docs accordingly.

---

## Progress Tracker

- Completed
  - Migrated AgentQMS toolbelt and `docs/agents/**`
  - Pruned and archived legacy docs; removed `docs/ai_handbook/**`, `docs/_deprecated/**`, old `docs/performance/**`, `docs/project/**`
  - Implemented link pruning and fixed broken links after deletions
  - Reorganized `scripts/`:
    - Deprecated root duplicates; moved troubleshooting/test utilities to `scripts/troubleshooting/`
    - Moved `check_training_data.py` to `scripts/data/`
    - Added unified CLI `python -m scripts.agent_tools ...`
  - Seeded `docs/artifacts/MASTER_INDEX.md` and category indexes; updater now passes
  - Wired CLI usage into `AGENT_ENTRY.md`
  - Imported implementation plans into `artifacts/implementation_plans/**` and refreshed indexes

- In Scope (remaining)
  - Optional: add a short section in `docs/agents/index.md` linking to `docs/sitemap.md`
  - Optional: retire any lingering ad-hoc validation scripts if discovered later

- Out of Scope (future)
  - Deeper code-level refactors referenced by imported implementation plans (training stabilization, import-time optimizations, inference service consolidation)
