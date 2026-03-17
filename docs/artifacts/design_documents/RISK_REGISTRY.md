---
type: design_document
title: "Risk Registry — Global Stateless AgentQMS (Spec 004)"
date: 2026-03-17
category: architecture
status: active
version: 1.0
tags: risk-registry, agentqms, refactoring, spec-004
---

# Risk Registry — Global Stateless AgentQMS

## Purpose

Pre-implementation risk ledger for Spec 004 (Global Stateless AgentQMS). Focuses on **Orphaned Features**, **Cascading Regressions**, and **Integration Debt** risks that the current Speckit plan does not adequately address.

Generated from AST analysis (`adt analyze-dependencies`, `adt analyze-complexity`, `adt analyze-imports`) and manual codebase audit on 2026-03-17.

---

## Risk Summary

| Severity | Open | Resolved | Category |
|----------|------|----------|----------|
| Critical | 1    | 2        | Cascading Regression (R-01, R-03 resolved) |
| High     | 1    | 4        | Orphaned Features / Broken Imports (R-04, R-05, R-07 resolved) |
| Medium   | 3    | 2        | Redundancy / Confusion (R-09 resolved, R-11 fully resolved) |
| Low      | 1    | 1        | Cosmetic / Technical Debt (R-15 resolved) |

---

## Critical Risks — Cascading Regression

### R-01: Four Competing Root-Detection Mechanisms (+ unified_server.py) — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-01 |
| **Category** | Cascading Regression |
| **Impact** | ~~Critical~~ → Resolved |
| **Affected Files** | `config.py`, `mcp_server.py`, `cli.py`, `bin/aqms`, `scripts/mcp/unified_server.py` |
| **Description** | Root resolution has been centralized through the canonical resolver (`AgentQMS.tools.utils.paths.get_project_root()`) and wired into the affected entry points. |
| **Evidence** | `mcp_server.py::find_project_root()`, `cli.py`, and `bin/aqms` now delegate to canonical path utilities. Smoke scenarios for env override and marker traversal pass, and `scripts/mcp/unified_server.py` resolves correctly under `AGENTQMS_PROJECT_ROOT`. |
| **Resolution** | Completed in Spec A execution: replaced local root-finder hacks and validated convergence across entry points. |
| **Trigger** | Keep monitor query: `rg "parents\\[2\\]|parent\\.parent|def find_project_root\\(" AgentQMS/` should only show canonical delegations and no hardcoded traversal hacks. |

### R-02: Two Incompatible `ConfigLoader` Classes

| Field | Value |
|:---|:---|
| **Risk ID** | R-02 |
| **Category** | Cascading Regression |
| **Impact** | Critical |
| **Affected Files** | `tools/utils/config/config.py`, `tools/utils/config/loader.py` |
| **Description** | Two classes named `ConfigLoader` still coexist (`config.py` canonical resolver, `loader.py` YAML/Redis utility), but package-level ambiguity has been reduced. |
| **Evidence** | `tools/utils/config/__init__.py` now exists and explicitly exports canonical `ConfigLoader`/`load_config` and exposes utility loader via `YamlCacheLoader` alias. |
| **Mitigation** | Keep follow-up: rename `loader.py::ConfigLoader` to `YamlCacheLoader` (with compatibility alias) and migrate direct `...config.loader import ConfigLoader` call sites. |
| **Trigger** | `rg "from AgentQMS.tools.utils.config" --type py` — verify all imports resolve unambiguously. |

### R-03: Singleton `get_config_loader()` Caches Stale Root on Re-entry — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-03 |
| **Category** | Cascading Regression |
| **Impact** | ~~Critical~~ → Resolved |
| **Affected Files** | `tools/utils/config/config.py:231-239` |
| **Description** | `get_config_loader()` is a module-level singleton. Once instantiated, `project_root` and `framework_root` are frozen. If `AGENTQMS_PROJECT_ROOT` changes (e.g., in tests or multi-project scenarios), the singleton returns the old root. The new Spec A resolution algorithm makes this worse because CWD-based detection can change within a process. |
| **Resolution** | Implemented singleton invalidation keyed by (`AGENTQMS_PROJECT_ROOT`, CWD) and added explicit `reset_config_loader()` helper for tests/multi-project workflows. |
| **Trigger** | Smoke gate + targeted checks confirm root changes are reflected after env/CWD changes. Keep regression test in place. |

---

## High Risks — Orphaned Features / Broken Imports

### R-04: Broken Import in `mcp_server.py` — `artifact_templates` — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-04 |
| **Category** | Orphaned Features |
| **Impact** | ~~High~~ → Resolved |
| **Affected Files** | `mcp_server.py:187` |
| **Description** | `from AgentQMS.tools.core.artifact_templates import ArtifactTemplates` — module path does not exist. Correct path: `AgentQMS.tools.core.artifacts.artifact_templates`. This means the `list_artifact_templates` and `_get_plugin_artifact_types` MCP tools will throw `ImportError` at runtime. |
| **Resolution** | Fixed to `AgentQMS.tools.core.artifacts.artifact_templates` in all affected `mcp_server.py` call sites. |
| **Trigger** | `rg "AgentQMS.tools.core.artifact_templates" AgentQMS/mcp_server.py` should return 0 matches. |

### R-05: Broken Import in `workflow_detector.py` — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-05 |
| **Category** | Orphaned Features |
| **Impact** | ~~High~~ → Resolved |
| **Affected Files** | `tools/core/plugins/workflow_detector.py` |
| **Description** | Imports `AgentQMS.tools.core.context_bundle` — correct path is `AgentQMS.tools.core.context.context_bundle`. This breaks any workflow detection triggered by the plugin system. |
| **Resolution** | Import corrected to `AgentQMS.tools.core.context.context_bundle`; module import succeeds. |
| **Trigger** | `rg "from AgentQMS.tools.core.context_bundle" AgentQMS/tools/core/plugins/workflow_detector.py` should return 0 matches. |

### R-06: `commands.json` — RESOLVED (moved to _deprecated/)

| Field | Value |
|:---|:---|
| **Risk ID** | R-06 |
| **Category** | Orphaned Features |
| **Impact** | ~~High~~ → Resolved |
| **Affected Files** | `.agentqms/commands.json` → moved to `_deprecated/.agentqms/commands.json` |
| **Description** | Confirmed via `grep` across all `.py`, `.sh`, `.yaml`, `.yml`, `.ts`, `.js`, and `Makefile` files: **zero code references** to `commands.json`. It was a legacy Makefile extraction artifact with 11+ stale paths. Moved to `_deprecated/` on 2026-03-17. The canonical command surface is `cli.py` subcommands. |
| **Resolution** | Moved to `AgentQMS/_deprecated/.agentqms/commands.json`. No code changes required. |

### R-07: `tools/utils/config/` Missing `__init__.py` — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-07 |
| **Category** | Broken Import Chain |
| **Impact** | ~~High~~ → Resolved |
| **Affected Files** | `tools/utils/config/` directory |
| **Description** | Without `__init__.py`, `from AgentQMS.tools.utils.config import load_config` (used by `timestamps.py`, `validate_artifacts.py`) will fail at runtime in strict import scenarios. |
| **Resolution** | Added `tools/utils/config/__init__.py` with canonical exports (`ConfigLoader`, `load_config`, `get_config_loader`) and `YamlCacheLoader` alias. |
| **Trigger** | `python -c "from AgentQMS.tools.utils.config import load_config, ConfigLoader"` should succeed. |

### R-08: 38 Modules Depend on `tools.utils.paths` — Single Point of Failure

| Field | Value |
|:---|:---|
| **Risk ID** | R-08 |
| **Category** | Cascading Regression |
| **Impact** | High |
| **Affected Files** | `tools/utils/paths.py` → `tools/utils/system/paths.py` → `config.py` |
| **Description** | ADT analysis confirms 38 import sites depend on `AgentQMS.tools.utils.paths`. This module is a thin re-export layer over `system/paths.py`, which calls `get_config_loader()`. Any behavioral change to root resolution propagates to every consumer instantly. |
| **Mitigation** | Spec A must include integration tests that exercise all 38 consumer modules with the new resolution logic. At minimum, a smoke-import test for each. |
| **Trigger** | `adt analyze-dependencies AgentQMS/ | grep "AgentQMS.tools.utils.paths"` — count must be verified pre/post refactor. |

---

## Medium Risks — Redundancy / Confusion

### R-09: Duplicate `init_debug_session.py` — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-09 |
| **Category** | Redundancy |
| **Impact** | ~~Medium~~ → Resolved |
| **Affected Files** | `tools/maintenance/init_debug_session.py`, `tools/core/artifacts/init_debug_session.py` |
| **Description** | Two identical files with `create_session()` and `main()`. Neither imports the other. Creates confusion about which is canonical. |
| **Resolution** | Duplicate `tools/maintenance/init_debug_session.py` removed; canonical implementation remains under `tools/core/artifacts/`. |
| **Trigger** | `ls AgentQMS/tools/maintenance/init_debug_session.py` should fail (deleted). |

### R-10: `bin/aqms` vs `cli.py` — Overlapping CLI Surfaces

| Field | Value |
|:---|:---|
| **Risk ID** | R-10 |
| **Category** | Redundancy |
| **Impact** | Medium |
| **Affected Files** | `bin/aqms`, `cli.py` |
| **Description** | Both define an `aqms` CLI with overlapping subcommands (`registry`, `plugin`, `artifact`). `bin/aqms` delegates to subprocess calls; `cli.py` uses direct Python imports. Spec B says "locate existing CLI entry points and decide canonical command name" but doesn't address the fact that two exist with different architectures. |
| **Mitigation** | Spec B1 must explicitly choose one as canonical and deprecate/delete the other. Recommendation: keep `cli.py` (richer subcommands) and make `bin/aqms` a thin wrapper that calls `cli.py::main()`. |
| **Trigger** | Only one of `bin/aqms` or `cli.py` should define argparse parsers post-merge. |

### R-11: Orphaned Modules Never Imported — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-11 |
| **Category** | Orphaned Features |
| **Impact** | ~~Medium~~ → Resolved |
| **Resolved** | `tools/compliance/doc_sync_audit.py`, `tools/maintenance/janitor.py`, `tools/multi_agent/rabbitmq_transport.py` moved to `_deprecated/`; `tools/utils/telemetry.py` deleted. |
| **Resolution** | Completed A0 orphan cleanup. No active call sites for removed telemetry stub. |
| **Trigger** | `rg "tools.utils.telemetry|doc_sync_audit|janitor|rabbitmq_transport" AgentQMS/ --type py` should only match `_deprecated` references (or none). |

### R-12: Legacy Path References in Code

| Field | Value |
|:---|:---|
| **Risk ID** | R-12 |
| **Category** | Orphaned Features |
| **Impact** | Medium |
| **Affected Files** | Multiple modules |
| **Description** | Several modules contain import paths or string references to `tools/utilities/` and `tools/documentation/` — directories that no longer exist. These include `AgentQMS.tools.utilities.versioning`, `AgentQMS.tools.utilities.tracking_integration`, `AgentQMS.tools.utils.git` (should be `system.git`), `AgentQMS.tools.utils.timestamps` (should be `system.timestamps`). |
| **Mitigation** | Run `rg "tools.utilities\|tools.documentation\|tools\.utils\.git\b\|tools\.utils\.timestamps\b" AgentQMS/` and fix all hits before Spec A implementation. |
| **Trigger** | `rg` command above should return 0 matches post-fix. |

### R-13: `generate_mechanized_graph.py` — Extreme Complexity

| Field | Value |
|:---|:---|
| **Risk ID** | R-13 |
| **Category** | Cascading Regression |
| **Impact** | Medium |
| **Affected Files** | `tools/generate_mechanized_graph.py` |
| **Description** | Cyclomatic complexity of 60 (highest in the codebase), 290 LOC, nesting depth 5. This module generates dependency graphs from the registry. If the registry format or paths change during Spec A/B refactoring, this module will silently produce incorrect graphs without failing. |
| **Mitigation** | Add a known-good graph output as a regression test fixture. Verify after each spec completion. |
| **Trigger** | Diff of graph output pre/post refactor should be intentional only. |

---

## Low Risks — Technical Debt

### R-14: `mcp_server.py::call_tool` — Nesting Depth 10

| Field | Value |
|:---|:---|
| **Risk ID** | R-14 |
| **Category** | Technical Debt |
| **Impact** | Low |
| **Affected Files** | `mcp_server.py:456-641` |
| **Description** | CC=21, nesting=10. The `call_tool` function is a monolithic if/elif chain. Adding new tools or modifying root resolution within it is error-prone. |
| **Mitigation** | After Spec A/B, refactor to a tool dispatch registry pattern. Not blocking for initial implementation. |

### R-15: Deprecated Command Aliases Not Guarded — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-15 |
| **Category** | Technical Debt |
| **Impact** | ~~Low~~ → Resolved |
| **Affected Files** | `.agentqms/commands.json` |
| **Description** | `deprecated_commands` section listed 7 deprecated commands with no runtime guard. |
| **Resolution** | `commands.json` moved to `_deprecated/` on 2026-03-17. File had zero code references. The canonical CLI is `cli.py`. |

---

## Pruning Candidates (Pre-Implementation)

Based on AST analysis, the following should be consolidated or removed **before** beginning Spec A implementation:

| Action | Target | Status |
|--------|--------|--------|
| ~~MOVE to _deprecated/~~ | `tools/multi_agent/rabbitmq_transport.py` | **Done** 2026-03-17 |
| ~~MOVE to _deprecated/~~ | `tools/maintenance/janitor.py` | **Done** 2026-03-17 |
| ~~MOVE to _deprecated/~~ | `tools/compliance/doc_sync_audit.py` | **Done** 2026-03-17 |
| ~~MOVE to _deprecated/~~ | `.agentqms/commands.json` | **Done** 2026-03-17 (zero code refs) |
| ~~DELETE~~ | `tools/maintenance/init_debug_session.py` | **Done** (deleted) |
| ~~DELETE~~ | `tools/utils/telemetry.py` | **Done** (deleted) |
| ~~FIX~~ | `tools/core/plugins/workflow_detector.py` | **Done** (import corrected) |
| ~~FIX~~ | `mcp_server.py:187` | **Done** (import corrected in all call sites) |
| ~~CREATE~~ | `tools/utils/config/__init__.py` | **Done** (canonical exports added) |

---

## Review Schedule

- **Spec A0 (pre-flight):** R-04, R-05, R-07 — completed (imports + package init)
- **Spec A (root resolution):** R-01 and R-03 completed; R-02 (partial, naming cleanup pending), R-08 (ongoing cascade risk)
- **After Spec A, before Spec B:** R-09, R-10, R-12 (duplicate files, CLI consolidation, legacy paths)
- **After Spec B:** R-13, R-14 (complexity and debt)
- **Resolved:** R-01, R-03, R-04, R-05, R-06, R-07, R-09, R-11, R-15
