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

| Severity | Open | Resolved/Mitigated | Category |
|----------|------|--------------------|----------|
| Critical | 0    | 3                  | Cascading Regression (R-01, R-02, R-03 closed) |
| High     | 0    | 5                  | Orphaned Features / Broken Imports (R-04, R-05, R-07, R-08 closed) |
| Medium   | 0    | 5                  | Redundancy / Confusion (R-10, R-12, R-13 closed) |
| Low      | 0    | 3                  | Technical Debt (R-14, R-15 resolved) |

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

### R-02: Two Incompatible `ConfigLoader` Classes — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-02 |
| **Category** | Cascading Regression |
| **Impact** | ~~Critical~~ → Resolved |
| **Affected Files** | `tools/utils/config/config.py`, `tools/utils/config/loader.py` |
| **Description** | Utility loader has been renamed to `YamlCacheLoader`; canonical root resolver remains `config.py::ConfigLoader`. |
| **Evidence** | `tools/utils/config/loader.py` now defines `YamlCacheLoader` with compatibility alias `ConfigLoader = YamlCacheLoader`; imports in `cli.py`, `mcp_server.py`, `workflow_detector.py`, `context_bundle.py`, and `artifact_templates.py` were migrated to explicit `from AgentQMS.tools.utils.config import YamlCacheLoader`. |
| **Resolution** | Class rename and call-site migration completed. Package exports stay explicit: canonical `ConfigLoader` (root-aware) + `YamlCacheLoader` (YAML/cache utility). |
| **Trigger** | `rg "from AgentQMS.tools.utils.config.loader import ConfigLoader" AgentQMS/ --type py` should return 0 matches. |

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

### R-08: `tools.utils.paths` Consumers — VERIFIED/RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-08 |
| **Category** | Cascading Regression |
| **Impact** | ~~High~~ → Verified/Resolved |
| **Affected Files** | `tools/utils/paths.py` → `tools/utils/system/paths.py` → `config.py` |
| **Description** | Consumer import stability and root/path behavior have been smoke-verified against env-override and marker-traversal scenarios. |
| **Evidence** | Added `scripts/mcp/smoke_paths_consumers.py`; run output: `Env Override PASS`, `Marker Traversal PASS`, `modules checked=29`, `import failures=0`, `root mismatches=0`. Existing root gate (`scripts/mcp/smoke_project_root_resolution.py`) also passes all scenarios after changes. |
| **Resolution** | Dependency fan-out remains, but regression risk is now covered by an executable consumer smoke integration gate and root behavior evidence. |
| **Trigger** | `python scripts/mcp/smoke_paths_consumers.py` and `python scripts/mcp/smoke_project_root_resolution.py` must both pass. |

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

### R-10: `bin/aqms` vs `cli.py` — RESOLVED (Canonical CLI Consolidated)

| Field | Value |
|:---|:---|
| **Risk ID** | R-10 |
| **Category** | Redundancy |
| **Impact** | ~~Medium~~ → Resolved |
| **Affected Files** | `bin/aqms`, `cli.py` |
| **Description** | Canonical CLI is `python -m AgentQMS.cli`; wrapper surface is reduced to pure deprecation + forwarder behavior. |
| **Evidence** | `AgentQMS/cli.py` now owns plugin compatibility routing (`plugin list|validate|show`), while `AgentQMS/bin/aqms` emits deprecation guidance and delegates all commands to canonical CLI. Active docs and specs command examples were migrated to `python -m AgentQMS.cli ...` (`AGENTS.md`, `AgentQMS/AGENTS.yaml`, tier specs, copilot instructions). |
| **Resolution** | Wrapper compatibility branch retired (including plugin-specific handler). `bin/aqms` remains only as a temporary deprecation shim to avoid abrupt breaking change. |
| **Trigger** | Keep canonical-doc checks (`rg "python -m AgentQMS\\.cli" AGENTS.md AgentQMS/AGENTS.yaml AgentQMS/specs/ .github/copilot-instructions.md`) plus wrapper parser-free check (`rg "argparse.ArgumentParser" AgentQMS/bin/aqms`). |

### R-11: Orphaned Modules Never Imported — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-11 |
| **Category** | Orphaned Features |
| **Impact** | ~~Medium~~ → Resolved |
| **Resolved** | `tools/compliance/doc_sync_audit.py`, `tools/maintenance/janitor.py`, `tools/multi_agent/rabbitmq_transport.py` moved to `_deprecated/`; `tools/utils/telemetry.py` deleted. |
| **Resolution** | Completed A0 orphan cleanup. No active call sites for removed telemetry stub. |
| **Trigger** | `rg "tools.utils.telemetry|doc_sync_audit|janitor|rabbitmq_transport" AgentQMS/ --type py` should only match `_deprecated` references (or none). |

### R-12: Legacy Path References in Code — RESOLVED

| Field | Value |
|:---|:---|
| **Risk ID** | R-12 |
| **Category** | Orphaned Features |
| **Impact** | ~~Medium~~ → Resolved |
| **Affected Files** | Multiple modules |
| **Description** | Removed obsolete module references and moved call sites to current import surfaces. |
| **Evidence** | `workflow.py` now imports tracking boundary from `tools/core/artifacts/tracking_integration.py`; `status.py` imports `tools/core/artifacts/versioning.py`; `artifact_templates.py` now imports `tools.utils.system.git` and `tools.utils.system.timestamps`; legacy references removed from utility docs/comments. |
| **Resolution** | `rg "tools.utilities\|tools.documentation\|tools\.utils\.git\b\|tools\.utils\.timestamps\b" AgentQMS/ --type py` now returns 0 matches. |
| **Trigger** | Keep zero-match `rg` check in pre-merge validation. |

### R-13: `generate_mechanized_graph.py` — RESOLVED (Refactor + Deterministic Smoke Gate)

| Field | Value |
|:---|:---|
| **Risk ID** | R-13 |
| **Category** | Cascading Regression |
| **Impact** | ~~Medium~~ → Resolved |
| **Affected Files** | `tools/generate_mechanized_graph.py` |
| **Description** | Graph generation path had high complexity and weak regression detection. |
| **Evidence** | `tools/generate_mechanized_graph.py` is now decomposed into focused helpers for tier rendering, edge rendering, legend rendering, parser/output orchestration, and edge counting. Added deterministic smoke gate: `scripts/mcp/smoke_mechanized_graph.py` asserting same-input/same-output and required governance/dependency/critical-path edges. |
| **Resolution** | Complexity surface split into smaller pure functions while preserving output behavior and CLI contract (`--no-legend`, `--no-domains`, `--dry-run`, `--output`). |
| **Trigger** | `python scripts/mcp/smoke_mechanized_graph.py` must pass and output diffs should be intentional only. |

---

## Low Risks — Technical Debt

### R-14: `mcp_server.py::call_tool` — RESOLVED (Handlers Extracted)

| Field | Value |
|:---|:---|
| **Risk ID** | R-14 |
| **Category** | Technical Debt |
| **Impact** | ~~Low~~ → Resolved |
| **Affected Files** | `mcp_server.py:456-641` |
| **Description** | `mcp_server.py` previously contained both wiring and concrete tool handlers, limiting testability and raising change-coupling risk. |
| **Evidence** | Extracted handlers to `AgentQMS/tools/core/mcp/handlers.py` with `TOOL_HANDLERS` registry + `HandlerContext`; `mcp_server.py` now remains a thin server wiring + dispatch layer. Added smoke gate `scripts/mcp/smoke_mcp_dispatch.py` verifying handler registry keys, known-tool dispatch shape, and unknown-tool error behavior. |
| **Resolution** | Slice 2 completed: concrete handlers isolated from transport/wiring without changing MCP tool contracts. |
| **Trigger** | `python scripts/mcp/smoke_mcp_dispatch.py` must pass and unknown-tool path must return JSON payload with `error`. |

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
- **Spec A (root resolution):** R-01, R-02, R-03 completed; R-08 verification completed with consumer smoke gate
- **After Spec A, before Spec B:** R-09 done, R-10 and R-12 closed
- **After Spec B:** R-13 refactor + deterministic smoke closure, R-14 handler extraction closure
- **Resolved/Mitigated:** R-01, R-02, R-03, R-04, R-05, R-06, R-07, R-08, R-09, R-10, R-11, R-12, R-13, R-14, R-15
