# Spec A — Dynamic Project Resolution

## Goal
Decouple framework scope from project scope by resolving `project_root` dynamically at runtime.

## ConfigLoader Internal Refactoring (mandatory)

### Why this subsection exists
Spec A can fail passively if execution focuses only on `_detect_project_root()` and does not redirect downstream config IO from framework scope to project scope. This subsection is normative and must be completed in full.

### ConfigLoader duality boundary (must not be confused)
- `AgentQMS/tools/utils/config/config.py::ConfigLoader`
  - Role: canonical root resolver + layered framework/project/environment merge.
  - Consumed by `paths.py` / `system/paths.py` and therefore all path-dependent runtime behavior.
- `AgentQMS/tools/utils/config/loader.py::ConfigLoader`
  - Role: generic YAML/Redis cache loader + virtual config generation helpers.
  - Consumed by MCP/CLI utilities requiring ad-hoc YAML loads or virtual snapshots.

### Namespace strategy (required during refactor)
- All touched modules must use explicit import aliases when both concepts are in scope:
  - `from AgentQMS.tools.utils.config.config import ConfigLoader as RootConfigLoader`
  - `from AgentQMS.tools.utils.config.loader import ConfigLoader as YamlCacheLoader`
- If only one class is in scope, import from the fully-qualified module path (`...config.config` or `...config.loader`) and avoid bare re-exports during migration.
- Future cleanup item (follow-up spec): rename `loader.py::ConfigLoader` to `YamlCacheLoader` with a temporary compatibility alias to eliminate long-term ambiguity.

### Surgical logic redirection in `config.py` (required)

#### Method: `load()`
- **Before (incorrect post-decoupling):**
  - `settings_path = self.framework_root / ".agentqms" / "settings.yaml"`
- **After (required):**
  - `settings_path = self.project_root / ".agentqms" / "settings.yaml"`
- **Intent:** settings file is project-owned state, never framework-install state.

#### Method: `_load_project_overrides()`
- **Before (incorrect post-decoupling):**
  - `framework_config_dir = self.framework_root / ".agentqms" / "project_config"`
- **After (required):**
  - `project_config_dir = self.project_root / ".agentqms" / "project_config"`
- **Intent:** project override layer must be loaded from target workspace.

#### Method: `_write_runtime_snapshot()`
- **Before (incorrect post-decoupling):**
  - `runtime_dir = self.framework_root / ".agentqms"`
- **After (required):**
  - `runtime_dir = self.project_root / ".agentqms"`
- **Intent:** generated `effective.yaml` must be emitted to project-local state.

### Mandatory execution checklist (blocking)
- [ ] `load()` reads `.agentqms/settings.yaml` from `project_root`, not `framework_root`.
- [ ] `_load_project_overrides()` loads `project_config` from `project_root`, not `framework_root`.
- [ ] `_write_runtime_snapshot()` writes `.agentqms/effective.yaml` under `project_root`, not `framework_root`.
- [ ] Runtime `layers` metadata remains coherent after path redirection (no stale framework-local project paths).
- [ ] `framework_root` remains exclusively for framework assets (e.g., `config_defaults`) and is not used for project-owned mutable state.
- [ ] No module touched in Spec A imports ambiguous `ConfigLoader` symbols without explicit module path or alias.

## In scope (code touchpoints)

### Canonical resolver (primary target)
- `AgentQMS/tools/utils/config/config.py` — `ConfigLoader._detect_project_root` must implement the new CWD/env-var algorithm.

### Entry points that MUST delegate to the canonical resolver
- `AgentQMS/mcp_server.py` — contains its own `find_project_root()` (lines 37–50). Replace with call to canonical resolver.
- `AgentQMS/cli.py` — fallback `Path(__file__).parent.parent` (lines 42–46). Replace with canonical resolver.
- `AgentQMS/bin/aqms` — hardcoded `Path(__file__).resolve().parents[2]` (line 38). Replace with canonical resolver.

### Downstream path layer
- `AgentQMS/tools/utils/paths.py` — thin re-export from `system/paths.py`. Ensure no `__file__`-relative assumptions remain.
- `AgentQMS/tools/utils/system/paths.py` — calls `get_config_loader()`. No changes expected if canonical resolver is correct.

### Production server (verification target)
- `scripts/mcp/unified_server.py` — the production MCP server. Already uses `get_project_root()` from `AgentQMS.tools.utils.system.paths`, but also dynamically imports `AgentQMS.mcp_server` (which has its own root logic). Must be verified end-to-end after all sub-entry points are unified.

### Singleton invalidation
- `config.py::get_config_loader()` — module-level singleton caches root at first instantiation. Must support reset/re-detection when `AGENTQMS_PROJECT_ROOT` changes (for tests and multi-project scenarios).

## Resolution algorithm (must implement)
1. If `AGENTQMS_PROJECT_ROOT` is set, use it (resolved absolute).
2. Else traverse upward from CWD to find marker:
   - `.agentqms/` OR `AGENTS.yaml` (primary markers)
   - Optional: `.git/` or `pyproject.toml` (secondary markers) if already used elsewhere
3. Else fallback to CWD (supports `init` in empty directories).

## Acceptance criteria
- Env override:
  - Given `AGENTQMS_PROJECT_ROOT=/tmp/projB`, root resolves to `/tmp/projB` regardless of CWD.
- Marker traversal:
  - From a nested directory inside a project containing `.agentqms/` or `AGENTS.yaml`, root resolves to the project root.
- Fallback:
  - In an empty directory, root resolves to CWD (enables `init`).
- Entry point convergence:
  - `mcp_server.py`, `cli.py`, and `bin/aqms` all produce the same resolved root as `config.py` for identical inputs.
- Server respect:
  - `scripts/mcp/unified_server.py` starts and operates using the resolved root, not framework-relative paths.
- Singleton invalidation:
  - Changing `AGENTQMS_PROJECT_ROOT` after initial load produces the updated root.

## Verification (evidence required)
- Provide at least:
  - one command verifying env override
  - one command verifying marker traversal
  - one command verifying fallback behavior
  - one command showing all entry points resolve the same root
  - `adt analyze-imports AgentQMS/` diff against baseline shows zero new broken imports

## Paths layer blast-radius map (38 Python modules)
All modules below consume `AgentQMS.tools.utils.paths` and/or `AgentQMS.tools.utils.system.paths`; they are verification targets because they indirectly depend on `config.py::ConfigLoader.project_root`.

- `scripts/utils/fix_hydra_overrides.py`
- `scripts/audit_config_compliance.py`
- `AgentQMS/middleware/policies.py`
- `AgentQMS/tools/compliance/validators/directory.py`
- `AgentQMS/tools/core/context/context_bundle.py`
- `AgentQMS/tools/core/plugins/workflow_detector.py`
- `scripts/utils/show_config.py`
- `scripts/mcp/unified_server.py`
- `scripts/mcp/verify_feedback.py`
- `scripts/mcp/analyze_telemetry.py`
- `scripts/hooks/verify_python_structure.py`
- `dev_tools/agent_debug_toolkit/agent_debug_toolkit/mcp_server.py`
- `AgentQMS/tools/utils/generate_ide_configs.py`
- `AgentQMS/tools/utils/system/timestamps.py`
- `AgentQMS/tools/utils/system/git.py`
- `AgentQMS/tools/maintenance/grok.py`
- `AgentQMS/tools/core/plugins/loader.py`
- `AgentQMS/tools/middleware/dashboard.py`
- `AgentQMS/tools/compliance/validate_artifacts.py`
- `AgentQMS/tools/compliance/generate_report.py`
- `AgentQMS/tools/compliance/validators/bundles.py`
- `AgentQMS/tools/compliance/validate_boundaries.py`
- `AgentQMS/tools/compliance/monitor_artifacts.py`
- `AgentQMS/tools/core/context/context_inspector.py`
- `AgentQMS/tools/core/context/suggest_context.py`
- `AgentQMS/tools/core/context/get_context.py`
- `AgentQMS/tools/core/context/context_control.py`
- `AgentQMS/tools/core/plugins/discovery.py`
- `AgentQMS/tools/core/artifacts/smart_populate.py`
- `AgentQMS/tools/core/artifacts/status.py`
- `AgentQMS/tools/core/artifacts/reindex_artifacts.py`
- `AgentQMS/tools/core/artifacts/workflow.py`
- `AgentQMS/tools/core/artifacts/autofix_artifacts.py`
- `AgentQMS/cli.py`
- `AgentQMS/middleware/health.py`
- `AgentQMS/middleware/logging_config.py`
- `AgentQMS/middleware/telemetry.py`
- `AgentQMS/tools/core/plugins/cli.py`

## Session 2 handover (M2M output)

### Surgery plan: exact `config.py` line targets
- `AgentQMS/tools/utils/config/config.py`
  - `load()` settings path assignment (current line 41)
  - `_load_project_overrides()` project config directory assignment (current line 111)
  - `_write_runtime_snapshot()` runtime output directory assignment (current line 164)
- Scope guard:
  - Keep `framework_root` usage intact for `_load_framework_defaults()` (current line 92) and framework-only assets.
  - Do not alter default-layer semantics except for project-state IO redirection.

### Smoke test blueprint (next session requirement)
Implement a minimal script (e.g., `scripts/mcp/smoke_project_root_resolution.py`) that:
- Instantiates canonical resolver from `config.py` and prints `framework_root`, `project_root`.
- Calls `get_project_root()` through both:
  - `AgentQMS.tools.utils.paths`
  - `AgentQMS.tools.utils.system.paths`
- Invokes entry-point root paths without divergence:
  - `AgentQMS/cli.py` root resolution path
  - `AgentQMS/mcp_server.py` root resolution path
  - `AgentQMS/bin/aqms` bootstrap path
- Executes three scenarios:
  1. `AGENTQMS_PROJECT_ROOT` override set
  2. nested CWD marker traversal (`.agentqms` or `AGENTS.yaml`)
  3. empty directory fallback to CWD
- Asserts all entry points return identical `project_root` per scenario and exits non-zero on mismatch.

## Out of scope
- CLI entrypoint registration (Spec B)
- `init` scaffolding templates (Spec C)
