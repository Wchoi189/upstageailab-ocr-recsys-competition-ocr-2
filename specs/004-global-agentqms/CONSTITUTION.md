# Constitution — Global Stateless AgentQMS

## Problem Statement & Mission

`AgentQMS` is the quality-management framework that governs artifact creation, compliance validation, context bundling, and standards resolution for AI coding agents. Today it is **fundamentally broken as a portable tool** because it cannot answer the question "which project am I managing?" without first locating its own source code.

**Why this matters:**
1. **Tight coupling kills reuse.** The current `_detect_project_root()` walks up from `__file__` to find the `AgentQMS/` directory, then looks at its parent. This means the framework *is* the project — you cannot install AgentQMS globally and point it at an arbitrary workspace the way you can with `git`, `npm`, or `uv`.
2. **Four competing root-detection mechanisms exist (plus a fifth transitive one).** `config.py`, `mcp_server.py`, `cli.py`, and `bin/aqms` each hardcode their own heuristic for finding the project root. The production server `scripts/mcp/unified_server.py` correctly delegates to `get_project_root()` but dynamically imports `AgentQMS.mcp_server` — inheriting its stale root logic. Any refactor that touches only one will leave the others diverged, causing silent correctness bugs.
3. **Downstream tooling is fragile.** 38 internal modules import `AgentQMS.tools.utils.paths`, which delegates to the singleton `ConfigLoader`. A wrong root poisons every artifact path, validation scope, and compliance check in the system.
4. **Integration debt is compounding.** Orphaned scripts, broken imports referencing deleted `tools/utilities/` and `tools/documentation/` paths, duplicate `ConfigLoader` classes, and stale `commands.json` entries mean the codebase already has latent failures that any refactor will surface.

**The goal of this spec is to decouple the Framework Scope (where the code lives) from the Project Scope (where the work happens)** so that AgentQMS becomes a stateless, globally installable tool that can manage any project from any location — while simultaneously cleaning up the integration debt that makes the current implementation unreliable.

## Non-negotiables
- All planning + session artifacts MUST live under `specs/004-global-agentqms/` (no workspace clutter).
- Session handovers use **Option A**: multiple files with timestamped filenames.
- **Single source of truth for root resolution:** After this spec, exactly ONE function determines `project_root`. All entry points (`cli.py`, `mcp_server.py`, `bin/aqms`) delegate to it.

## Invariants (must always hold)
### Project root resolution
Order of precedence for `project_root` detection:
1. `AGENTQMS_PROJECT_ROOT` environment variable (absolute or relative, resolved to absolute).
2. Upward traversal from current working directory (CWD) to find a marker directory/file:
   - `.agentqms/` OR `AGENTS.yaml` (primary markers)
   - optionally `.git/` or `pyproject.toml` (secondary markers) if the implementation chooses to include them
3. Fallback: CWD (supports running in empty directories for `init`).

### Statelessness boundary
- The framework install location (where `AgentQMS` code lives) MUST NOT influence `project_root`.
- All project-scoped state MUST be confined to the resolved `project_root` (e.g., `.agentqms/` contents).

### Isolation & safety
- No cross-project registry contamination: resolved `project_root` defines the only writable scope.
- Global invocation MUST be deterministic: identical `AGENTQMS_PROJECT_ROOT` + CWD markers ⇒ identical root resolution.

### Consolidated entry points
- `bin/aqms`, `cli.py`, `mcp_server.py`, and `scripts/mcp/unified_server.py` MUST NOT contain independent root-detection logic. They MUST import and use the canonical resolver from `tools/utils/config/config.py`.
- `tools/utils/config/loader.py` (the YAML/Redis ConfigLoader) MUST NOT be confused with `tools/utils/config/config.py` (the framework ConfigLoader). The two serve different purposes and must not share a class name without an explicit adapter.

## Evidence-first workflow
- Any spec completion must include at least one reproducible verification command + expected observable outcome.
- Before marking a spec task as `done`, broken-import and orphaned-script checks (via `adt analyze-imports` and `adt analyze-dependencies`) must show zero regressions.

