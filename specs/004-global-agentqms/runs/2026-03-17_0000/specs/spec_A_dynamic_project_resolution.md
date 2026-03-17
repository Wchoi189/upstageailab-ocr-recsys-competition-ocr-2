# Spec A — Dynamic Project Resolution

## Goal
Decouple framework scope from project scope by resolving `project_root` dynamically at runtime.

## In scope (code touchpoints)
- `AgentQMS/tools/utils/config/config.py` (`ConfigLoader._detect_project_root` or equivalent)
- `AgentQMS/tools/utils/paths.py` (ensure paths don’t assume relative to `__file__`)
- `unified_server.py` (or server entry) must respect dynamically resolved root

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
- Server respect:
  - `unified_mcp` server starts and operates using the resolved root, not framework-relative paths.

## Verification (evidence required)
- Provide at least:
  - one command verifying env override
  - one command verifying marker traversal
  - one command verifying fallback behavior

## Out of scope
- CLI entrypoint registration (Spec B)
- `init` scaffolding templates (Spec C)

