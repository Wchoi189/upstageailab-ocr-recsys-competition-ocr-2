# Backlog — Run 2026-03-17_0000

Status values: `todo` | `in_progress` | `blocked` | `done`

## Spec A — Dynamic project resolution
- [todo] A1: Identify current `project_root` detection implementation and call sites (ConfigLoader + server startup).
- [todo] A2: Implement env override `AGENTQMS_PROJECT_ROOT` (resolve absolute; validate exists?).
- [todo] A3: Implement CWD upward traversal for markers (`.agentqms/`, `AGENTS.yaml`).
- [todo] A4: Implement fallback to CWD for empty dirs (supports `init`).
- [todo] A5: Update `paths.py` to remove framework-relative assumptions.
- [todo] A6: Ensure `unified_server.py` uses the resolved project root consistently.
- [todo] A7: Add verification commands/tests for env override, marker traversal, fallback.

## Spec B — CLI entry point
- [todo] B1: Locate existing CLI entry points and decide canonical command name (`aqms` alignment).
- [todo] B2: Register entry point in `pyproject.toml`.
- [todo] B3: Ensure CLI commands route through Spec A root resolution.
- [todo] B4: Add verification commands for “run anywhere”.

## Spec C — init scaffolding
- [todo] C1: Implement `init` command to create `.agentqms/` + minimal templates in resolved root.
- [todo] C2: Enforce idempotency/no overwrite without explicit flag.
- [todo] C3: Add verification commands for empty-dir init + idempotency.

