# Backlog — Run 2026-03-17_0000

Status values: `todo` | `in_progress` | `blocked` | `done`

## Spec A0 — Pre-flight cleanup (prerequisite for Spec A)
- [done] A0.1: Move orphaned files to `AgentQMS/_deprecated/` (rabbitmq_transport.py, janitor.py, doc_sync_audit.py, commands.json).
- [done] A0.2: Fix broken import in `mcp_server.py:187` (`AgentQMS.tools.core.artifact_templates` → `AgentQMS.tools.core.artifacts.artifact_templates`).
- [done] A0.3: Fix broken import in `tools/core/plugins/workflow_detector.py` (`AgentQMS.tools.core.context_bundle` → `AgentQMS.tools.core.context.context_bundle`).
- [done] A0.4: Create `tools/utils/config/__init__.py` re-exporting canonical `ConfigLoader` and `load_config`.
- [done] A0.5: Delete duplicate `tools/maintenance/init_debug_session.py` (exact copy of `tools/core/artifacts/init_debug_session.py`).
- [done] A0.6: Delete empty stub `tools/utils/telemetry.py`.
- [done] A0.7: Establish broken-import baseline: `adt analyze-imports AgentQMS/ > _baseline_imports.json`.

## Spec A — Dynamic project resolution
- [done] A1: Identify all project_root detection implementations (ConfigLoader, mcp_server.py, cli.py, bin/aqms) and their call sites.
- [done] A2: Implement env override `AGENTQMS_PROJECT_ROOT` in canonical `ConfigLoader._detect_project_root`.
- [done] A3: Implement CWD upward traversal for markers (`.agentqms/`, `AGENTS.yaml`).
- [done] A4: Implement fallback to CWD for empty dirs (supports `init`).
- [done] A5: Update `paths.py` to remove framework-relative assumptions.
- [done] A6: Replace `mcp_server.py::find_project_root()` with call to canonical resolver.
- [done] A7: Replace `cli.py` fallback path hack (lines 42–46) with canonical resolver.
- [done] A8: Replace `bin/aqms` hardcoded `parents[2]` (line 38) with canonical resolver.
- [done] A9: Verify `scripts/mcp/unified_server.py` resolves correct root end-to-end (it imports `mcp_server.py` dynamically).
- [done] A10: Add singleton reset/invalidation to `get_config_loader()` for env-var changes.
- [done] A11: Add verification commands/tests for env override, marker traversal, fallback, and entry-point convergence.
- [done] A12: Run `adt analyze-imports` regression check against baseline.

## Spec B — CLI entry point
- [done] B1: Consolidate `bin/aqms` and `cli.py` — choose `cli.py` as canonical, make `bin/aqms` a thin wrapper.
- [done] B2: Register entry point in `pyproject.toml` (`aqms = "AgentQMS.cli:main"`).
- [done] B3: Ensure CLI commands route through Spec A root resolution.
- [done] B4: Add verification commands for "run anywhere".

## Spec C — init scaffolding
- [done] C1: Implement `init` command to create `.agentqms/` + minimal templates in resolved root.
- [done] C2: Enforce idempotency/no overwrite without explicit flag.
- [done] C3: Add verification commands for empty-dir init + idempotency.
