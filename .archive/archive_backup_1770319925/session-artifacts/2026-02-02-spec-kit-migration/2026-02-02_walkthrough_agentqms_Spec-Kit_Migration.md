# Walkthrough - Spec-Kit Migration

**Date**: 2026-02-02
**Task**: Migrate AgentQMS from legacy [standards/](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/utils/bundle_standards.py#5-26) to atomic [specs/](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/standards/tier1-sst/specs).

## Changes

### 1. New Directory Structure
The `AgentQMS/standards/` directory has been **purged**. The new source of truth is `AgentQMS/specs/`:

```bash
AgentQMS/specs/
├── tier1-contracts/
│   ├── architecture.spec.md
│   ├── compliance.spec.md
│   └── validation.spec.md
├── tier2-framework/
│   ├── configuration.spec.md
│   ├── core_infra.spec.md
│   ├── ocr_engine.spec.md
│   └── ...
├── tier3-agents/
│   └── agent_identities.spec.md
├── tier4-workflows/
│   ├── validation.spec.md
│   └── workflows.spec.md
└── schemas/
    └── ads_schemas.spec.md
```

### 2. Validation Logic Update
`validate_artifacts.py` was refactored to parse Markdown tables instead of loading YAML files.
*   **Source**: `tier1-contracts/compliance.spec.md` (Table 2).
*   **Parser**: `AgentQMS/tools/compliance/spec_parser.py`.

### 3. Tool Repair (Phase 6)
*   Updated `AgentQMS/middleware/policies.py` to monitor `specs/`.
*   Updated `AgentQMS/tools/compliance/generate_report.py` to scan `specs/`.
*   Updated `scripts/audit_config_compliance.py` to use `configuration.spec.md`.
*   Updated `AgentQMS/tools/core/context/suggest_context.py` path.

### 4. Verification
Ran `tests/compliance/test_validate_dual_version.py`:
```
Test Results:
- test_legacy_artifact: OK (V1 Fallback working)
- test_modern_artifact: OK (V2 Spec Parsing working)
- test_modern_artifact_missing_field: OK (Validation Logic active)
----------------------------------------------------------------------
Ran 3 tests in 0.539s
OK
```

## Review & Next Steps
-   The system is now safe for mixed-mode operations.
-   Legacy artifacts can stay as-is until they are ready to be upgraded.
-   New artifacts should use `ads_version: '2.0'`.

## Proof of Work
### Legacy vs New
*   **Before**: > 60 fragmented YAML files.
*   **After**: ~15 Atomic Markdown Specs (< 600 tokens each).

## Next Steps
*   Update `Makefile` to ensure `make qms-validate` points to `validate_artifacts.py` (It does).
*   Agents can now use **Project Context Bundler** to retrieve strictly relevant specs.
