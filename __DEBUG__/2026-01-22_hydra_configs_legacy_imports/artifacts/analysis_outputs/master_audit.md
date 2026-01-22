# Systemic Dependency & Configuration Audit Plan

## Goal
Eliminate the cycle of "fix one, break one" by performing a comprehensive, one-shot audit of the entire codebase for broken imports and invalid Hydra targets.

## Strategy
1.  **Develop `scripts/audit/master_audit.py`**:
    *   **Python Import Scanner**: Uses `ast` to parse every [.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/run_ui.py) file in the workspace. Resolves imports (absolute and relative) and verifies that the target module/object exists.
    *   **Hydra Target Scanner**: Parses every [.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AGENTS.yaml) file, extracts [_target_](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/audit/check_targets.py#7-18) and `_partial_` keys, and verifies that the referenced Python path is importable.
    *   **Path Checker**: Checks for hardcoded paths (like the LMDB error) defined in configs.
2.  **Run & Report**:
    *   Generate a structured logical report (JSON/Console) grouping errors by type:
        *   `BrokenImport`: Python file trying to import non-existent module.
        *   `BrokenTarget`: YAML file pointing to non-existent class/function.
        *   `MissingPath`: Configuration pointing to non-existent file/directory.
3.  **Batch Execution**:
    *   Apply fixes to *all* identified issues in a single pass.
4.  **Verification**:
    *   Re-run the audit to confirm 0 issues.
    *   Run smoke tests for both domains.

## User Review Required
- None (Internal debugging tool).

## Proposed Audit Script Structure
- **Input**: Workspace root.
- **Process**:
    - Walk [ocr/](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr) and `drivers/` (or `runners/`).
    - Parse AST for imports.
    - Walk [configs/](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#205-257).
    - Parse YAML for [_target_](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/audit/check_targets.py#7-18).
    - Importlib/Pkgutil for verification.
- **Output**: List of anomalies.
