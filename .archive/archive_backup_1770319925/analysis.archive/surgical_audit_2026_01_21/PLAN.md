# Surgical Audit Plan: ocr/core Autopsy

**Objective**: Perform a line-by-line, file-by-file forensic analysis of `ocr/core` to identify, root out, and destroy:
1.  **Domain Leakage**: Detection/Recognition specific logic hiding in Core.
2.  **Dual/Shadow Architectures**: Code that duplicates domain functionality.
3.  **Bloat**: Dead code, legacy shims, and over-engineered abstractions.
4.  **"Junk"**: Unnecessary files, temp scripts, or non-standard artifacts.

## Strategy: "Trust No File"

We will stop "passive" audits (checking file names/existence) and move to "active" audits (AST parsing and content analysis).

### Phase 1: The Autopsy Tool (`scripts/audit/core_autopsy.py`)
We will write a specialized script to scan `ocr/core` with the following heuristics:

*   **Import Analysis**:
    *   **Inward Violation**: Does a core file import *from* `ocr.domains`? (Strictly Forbidden, indicates Core depends on Domain).
    *   **Sibling Violation**: Does a core file import *from* `ocr.features`? (Legacy ghost artifact).
*   **Keyword/Symbol Analysis**:
    *   Scan for hardcoded domain terms: `DBNet`, `CRAFT`, `PARSeq`, `Polygon`, `BBox` (context dependent), `accuracy` (context dependent).
    *   Flag files with high concentrations of these terms.
*   **Complexity/Bloat Metrics**:
    *   Lines of Code (LOC).
    *   Cyclomatic Complexity.
    *   Number of Classes/Functions.
*   **Legacy Artifact Detection**:
    *   "Compat", "Shim", "Deprecated", "Legacy" markers.

### Phase 2: The "Kill List" Generation
Based on the Autopsy Report, we will categorize every file in `ocr/core` into:
1.  **KEEP**: True Shared Infrastructure (Logger, Config parsing).
2.  **MOVE**: Domain logic masquerading as core (Move to `ocr/domains/...`).
3.  **DELETE**: Dead code, shims, or junk.
4.  **REFACTOR**: "God Objects" that need splitting (e.g., `ocr/core/validation.py`).

### Phase 3: Execution (Surgical Removal)
*   **Move**: Physically relocate files using `git mv` (or equivalent) to preserve history if possible, or create/delete.
*   **Refactor**: Split files like `validation.py` into `ocr/domains/detection/schemas.py` etc.
*   **Purge**: Delete the junk.

## Implementation Steps

1.  Create `analysis/surgical_audit_2026_01_21/AUTOPSY_REPORT.md` to track findings.
2.  Develop `scripts/audit/core_autopsy.py`.
3.  Run Autopsy.
4.  Present "Kill List" to User for approval.
5.  Execute.
