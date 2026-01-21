# Session Handover: OCR Domain Refactor Planning Complete

## 1. Executive Summary
The Planning Phase for the **OCR Domain Separation Refactor** is complete.
We have designed a **"Domains First" Architecture** to eliminate domain leakage (e.g., Detection logic in `ocr/core`).

## 2. Key Artifacts
- **Roadmap**: `project_compass/roadmap/ocr-domain-refactor.yml` (Project Root)
- **Detailed Tree**: `docs/artifacts/implementation_plans/...proposed-directory-tree.md` (The "Bible" for the move)
- **Classification Report**: `docs/reports/vlm_reports/...classification_report.md` (Audit findings)

## 3. Architecture Overview
- **Core**: Strictly generic utilities. No `polygon`, `box`, or `token` logic.
- **Domains**: `detection`, `recognition`, `kie`, `layout`.
- **Pipelines**: New orchestration layer to bridge domains.
- **Interfaces**: New `ocr/core/interfaces/schemas.py` for data contracts.

## 4. Next Steps (Execution Phase)
The next session should immediately begin execution:
1.  **Initialize Directories**: Create `ocr/domains/`, `ocr/pipelines/`, `ocr/core/interfaces/`.
2.  **The Mover**: Execute bulk file moves based on the **Detailed Tree** artifact.
3.  **The Splitter**: Refactor `wandb_utils.py` and `ocr_pl.py`.

## 5. Tooling
- Use **Cliff (CLI with venv)** for `adt` commands if MCP fails.
- `sg_search` is operational via MCP.
