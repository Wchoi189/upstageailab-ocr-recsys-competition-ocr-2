---
category: planning
status: active
type: implementation_plan
date: 2025-11-29 18:00 (KST)
---

# Cleanup & Consolidation Implementation Plan

**Goal**: Organize `ocr/` into a clean, strictly hierarchical structure by removing duplicates and consolidating shared resources.

## User Review Required

> [!WARNING]
> This plan involves DELETING several directories (`ocr/lightning_modules`, `ocr/datasets`, `ocr/metrics`, `ocr/models`, `ocr/utils`).
> It also moves major features into a new `features/` container.
> Backup is implicit via Git.

## Proposed Changes

### Phase 1: Remove Exact Duplicates (Cleanup)

#### [DELETE] `ocr/lightning_modules`
*   **Reason**: Exact duplicate of `ocr/core/lightning`.
*   **Action**: Delete directory.
*   **Update**: Find/replace all imports of `ocr.lightning_modules` to `ocr.core.lightning`.

#### [DELETE] `ocr/metrics`
*   **Reason**: Duplicate of `ocr/core/metrics`.
*   **Action**: Delete directory.
*   **Update**: Imports to `ocr.core.metrics`.

#### [DELETE] `ocr/datasets`
*   **Reason**: Superseded by `ocr/data/datasets` (which has larger/newer `base.py`).
*   **Action**: Delete directory.
*   **Update**: Imports `ocr.datasets` -> `ocr.data.datasets`.

### Phase 2: Feature Containerization

#### [NEW] `ocr/features`
*   Create directory to house domain-specific packages.

#### [MOVE] Domains to Features
*   `ocr/detection` -> `ocr/features/detection`
*   `ocr/recognition` -> `ocr/features/recognition`
*   `ocr/kie` -> `ocr/features/kie`
*   **Update**: All imports `ocr.detection` -> `ocr.features.detection`, etc.

### Phase 3: Split Inference

#### [MOVE] Domain Inference
*   `ocr/inference/recognizer.py` -> `ocr/features/recognition/inference/recognizer.py`
*   `ocr/inference/backends/` -> `ocr/features/recognition/inference/backends/`
*   `ocr/inference/extraction/` -> `ocr/features/kie/inference/extraction/` (Extraction is heavily KIE-related)

#### [MOVE] Core Inference to `ocr/core/inference`
*   Remaining `ocr/inference` contents (engine, orchestrator, manager, pipelines) -> `ocr/core/inference/`.
*   Layout logic (`ocr/inference/layout`) -> `ocr/core/inference/layout` (Generic layout analysis).
*   **Delete**: Empty `ocr/inference`.

### Phase 4: Consolidate Mixed Directories

#### [MERGE] `ocr/utils` -> `ocr/core/utils`
*   **Action**: Move unique files to `ocr/core/utils`. Delete `ocr/utils`.

#### [MOVE] `ocr/models` -> `ocr/core/models`
*   **Action**: Move contents to `ocr/core/models`. Delete `ocr/models`.

### Phase 5: Root Cleanup

#### [MOVE] Auxiliary Directories
*   `ocr/evaluation` -> `ocr/core/evaluation`.
*   `ocr/analysis` -> `ocr/core/analysis`.
*   `ocr/validation` -> `ocr/core/validation`.

## Verification Plan

### Automated Tests
Run existing tests to ensure no import errors or missing modules.
```bash
pytest tests/
```

### Manual Verification
1.  **Filesystem Check**:
    -   `list_dir ocr/` should ONLY show: `core`, `features`, `data` (if not moved to core).
    -   `list_dir ocr/features` should show: `detection`, `recognition`, `kie`.
2.  **Import Check**:
    -   Grep for `from ocr.detection` (should be `ocr.features.detection`).
    -   Grep for `from ocr.inference` (should be `ocr.core.inference` or `ocr.features...`).