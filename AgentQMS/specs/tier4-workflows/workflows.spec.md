# Workflows Specification

**Tier**: 4 (Workflows)
**Scope**: Operational Workflows for Maintenance and Experiments.

## 1. Bloat Detection
*   **Trigger**: Weekly or Pre-Release.
*   **Goal**: Ensure efficiency.
*   **Checks**:
    *   Files > 300 lines? -> Split.
    *   Images > 5MB? -> Compress.
    *   Dead Code? -> `vulture` scan.

## 2. Experiments
*   **Trigger**: New Model or Hyperparameter search.
*   **Protocol**:
    1.  Create `design_document` for hypothesis.
    2.  Use `configs/experiment/X.yaml`.
    3.  Log results to `wandb`.
    4.  Create `walkthrough` artifact with results.

## 3. VLM Tools
*   **Purpose**: Analyze visual artifacts (charts, plots).
*   **Input**: Images in `docs/artifacts/vlm_reports/`.
*   **Output**: Markdown analysis.
