# Session Handover: Refactor Rescue & Strict Organization

## Session Summary
**Date**: 2026-01-08 23:55 (KST)
**Session ID**: Refactor_Preparation
**Objective**: Prepare state for the "Deep Refactor" session.

## 🛑 Instructions for Next Session

**Roadmap**: `project_compass/roadmap/00_source_code_refactoring2.yaml`

### Priority Goals
1.  **Execute Phase 1 (Cleanup)**: Delete identified duplicates (`ocr/lightning_modules`, `ocr/datasets`, `ocr/metrics`).
2.  **Execute Phase 2 (Feature Containerization)**: Create `ocr/features/` and move domains (`detection`, `recognition`, `kie`) into it.
3.  **Execute Phase 3 (Inference Split)**: Dismantle `ocr/inference`, moving core engines to `ocr/core/inference` and domain logic to `ocr/features/{domain}/inference`.

## Artifacts in Context
*   `project_compass/roadmap/00_source_code_refactoring2.yaml`: The master plan.
*   `implementation_plan.md`: Detailed steps (at artifacts path).

## Recommended Immediate Action
**Start the new session by executing the deletes using the `roadmap` as a guide. Be bold but verify imports.**
