# Session Handover: Source Code Bridge Implemented

**Status**: Phase 3.5 (Bridge Implementation) Complete.
**Ready For**: Legacy Cleanup & Refactor.

## Context
- **Implemented**: `OCRProjectOrchestrator` in `ocr/pipelines/orchestrator.py`.
- **Refactored**: `runners/train.py` now delegates entirely to the Orchestrator.
- **Key Feature**: explicit `vocab_size` injection for Recognition models is now handled in the Orchestrator, replacing fragile logic in `get_pl_modules_by_cfg`.

## Immediate Next Steps (New Session)
1.  **Legacy Cleanup**:
    -   Verify `ocr/core/lightning/__init__.py` (specifically `get_pl_modules_by_cfg`) is no longer used.
    -   Delete or deprecate the legacy factory.
2.  **Refactor**:
    -   Review `ocr/pipelines/orchestrator.py` for any needed polish (e.g., adding Detection-specific paths if they differ significantly).
3.  **Phase 4 (Deferred)**:
    -   Once cleanup is done, run the full training restoration test.

## Critical Files
- `ocr/pipelines/orchestrator.py` (The new brain)
- `runners/train.py` (The entry point)
- `ocr/core/lightning/__init__.py` (To be deleted/cleaned)
