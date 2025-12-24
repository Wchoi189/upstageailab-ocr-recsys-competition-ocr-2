# BUG-20251224-001: ModuleNotFoundError 'ui' in OCR Callbacks

## Overview
**Date Detected:** 2025-12-24
**Severity:** High (Breaks metadata generation and legacy checkpointing)
**Status:** Resolved

## Description
During a smoke test of the OCR training pipeline, the `MetadataCallback` and `UniqueModelCheckpoint` failed with `ModuleNotFoundError: No module named 'ui'`. This occurred because these callbacks were still attempting to import types and services from the `ui.apps` namespace, which was either moved or removed during recent refactorings to decouple the OCR core from the UI components.

## Impact
- **Metadata Generation**: The `.metadata.yaml` files required for the Checkpoint Catalog V2 system were not being generated.
- **Training Logs**: The console was flooded with stack traces during checkpoint saving, making it difficult to monitor training progress.
- **Legacy Compatibility**: The `.metadata.json` files used by legacy systems were also not being created correctly.

## Root Cause
The refactoring that consolidated checkpoint utilities into `ocr.utils.checkpoints` did not update all call sites in the `ocr.lightning_modules.callbacks` package. Hardcoded imports like `from ui.apps.inference.services.checkpoint.metadata_loader import save_metadata` became invalid when the `ui` directory structure was reorganized or when the environment no longer included the `apps` path in the `PYTHONPATH`.

## Resolution
Updated the following files to use the consolidated `ocr.utils.checkpoints` package:

1.  **`ocr/lightning_modules/callbacks/metadata_callback.py`**:
    - Updated `TYPE_CHECKING` and runtime imports to use `ocr.utils.checkpoints.types` and `ocr.utils.checkpoints.metadata_loader`.
2.  **`ocr/lightning_modules/callbacks/unique_checkpoint.py`**:
    - Migrated from the legacy `CheckpointMetadataSchema` to the new `CheckpointMetadataV1`.
    - Updated all model imports to standard `ocr.utils.checkpoints.types`.

## Verification
- **Smoke Test**: Executed `runners/train.py` with `limit_train_batches=1`.
- **Logs**: Confirmed `Saved metadata to ...` log messages appeared without stack traces.
- **Artifacts**: Verified that both `best.metadata.yaml` and `last.metadata.yaml` exist in the experiment output directory and contain valid YAML data.

## Future Work
- Implement a linting rule or pre-commit hook to detect and prevent cross-package imports from `ocr` back into `apps` or `ui`.
- Decommission legacy `.metadata.json` generation once all downstream systems are migrated to the YAML-based Catalog V2.
