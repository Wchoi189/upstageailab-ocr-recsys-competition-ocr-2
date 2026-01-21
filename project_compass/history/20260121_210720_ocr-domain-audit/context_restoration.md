# Context Restoration: Surgical Core Audit 2

**Source**: `session_handover.md` (Legacy)
**Date**: 2026-01-21

## Critical Context

The previous session identified **Critical Architecture Failures** in `ocr/core` and `ocr/data`:

1. **Eager Import Hell**: Importing one model triggers a cascade that imports *every* model.
2. **Dataset Contamination**: `ocr/data/datasets` contains domain-specific logic (Detection/Recognition).
3. **Registry Anti-Pattern**: Current registry forces eager loading.

## Immediate Action Items

### 1. The Data Purge
- Move `advanced_detector.py` -> `ocr/domains/detection/data/preprocessing/`
- Move `augments.py` -> `ocr/domains/detection/data/transforms/`
- Audit `ocr/data` for any file using `Box` or `Polygon`.

### 2. Registry Demolition
- Stop using `@registry.register`.
- Use Hydra `_target_` directly in YAMLs.

## Blockers
- `ocr/data` depends on `scipy`, `cv2`, `numpy`, making generic data loading heavy.
- Validation schemas are checked at import time (Eager Validation).

## Reference
- See `project_compass/vault/milestones/ocr-domain-refactor.md` for full roadmap.
