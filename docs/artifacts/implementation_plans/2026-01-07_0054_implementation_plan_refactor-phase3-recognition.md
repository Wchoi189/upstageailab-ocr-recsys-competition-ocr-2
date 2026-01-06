---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "draft"
version: "1.0"
tags: "refactoring, architecture, hydra, phase3, recognition"
title: "Refactor Phase 3: Recognition Feature Migration"
date: "2026-01-07 00:54 (KST)"
branch: "main"
description: "Migrate PARSeq and recognition-specific components to 'ocr/recognition/models' and 'configs/model/recognition'."
---

# Implementation Plan - Refactor Phase 3: Recognition Feature Migration

## Goal
Fully isolate the **Recognition** domain by moving PARSeq models, encoders, and decoders into `ocr/recognition/models`.

## User Review Required
> [!WARNING]
> **Dependency**: This phase depends on the completion of Phase 1 & 2.
> **Breaking Changes**: Moving model components will break existing checkpoints if class paths are pickled (PyTorch Lightning sometimes does this, though Hydra instantiation uses import paths).
> - **Checkpoint Compatibility**: Verify if old checkpoints can load with new class paths (may need a migration script or `torch.package` re-mapping).

## Proposed Changes

### 1. Recognition Models (`ocr/recognition/models`)
Group all recognition-related model components.

#### [NEW] `ocr/recognition/models/`
- Create directory.

#### [MOVE] `ocr/models/head/parseq_head.py` -> `ocr/recognition/models/head.py`
- Move PARSeq head.

#### [MOVE] `ocr/models/encoder/parseq_encoder.py` -> `ocr/recognition/models/encoder.py`
- (Verify existence first, if not found, find where PARSeq encoder is defined).
- *Self-Correction*: If PARSeq encoder is part of a larger file, extract it.

#### [MOVE] `ocr/models/decoder/*` (PARSeq related) -> `ocr/recognition/models/decoder.py`
- Move PARSeq decoder logic.

### 2. Config Updates (`configs/model/recognition`)
Update Hydra configs to point to new model locations.

#### [NEW] `configs/model/recognition/`
- Create directory.

#### [MODIFY] `configs/model/parseq.yaml` (or equivalent)
- Update `_target_` to `ocr.recognition.models...`.

## Verification Plan

### Automated Verification
1.  **ADT Analysis**:
    ```bash
    adt find-instantiations ocr/recognition/models
    ```
    - Verify that components are correctly instantiated by existing configs (after update).

2.  **Unit Tests**:
    - Run recognition-specific tests (if any).
    ```bash
    pytest tests/models/test_parseq.py  # (Update path to actual test)
    ```

### Manual Verification
1.  **Overfit Test**:
    - Train on a tiny dataset (1 batch) to ensure convergence.
    ```bash
    python ocr_pl.py experiment=test/overfit_recognition
    ```
