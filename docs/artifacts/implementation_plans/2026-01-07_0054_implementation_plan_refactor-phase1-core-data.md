---
ads_version: "1.0"
version: "1.0"
type: "implementation_plan"
category: "development"
status: "completed"
tags: "refactoring, architecture, hydra, phase1, phase2"
title: "Refactor Phase 1 & 2: Core Foundation & Data Pivot"
date: "2026-01-07 00:54 (KST)"
branch: "main"
description: "Execute the foundational restructuring of the 'ocr' package and 'configs' directory, establishing 'ocr/core' and consolidating data components into 'ocr/recognition/data' (packaged by feature)."
---

# Implementation Plan - Refactor Phase 1 & 2: Core Foundation & Data Pivot

## Goal
Establish the 'Feature-First' architecture foundation by creating `ocr/core`, `configs/_foundation` and pivoting data components to `ocr/recognition/data`.

## User Review Required
> [!IMPORTANT]
> **Breaking Changes**: This refactor moves core components (`architecture.py`) and data foundations (`tokenizer.py`). All configuration files referencing these must be updated.
> - **Session Tracking**: Use `Project Compass` to log significant architectural changes.
> - **ADT Verification**: This plan relies on `agent-debug-toolkit` to verify config integrity.

## Proposed Changes

### 1. Core Structure (`ocr/core`)
Create the shared foundation to decouple generic logic from specific domains.

#### [NEW] `ocr/structure_manifest.json`
- Define the mapping of legacy paths to new paths for automated verification.

#### [NEW] `ocr/core/`
- Create directory.

#### [MOVE] `ocr/models/architecture.py` -> `ocr/core/architecture.py`
- Move the generic `OCRModel` class.

#### [MOVE] `ocr/models/core/*` -> `ocr/core/`
- Move `base_classes.py` and `registry.py` to the new core.

### 2. Config Foundation (`configs/_foundation`)
Align configuration structure with code structure.

#### [NEW] `configs/_foundation/`
- Create directory.

#### [MOVE] `configs/_base/*` -> `configs/_foundation/`
- Migrate base config fragments. Updates `defaults` lists in all configs to point to `_foundation`.

### 3. Data Pivot - Recognition (`ocr/recognition/data`)
Consolidate recognition-specific data logic.

#### [NEW] `ocr/recognition/data/`
- Create directory.

#### [MOVE] `ocr/data/tokenizer.py` -> `ocr/recognition/data/tokenizer.py`
- Move tokenizer logic.

#### [MOVE] `ocr/datasets/lmdb_dataset.py` -> `ocr/recognition/data/lmdb_dataset.py`
- Move LMDB dataset logic.

#### [MODIFY] `configs/data/recognition.yaml`
- Update `_target_` references to point to `ocr.recognition.data...`.

## Verification Plan

### Automated Verification
1.  **ADT Analysis**:
    ```bash
    adt analyze-config configs/ --output markdown
    ```
    - Check for broken references in configs.
    - Verify `adt trace-merges` on `configs/domain/recognition.yaml` (or equivalent) to ensure precedence is preserved.

2.  **Unit Tests**:
    ```bash
    pytest tests/ -v
    ```
    - Ensure existing tests pass (update imports in tests as needed).

### Manual Verification
1.  **Dry Run Training**:
    - Run a fast dev run to ensure the trainer can instantiate the model and data module.
    ```bash
    python ocr_pl.py experiment=test/fast_dev_run
    ```