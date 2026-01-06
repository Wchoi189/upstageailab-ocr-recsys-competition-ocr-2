---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "draft"
version: "1.0"
tags: "refactoring, architecture, hydra, phase4, detection, kie"
title: "Refactor Phase 4: Detection & KIE Feature Migration"
date: "2026-01-07 00:54 (KST)"
branch: "main"
description: "Migrate DBNet, CRAFT, and KIE models to their respective feature directories."
---

# Implementation Plan - Refactor Phase 4: Detection & KIE Feature Migration

## Goal
Isolate the **Detection** and **KIE** domains into their own feature packages.

## User Review Required
> [!WARNING]
> **Parallel Execution**: This phase can be executed in parallel with Phase 3 (Recognition) if resources allow, but sequential is safer for "one-agent" operation.

## Proposed Changes

### 1. KIE Feature (`ocr/kie`)

#### [NEW] `ocr/kie/`
- Create directory structure: `ocr/kie/models`, `ocr/kie/data`.

#### [MOVE] `ocr/kie_dataset.py` -> `ocr/kie/data/dataset.py`
- Move KIE dataset.

#### [MOVE] `ocr/kie_models.py` -> `ocr/kie/models/model.py`
- Move KIE model definitions.

#### [MOVE] `ocr/kie_pl.py` -> `ocr/kie/trainer.py`
- Move KIE LightningModule.

### 2. Detection Feature (`ocr/detection`)

#### [NEW] `ocr/detection/`
- Create directory structure.

#### [MOVE] `ocr/models/head/db_head.py`, `craft_head.py` -> `ocr/detection/models/`
- Move detection heads.

#### [MOVE] `ocr/models/head/db_postprocess.py`, `craft_postprocess.py` -> `ocr/detection/models/postprocess/`
- Move post-processing logic.

### 3. Final Cleanup (`ocr/models`)
Remove the scaffold.

#### [DELETE] `ocr/models/`
- Once empty (or delete remaining empty subdirs).

## Verification Plan

### Automated Verification
1.  **ADT Analysis**:
    ```bash
    adt full-analysis ocr/detection
    ```
    - comprehensive scan of the new module.

### Manual Verification
1.  **KIE Training Loop**:
    ```bash
    python ocr/kie/trainer.py ... # (or via hydra entry point)
    ```
