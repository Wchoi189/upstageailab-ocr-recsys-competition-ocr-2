---
ads_version: "1.0"
type: implementation_plan
category: code_quality
status: active
version: "1.0"
tags: cleanup, refactoring, data, schemas
title: Implementation Plan - Survey Follow-up & Cleanup
date: "2026-01-05 17:50 (KST)"
branch: main
---

# Implementation Plan - Survey Follow-up & Cleanup

**Goal:** Address "Immediate" and "Short-term" findings from the AI Workflow Efficiency Survey to prepare the environment for efficient execution. Focus is on structural cleanup (`ocr/data`), data contracts (Schemas), and documentation, without major Hydra refactoring.

## User Review Required
> [!IMPORTANT]
> **Hydra Impact:** This plan assumes `ocr.data.datasets.kie_dataset.KIEDataset` is NOT directly referenced in Hydra configs (verified via grep). if dynamic instantiation is used, it might require runtime fixes.
> **Breaking Change:** Moving `ocr/data/datasets/kie_dataset.py` to `ocr/datasets/kie_dataset.py` changes import paths.

## Proposed Changes

### Configuration & Data Structure Cleanup
#### [MODIFY] [ocr/data](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data)
- [DELETE] `ocr/data/` (Directory - **FULL MERGE**)
    - `ocr/data/datasets/kie_dataset.py` -> `ocr/datasets/kie_dataset.py`
    - `ocr/data/tokenizer.py` -> `ocr/datasets/tokenizer.py`
    - `ocr/data/charset.json` -> `ocr/datasets/charset.json`
    - `ocr/data/schemas/` -> `ocr/datasets/schemas/`

#### [MODIFY] Configuration & References
- [MODIFY] `configs/data/recognition.yaml`: Update charset path to `ocr/datasets/charset.json`.
- [MODIFY] `ocr/datasets/lmdb_dataset.py`: Update tokenizer import to `ocr.datasets.tokenizer`.
- [MODIFY] `ocr/datasets/kie_dataset.py`: Update imports if necessary.

### Data Contracts (Schemas)
#### [NEW] [ocr/datasets/schemas/lmdb.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/datasets/schemas/lmdb.py)
- Implement `LMDBSchema` using Pydantic V2.
- Define keys: `image-{idx:09d}`, `label-{idx:09d}`, `num-samples`.
- Enforce validation rules.

#### [NEW] [ocr/datasets/schemas/contracts.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/datasets/schemas/contracts.py)
- Define `OCRDatasetContract` for dataset outputs.

### Documentation
#### [MODIFY] [ocr/datasets/__init__.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/datasets/__init__.py)
- Update/Verify exports (add `KoreanOCRTokenizer`).
- Add docstrings to exposed classes if missing.

## Verification Plan

### Automated Tests
1.  **Run Existing Tests (if any):**
    - `pytest tests/` (targeted if KIE tests exist).
2.  **Import Verification:**
    - Create a small script `scripts/verify_imports.py` to try importing `ocr.datasets.kie_dataset` and `ocr.data.schemas.lmdb`.
    - Run: `uv run python scripts/verify_imports.py`

### Manual Verification
1.  **Check Project Structure:**
    - Verify `ocr/data/datasets` is gone.
    - Verify `ocr/datasets/kie_dataset.py` exists.
2.  **Schema Check:**
    - Instantiate `LMDBSchema` with sample data in a Python shell to verify validation logic.
