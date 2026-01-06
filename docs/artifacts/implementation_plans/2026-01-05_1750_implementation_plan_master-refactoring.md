---
ads_version: "1.0"
type: implementation_plan
category: strategy
status: active
version: "1.0"
tags: refactoring, architecture, hydra, ocr
title: Master Architecture Refactoring Plan "Project Polaris"
date: "2026-01-05 17:50 (KST)"
branch: main
---

# Master Architecture Refactoring Plan: "Project Polaris"

**Date:** 2026-01-05 17:50 (KST)
**Type:** Strategic Master Plan
**Goal:** Align Codebase and Configuration into a "Feature-First" Architecture to maximize AI agency and developer productivity.
**Status:** DRAFT (Planning Phase)

## 1. Executive Summary

This plan unifies the **Hydra Restructuring Plan** (`2026-01-05_0441...`) and the **Code Architecture Assessment** (`2026-01-05_ocr_architecture...`) into a single execution roadmap.

**The "Nuclear" Concept:**
We shift from "Package by Layer" (models/metrics mixed together) to **"Package by Feature"** (Domain-First).
- **Code:** `ocr/{feature}/*` (Self-contained models, data, training logic)
- **Config:** `configs/domain/{feature}.yaml` (Self-contained entry points)

## 2. Target Architecture

### 2.1 Code Structure (`ocr/`)
```text
ocr/
├── core/                   # Shared foundations (Base classes, unified Loss, Utils)
│   ├── architecture.py     # The generic OCRModel wrapper
│   └── data/               # ValidatedOCRDataset (Generic)
│
├── detection/              # [FEATURE] Text Detection
│   ├── models/             # DBNet, CRAFT architectures & heads
│   ├── data/               # Detection-specific transforms/datasets
│   └── trainer.py          # Detection LightningModule
│
├── recognition/            # [FEATURE] Text Recognition
│   ├── models/             # PARSeq, CRNN
│   ├── data/               # Tokenizer, Charset, LMDB Dataset
│   └── trainer.py          # Recognition LightningModule
│
├── kie/                    # [FEATURE] Key Information Extraction
│   ├── models/             # KIE Architectures
│   └── trainer.py          # KIE LightningModule
│
└── utils/                  # Truly generic non-domain utils (logging, pathing)
```

### 2.2 Config Structure (`configs/`)
*Aligns 1:1 with Code Structure*
```text
configs/
├── _foundation/            # Base fragments (cf. ocr/core)
├── domain/                 # Domain Entry Points
│   ├── detection.yaml      # maps to ocr/detection
│   ├── recognition.yaml    # maps to ocr/recognition
│   └── kie.yaml            # maps to ocr/kie
└── model/                  # Component definitions (structured by domain)
    ├── detection/
    └── recognition/
```

## 3. Alignment Assessment

| Aspect           | Hydra Plan Proposal           | Code Plan Proposal            | Alignment     |
| :--------------- | :---------------------------- | :---------------------------- | :------------ |
| **Domains**      | Split into Det/Rec/KIE/Layout | Split into Det/Rec/KIE/Layout | ✅ **Perfect** |
| **Foundation**   | `_foundation/`                | `ocr/core/`                   | ✅ **Strong**  |
| **Entry Points** | `+domain=X`                   | `ocr/{domain}/trainer.py`     | ✅ **Strong**  |
| **Data**         | `configs/data/{domain}.yaml`  | `ocr/{domain}/data/`          | ✅ **Strong**  |

**Conclusion:** The two plans are mutually reinforcing. Refactoring one makes the other easier.

## 4. Execution Roadmap

We will execute in **Phases** to maintain a working state (or acceptable "construction state") throughout.

### Phase 1: Preparation (The "Manifest" & "Core")
*   **Goal:** Define the map and stabilize the foundation.
*   **Steps:**
    1.  Create `ocr/structure_manifest.json` (Logical map of where things *will* go).
    2.  Create `ocr/core/` and move `ocr/models/architecture.py`, `BaseHead`, `BaseDecoder` there.
    3.  Create `configs/_foundation/` and migrate `configs/_base/`.
    4.  **Verify:** Baseline tests pass.

### Phase 2: The "Data" Pivot (Consolidation)
*   **Goal:** Execute the `ocr/data` merged plan but direct it to features.
*   **Steps:**
    1.  Create `ocr/recognition/data/`.
    2.  Move `ocr/data/tokenizer.py`, `charset.json` -> `ocr/recognition/data/`.
    3.  Move `ocr/datasets/lmdb_dataset.py` -> `ocr/recognition/data/`.
    4.  Update `configs/data/recognition.yaml` to point to new paths.

### Phase 3: Feature Migration - Recognition
*   **Goal:** Fully isolate the Recognition domain (PARSeq).
*   **Steps:**
    1.  Create `ocr/recognition/models/`.
    2.  Move `PARSeq` components from `ocr/models/` to `ocr/recognition/models/`.
    3.  Move `ocr_pl.py` (relevant parts) to `ocr/recognition/trainer.py`.
    4.  Update `ocr/structure_manifest.json`.

### Phase 4: Feature Migration - KIE & Detection
*   **Goal:** Isolate remaining domains.
*   **Steps (Parallelizable):**
    1.  Move `kie_dataset.py`, `kie_pl.py`, `kie_models.py` to `ocr/kie/`.
    2.  Move `DBNet`/`CRAFT` components to `ocr/detection/`.

### Phase 5: Final Cleanup
*   **Goal:** Remove the scaffold.
*   **Steps:**
    1.  Delete empty `ocr/models/`, `ocr/data/`, `ocr/lightning_modules/`.
    2.  Update all Documentation and `AGENTS.yaml`.

## 5. Recommendation for Next Step

**Do NOT start Phase 3/4 yet.**
**START with Phase 2 (Data Pivot).**
Why? The user already identified `ocr/data` as a pain point. Moving `tokenizer` and `charset` to `ocr/recognition/data` immediately solves the specific "confusion" issue and sets the pattern for the rest of the refactor.

**Revised Immediate Plan (Session Goal):**
1.  **Refine** the Data Merge implementation plan to target `ocr/recognition/data` (Feature-based) instead of `ocr/datasets` (Layer-based).
2.  **Execute** Phase 2 (Data Pivot) in this session if execution is permitted, OR finalize this Master Plan if Planning Only.
