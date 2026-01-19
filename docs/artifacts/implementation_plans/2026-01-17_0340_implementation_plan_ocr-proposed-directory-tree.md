---
category: architecture
status: active
type: implementation_plan
date: 2024-10-18 10:00 (KST)
title: OCR Proposed Directory Tree Structure (Comprehensive)
version: 1.0.0
ads_version: "1.0"
---

# OCR Proposed Directory Tree Structure (Comprehensive)

> [!IMPORTANT]
> **Final Architecture**: "**Domains First**".
> **Constraint 1**: `ocr/core` contains *only* domain-agnostic utilities.
> **Constraint 2**: `ocr/pipelines` orchestrates domains (Bridge Pattern).
> **Constraint 3**: Domains NEVER import other domains directly (use Core Interfaces).

## 1. Directory Tree & Migration Map

Legend: `[NEW]` = New Directory | `(<- Source)` = Source file location

```text
ocr/
├── core/                                       # [STRICT] Generic Utilities
│   ├── interfaces/                             # [NEW] Data Contracts (The Bridge)
│   │   └── schemas.py                          (New file: Box, DetectionResult, PageResult)
│   ├── utils/
│   │   ├── image_loading.py                    (<- ocr/core/utils/image_loading.py)
│   │   ├── image_utils.py                      (<- ocr/core/utils/image_utils.py)
│   │   ├── logging.py                          (<- ocr/core/utils/logging.py [Stripped of domain logic])
│   │   └── path_utils.py                       (<- ocr/core/utils/path_utils.py)
│   ├── infrastructure/                         # [NEW] System Services
│   │   ├── agents/                             (<- ocr/agents/*)
│   │   └── comm/                               (<- ocr/communication/*)
│   └── architecture.py                         (<- ocr/core/models/architecture.py [Base Classes ONLY])
│
├── pipelines/                                  # [NEW] Orchestration Layer
│   ├── base_pipeline.py                        (New file)
│   ├── engine.py                               (<- ocr/core/inference/engine.py)
│   └── orchestrator.py                         (<- ocr/core/inference/orchestrator.py)
│
├── domains/                                    # [NEW] Root for Domain Logic
│   │
│   ├── detection/                              (Merged from ocr/features/detection & ocr/core split)
│   │   ├── analysis/
│   │   │   └── visualize.py                    (<- ocr/core/analysis/validation/analyze_worst_images.py)
│   │   ├── callbacks/
│   │   │   └── wandb.py                        (<- ocr/core/utils/wandb_utils.py [Polygon Logic])
│   │   ├── data/
│   │   │   ├── collate_db.py                   (<- ocr/data/datasets/db_collate_fn.py)
│   │   │   └── collate_craft.py                (<- ocr/data/datasets/craft_collate_fn.py)
│   │   ├── inference/
│   │   │   ├── coordinate_manager.py           (<- ocr/core/inference/coordinate_manager.py)
│   │   │   └── perspective/                    (<- ocr/core/utils/perspective_correction/*)
│   │   ├── metrics/
│   │   │   ├── box_types.py                    (<- ocr/core/metrics/box_types.py)
│   │   │   └── functional.py                   (<- ocr/core/metrics/eval_functions.py)
│   │   ├── utils/
│   │   │   ├── geometry.py                     (<- ocr/core/utils/geometry_utils.py)
│   │   │   ├── polygons.py                     (<- ocr/core/utils/polygon_utils.py)
│   │   │   └── visualization.py                (<- ocr/core/utils/ocr_utils.py [draw_boxes])
│   │   └── models/                             (<- ocr/features/detection/models/*)
│   │
│   ├── recognition/                            (Merged from ocr/features/recognition & ocr/core split)
│   │   ├── callbacks/
│   │   │   └── wandb.py                        (<- ocr/features/recognition/callbacks/wandb_image_logging.py)
│   │   ├── data/
│   │   │   ├── collate.py                      (<- ocr/data/datasets/recognition_collate_fn.py)
│   │   │   └── tokenizer.py                    (<- ocr/features/recognition/data/tokenizer.py)
│   │   ├── models/                             (<- ocr/features/recognition/models/*)
│   │   ├── module.py                           (<- ocr/core/lightning/ocr_pl.py [Recognition Parts])
│   │   └── utils/
│   │       └── visualization.py                (<- ocr/core/utils/text_rendering.py)
│   │
│   ├── kie/                                    (<- ocr/features/kie/*)
│   └── layout/                                 (<- ocr/features/layout/*)
│
└── data/                                       # Generic Data Infrastructure
    ├── datasets/
    │   └── base.py                             (<- ocr/data/datasets/base.py)
    └── transforms/                             (<- ocr/data/datasets/transforms.py)
```

## 2. Detailed Migration Table (Critical Fixes)

### A. The "Geometry Leak" Fix
| File                                     | Destination                                | Rationale               |
| :--------------------------------------- | :----------------------------------------- | :---------------------- |
| `ocr/core/utils/geometry_utils.py`       | `domains/detection/utils/geometry.py`      | Detection-specific math |
| `ocr/core/utils/polygon_utils.py`        | `domains/detection/utils/polygons.py`      | Detection-specific math |
| `ocr/core/utils/perspective_correction/` | `domains/detection/inference/perspective/` | Detection preprocessing |

### B. The "Inference Engine" Fix
| File                                 | Destination                     | Rationale                                  |
| :----------------------------------- | :------------------------------ | :----------------------------------------- |
| `ocr/core/inference/engine.py`       | `ocr/pipelines/engine.py`       | Imports both domains -> Must be above them |
| `ocr/core/inference/orchestrator.py` | `ocr/pipelines/orchestrator.py` | Orchestration logic                        |

### C. The "Splitter" Tasks
| File                 | Action    | Destination                                                                                                                                  |
| :------------------- | :-------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **`wandb_utils.py`** | **SPLIT** | `domains/detection/callbacks/wandb.py` (Polygons)<br>`domains/recognition/callbacks/wandb.py` (Text)<br>`core/utils/logging.py` (Base setup) |
| **`ocr_pl.py`**      | **SPLIT** | `domains/detection/module.py`<br>`domains/recognition/module.py`<br>`core/lightning/base.py`                                                 |

## 3. Success Metrics (Architecture Checks)
1.  `grep "from ocr.domains" ocr/core` -> **MUST BE EMPTY**
2.  `grep "from ocr.domains.recognition" ocr/domains/detection` -> **MUST BE EMPTY**
3.  `grep "ocr.core.utils.geometry" ocr/pipelines/` -> **FAIL** (Should import from domains)
