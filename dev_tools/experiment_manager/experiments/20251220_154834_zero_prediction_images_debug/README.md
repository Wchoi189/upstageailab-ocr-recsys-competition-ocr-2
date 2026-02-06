---
ads_version: "1.0"
type: experiment_manifest
experiment_id: "20251220_154834_zero_prediction_images_debug"
status: "completed"
created: "2025-12-20T15:48:34.936608+09:00"
updated: "2025-12-21T03:18:00+09:00"
tags: ["sepia", "image-enhancement", "ocr-preprocessing"]
---

# Experiment: Sepia Enhancement for Zero-Prediction OCR

## Overview

This experiment investigates sepia color transformation as a robust alternative to gray-scale and gray-world normalization for OCR preprocessing.

**Problem**: Certain low-contrast or aged document images (e.g., 000712, 000732) produce zero or limited OCR predictions using standard normalization methods.
**Hypothesis**: Sepia enhancement (specifically adaptive contrast via CLAHE) provides superior character-to-background differentiation, leading to more reliable OCR detection.

## Results Summary 🏆

- **Best Method**: `sepia_clahe` (Reddish Tint + Adaptive Contrast)
- **Edge Improvement**: **+164.0%** (vs baseline 0%)
- **Contrast Boost**: **+8.2** (vs baseline 0)
- **Insight**: Correct reddish tint mapping is critical; initial greenish tints provided 23% less edge clarity.

## Documentation Index

This experiment follows EDS v1.0. All detailed findings and guides are located in the `.metadata/` directory:

| Artifact | Purpose |
| :--- | :--- |
| [Final Report](.metadata/reports/20251221_0316_report_final-sepia-vs-clahe-performance-analysis.md) | Detailed performance analysis and metrics. |
| [Quick Start Guide](.metadata/guides/sepia_quick_start.md) | Commands for running enhancement and comparison scripts. |
| [Testing Plan](.metadata/plans/sepia_testing_plan.md) | The original workflow and success criteria. |
| [Status Log](.metadata/00-status/2025-12-21_sepia_testing_progress.md) | Final progress update and timeline. |

## Experiment Structure

```
.
├── scripts/              # Processing and benchmarking scripts
├── artifacts/            # Reference images and test samples
├── outputs/              # Generated results and metrics
└── .metadata/            # EDS-compliant documentation
```

---

*Verified by Antigravity AI | ETK v1.0.0 | EDS v1.0*
