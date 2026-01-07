# Phase 1 Experimental Modules Archive

**Archived:** 2025-12-24
**Reason:** Superseded by enhanced_pipeline.py and advanced_detector.py
**Original Location:** `ocr.data.datasets/preprocessing/`

## Overview

These modules were part of Phase 1 experimentation for Office Lens quality document detection and preprocessing. They have been superseded by newer, more integrated implementations.

## Archived Modules

1. **advanced_corner_detection.py** - Advanced corner detection algorithms
2. **geometric_document_modeling.py** - Geometric modeling for document validation
3. **high_confidence_decision_making.py** - Confidence-based processing decisions
4. **corner_selection.py** - Corner selection utilities
5. **geometry_validation.py** - Geometric validation utilities
6. **phase1_validation_framework.py** - Phase 1 validation test harness

## Usage

These modules were only used in:
- Archived legacy UI code (`/archive/legacy_ui_code/`)
- Phase 1 experimental testing

## Superseded By

- **`ocr.data.datasets/preprocessing/enhanced_pipeline.py`** - `EnhancedDocumentPreprocessor` class
- **`ocr.data.datasets/preprocessing/advanced_detector.py`** - `AdvancedDocumentDetector` class
- **`ocr.data.datasets/preprocessing/pipeline.py`** - `DocumentPreprocessor` class (stable baseline)

## Restoration

If these modules are needed, they can be restored from this archive or from git history:
```bash
git log --all --full-history -- "ocr.data.datasets/preprocessing/advanced_corner_detection.py"
```

## See Also

- Phase 3 Audit Resolution Plan: `docs/artifacts/implementation_plans/2025-12-24_1236_implementation_plan_audit-resolution.md`
- OCR Pipeline Evolution documentation
