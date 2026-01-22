# Debugging Workspace: Decoder Refactor Failure ("Red Line" Collapse)

## Executive Summary
The text detection pipeline is experiencing a catastrophic failure ("Red Line" Failure) characterized by **Coordinate Saturation**. Even with >100 ground truth boxes, predictions collapse into a singular horizontal line at the canvas boundary.

**Key Symptoms:**
-   **Zero-Area Polygon Degeneracy**: Collapsed heights.
-   **Coordinate Clipping**: Values saturated/clipped to max dimensions.
-   **Resolution Mismatch**: Discrepancy between head output scale and post-processing expectations.

## Technical References
### Core Documentation
-   **Data Contracts**: `ocr/validation/models.py` (Pydantic models for `DataItem`, `TransformInput`, `TransformOutput`)
-   **Dataset Base**: `ocr/data/datasets/base.py` (`ValidatedOCRDataset` logic)
-   **Architecture**: `ocr/core/models/architecture.py` (`OCRModel` forward/generate)
-   **Post-Processing**: `ocr/core/utils/polygon_utils.py` (`get_polygons_from_maps`)

### AgentQMS Standards
-   **Coordinate Transforms**: [`AgentQMS/standards/tier2-framework/ocr-components/coordinate-transforms.yaml`](../../AgentQMS/standards/tier2-framework/ocr-components/coordinate-transforms.yaml) - *Critical for understanding padding/unpadding and inverse mapping logic.*
-   **Post-Processing**: [`AgentQMS/standards/tier2-framework/ocr-components/postprocessing-logic.yaml`](../../AgentQMS/standards/tier2-framework/ocr-components/postprocessing-logic.yaml) - *Defines binarization and contour extraction expectations.*
-   **Pipeline Contracts**: [`AgentQMS/standards/tier2-framework/ocr-components/pipeline-contracts.yaml`](../../AgentQMS/standards/tier2-framework/ocr-components/pipeline-contracts.yaml) - *Specifies the `InferenceMetadata` structure used for coordinate recovery.*
-   **Preprocessing**: [`AgentQMS/standards/tier2-framework/ocr-components/preprocessing-logic.yaml`](../../AgentQMS/standards/tier2-framework/ocr-components/preprocessing-logic.yaml) - *Details resize/pad operations that precede detection.*

### Relevant Artifacts
-   [Decoder Refactoring and Segfault Fix Walkthrough](artifacts/2026-01-15_1500_Decoder_Refactoring_and_Segfault_Fix_Walkthrough.md)
-   [PARSeq Inference Refactor Walkthrough](artifacts/2026-01-15_1500_Walkthrough-PARSeq_Inference_Refactor.md)

## Current Status
-   **Segfault**: Fixed (via float mask verification).
-   **Detection**: **BROKEN**. Regression observed after refactoring `OCRModel.forward` for detection loss logic.
