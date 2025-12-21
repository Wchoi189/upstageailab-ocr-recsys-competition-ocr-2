---
title: "Shared Backend Package Contract"
date: "2025-12-14 12:00 (KST)"
type: "specification"
category: "architecture"
status: "draft"
version: "1.0"
tags: ['shared-backend', 'api-contract', 'inference-engine']
related_plan: "docs/artifacts/implementation_plans/2025-12-14_1746_implementation_plan_domain-driven-backends.md"
---

# Shared Backend Package Contract

**Package Location**: `apps/shared/backend_shared/`

This document defines the API contract for the shared backend package used by domain-specific backends (OCR Console, Playground Console). It specifies the modules, classes, functions, and data models that must be implemented to support common inference operations.

---

## Table of Contents

1. [Package Structure](#package-structure)
2. [InferenceEngine API](#inferenceengine-api)
3. [Pydantic Models (Data Contracts)](#pydantic-models-data-contracts)
4. [Environment Variables](#environment-variables)
5. [Import Paths](#import-paths)
6. [Migration Notes](#migration-notes)

---

## Package Structure

```
apps/shared/backend_shared/
├── __init__.py                     # Package exports
├── inference/
│   ├── __init__.py                 # Inference module exports
│   └── engine.py                   # InferenceEngine class (re-exported from ocr.inference.engine)
├── models/
│   ├── __init__.py                 # Models module exports
│   └── inference.py                # Pydantic v2 models for inference requests/responses
└── README.md                       # Usage documentation
```

---

## InferenceEngine API

### Location
`apps/shared/backend_shared/inference/engine.py`

### Source
Re-exported from `ocr.inference.engine.InferenceEngine` (proven implementation)

### Class: `InferenceEngine`

OCR Inference Engine for real-time predictions. Handles model loading, preprocessing, inference, and postprocessing.

#### Constructor

```python
def __init__(self) -> None:
    """Initialize the inference engine.

    Automatically detects device (CUDA if available, otherwise CPU).
    Model is not loaded until load_model() is called.
    """
```

**Attributes** (after initialization):
- `model`: PyTorch model (None until loaded)
- `device`: str - "cuda" or "cpu"
- `config`: Model configuration object (None until loaded)

---

#### Method: `load_model`

```python
def load_model(
    self,
    checkpoint_path: str,
    config_path: str | None = None
) -> bool:
    """Load a model from checkpoint.

    Args:
        checkpoint_path: Absolute or relative path to .ckpt file
        config_path: Optional path to config.yaml (auto-detected if None)

    Returns:
        bool: True if model loaded successfully, False otherwise

    Side Effects:
        - Sets self.model to loaded PyTorch model
        - Sets self.config to loaded configuration
        - Sets self.device based on CUDA availability

    Notes:
        - Config file is auto-detected by searching:
          1. Same directory as checkpoint
          2. Configs directory (PROJECT_ROOT/configs/)
        - Looks for: config.yaml, hparams.yaml, train.yaml, predict.yaml
    """
```

**Usage Example**:
```python
from apps.shared.backend_shared.inference import InferenceEngine

engine = InferenceEngine()
success = engine.load_model("outputs/experiments/train/ocr/checkpoint.ckpt")
if not success:
    raise RuntimeError("Failed to load model")
```

---

#### Method: `predict_array` (Primary API)

```python
def predict_array(
    self,
    image_array: np.ndarray,
    binarization_thresh: float | None = None,
    box_thresh: float | None = None,
    max_candidates: int | None = None,
    min_detection_size: int | None = None,
    return_preview: bool = True,
) -> dict[str, Any] | None:
    """Run inference on a numpy array (optimized, no file I/O).

    Args:
        image_array: Image as numpy array (BGR format, OpenCV standard)
        binarization_thresh: Override binarization threshold (default: from config)
        box_thresh: Override box threshold for NMS (default: from config)
        max_candidates: Override max detection candidates (default: from config)
        min_detection_size: Override minimum detection size in pixels (default: from config)
        return_preview: If True, maps polygons to 640x640 preview space and attaches base64 preview image.
                       If False, returns polygons in ORIGINAL image space.

    Returns:
        dict with keys:
        - "polygons": str - Space-separated coordinates, regions separated by "|"
                           Format: "x1 y1 x2 y2 x3 y3 x4 y4|x1 y1 x2 y2 x3 y3 x4 y4|..."
        - "texts": list[str | None] - Recognized text (may be None if OCR not run)
        - "confidences": list[float] - Confidence scores (0.0 to 1.0)
        - "preview_image_base64": str | None - Base64 JPEG preview (only if return_preview=True)
        - "meta": dict | None - Coordinate system metadata (only if return_preview=True)

        Returns None on failure.

    Coordinate Systems:
        - return_preview=True: Polygons in 640x640 preview space (resized/padded)
        - return_preview=False: Polygons in ORIGINAL image space

    Notes:
        - Image must be in BGR format (OpenCV standard)
        - If RGB, convert with: cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        - Preview image is JPEG encoded (quality=85) to reduce size (~10x smaller than PNG)
    """
```

**Usage Example**:
```python
import cv2
from apps.shared.backend_shared.inference import InferenceEngine

# Load image in BGR format (OpenCV standard)
image = cv2.imread("test.jpg")

# Run inference
engine = InferenceEngine()
engine.load_model("checkpoint.ckpt")

result = engine.predict_array(
    image_array=image,
    binarization_thresh=0.3,
    box_thresh=0.5,
    return_preview=True  # Get 640x640 preview with base64 image
)

# Access results
polygons_str = result["polygons"]  # "x1 y1 ... | x1 y1 ..."
confidences = result["confidences"]  # [0.95, 0.87, ...]
preview_base64 = result["preview_image_base64"]  # Base64 JPEG
meta = result["meta"]  # Coordinate system metadata
```

---

#### Method: `predict_image` (Legacy API)

```python
def predict_image(
    self,
    image_path: str,
    binarization_thresh: float | None = None,
    box_thresh: float | None = None,
    max_candidates: int | None = None,
    min_detection_size: int | None = None,
    return_preview: bool = True,
) -> dict[str, Any] | None:
    """Run inference on an image file (legacy path-based API).

    Args:
        image_path: Path to image file
        (other args same as predict_array)

    Returns:
        Same as predict_array

    Notes:
        - Kept for backward compatibility
        - New code should use predict_array() for better performance
        - Handles EXIF orientation automatically
    """
```

---

#### Method: `update_postprocessor_params`

```python
def update_postprocessor_params(
    self,
    binarization_thresh: float | None = None,
    box_thresh: float | None = None,
    max_candidates: int | None = None,
    min_detection_size: int | None = None,
) -> None:
    """Update postprocessor hyperparameters in-place.

    Args:
        binarization_thresh: Threshold for binarization (0.0 to 1.0)
        box_thresh: Threshold for box filtering/NMS (0.0 to 1.0)
        max_candidates: Maximum number of detections to return
        min_detection_size: Minimum detection size in pixels

    Notes:
        - Updates are persistent for subsequent predict_array/predict_image calls
        - Pass None to keep current value unchanged
    """
```

---

## Pydantic Models (Data Contracts)

### Location
`apps/shared/backend_shared/models/inference.py`

These models provide type safety and validation for API requests/responses. They align with TypeScript types in frontend applications.

### Model: `Padding`

```python
from pydantic import BaseModel, Field

class Padding(BaseModel):
    """Padding applied during preprocessing."""
    top: int = Field(default=0, ge=0)
    bottom: int = Field(default=0, ge=0)
    left: int = Field(default=0, ge=0)
    right: int = Field(default=0, ge=0)
```

---

### Model: `InferenceMetadata`

```python
class InferenceMetadata(BaseModel):
    """Metadata describing coordinate system and transformations.

    Critical for frontend coordinate handling and overlay alignment.
    """
    original_size: tuple[int, int] = Field(
        ...,
        description="Original image size (width, height) before preprocessing"
    )
    processed_size: tuple[int, int] = Field(
        ...,
        description="Processed image size (width, height) after resize/padding (typically 640x640)"
    )
    padding: Padding = Field(
        ...,
        description="Padding applied during preprocessing"
    )
    padding_position: str = Field(
        default="top_left",
        description="Padding position: 'top_left', 'center', etc."
    )
    content_area: tuple[int, int] = Field(
        ...,
        description="Content area size (width, height) within processed_size frame"
    )
    scale: float = Field(
        ...,
        description="Scaling factor applied during resize",
        gt=0
    )
    coordinate_system: str = Field(
        default="pixel",
        description="Coordinate system: 'pixel' (absolute) or 'normalized' (0-1)"
    )
```

---

### Model: `TextRegion`

```python
class TextRegion(BaseModel):
    """Detected text region with polygon coordinates."""
    polygon: list[list[float]] = Field(
        ...,
        description="Polygon vertices as [[x1, y1], [x2, y2], ...] (typically 4 points for quadrilaterals)",
        min_length=3  # At least 3 points for a valid polygon
    )
    text: str | None = Field(
        None,
        description="Recognized text (None if recognition not performed)"
    )
    confidence: float = Field(
        ...,
        description="Detection confidence score",
        ge=0.0,
        le=1.0
    )
```

---

### Model: `InferenceRequest`

```python
class InferenceRequest(BaseModel):
    """Request body for inference endpoint."""
    checkpoint_path: str = Field(
        ...,
        description="Path to model checkpoint (.ckpt file)"
    )
    image_base64: str | None = Field(
        None,
        description="Base64-encoded image data (with or without data URL prefix)"
    )
    image_path: str | None = Field(
        None,
        description="Path to image file (absolute or relative to PROJECT_ROOT)"
    )
    confidence_threshold: float = Field(
        default=0.3,
        description="Binarization threshold (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    nms_threshold: float = Field(
        default=0.5,
        description="Box threshold for NMS (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "checkpoint_path": "outputs/experiments/train/ocr/checkpoint.ckpt",
                "image_base64": "data:image/png;base64,iVBORw0KG...",
                "confidence_threshold": 0.3,
                "nms_threshold": 0.5
            }
        }
```

---

### Model: `InferenceResponse`

```python
class InferenceResponse(BaseModel):
    """Response body for inference endpoint."""
    status: str = Field(
        default="success",
        description="Response status: 'success' or 'error'"
    )
    regions: list[TextRegion] = Field(
        default_factory=list,
        description="Detected text regions"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Inference processing time in milliseconds",
        ge=0.0
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Optional notes or warnings"
    )
    preview_image_base64: str | None = Field(
        None,
        description="Base64-encoded preview image (JPEG) for overlay alignment"
    )
    meta: InferenceMetadata | None = Field(
        None,
        description="Coordinate system metadata (critical for frontend rendering)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "regions": [
                    {
                        "polygon": [[10, 10], [100, 10], [100, 50], [10, 50]],
                        "text": "Sample text",
                        "confidence": 0.95
                    }
                ],
                "processing_time_ms": 150.5,
                "notes": [],
                "preview_image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
                "meta": {
                    "original_size": [1920, 1080],
                    "processed_size": [640, 640],
                    "padding": {"top": 0, "bottom": 280, "left": 0, "right": 0},
                    "padding_position": "top_left",
                    "content_area": [640, 360],
                    "scale": 0.333,
                    "coordinate_system": "pixel"
                }
            }
        }
```

---

## Environment Variables

Backend applications using the shared package should support these environment variables:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OCR_CHECKPOINT_PATH` | str | None | Path to default OCR checkpoint |
| `PLAYGROUND_CHECKPOINT_PATH` | str | None | Path to default Playground checkpoint |
| `MODEL_DEVICE` | str | "auto" | Device for inference: "cuda", "cpu", or "auto" |
| `BACKEND_HOST` | str | "127.0.0.1" | Backend server host |
| `BACKEND_PORT` | int | 8001/8002 | Backend server port (8001=Playground, 8002=OCR) |
| `PROJECT_ROOT` | str | Auto-detected | Project root directory (for resolving relative paths) |
| `CONFIG_DIR` | str | "{PROJECT_ROOT}/configs" | Configuration directory |
| `OUTPUT_DIR` | str | "{PROJECT_ROOT}/outputs" | Output directory for experiments |

---

## Import Paths

### Inference Engine

```python
# Primary import
from apps.shared.backend_shared.inference import InferenceEngine

# Alternative (if __init__.py exports at package level)
from apps.shared.backend_shared import InferenceEngine
```

### Pydantic Models

```python
# Import specific models
from apps.shared.backend_shared.models.inference import (
    InferenceRequest,
    InferenceResponse,
    TextRegion,
    InferenceMetadata,
    Padding,
)

# Alternative (if __init__.py exports at package level)
from apps.shared.backend_shared.models import (
    InferenceRequest,
    InferenceResponse,
    TextRegion,
)
```

---

## Migration Notes

### From Archived Unified Backend

The shared backend package consolidates functionality from the deprecated unified backend:

**Before** (Deprecated):
```python
# apps/backend/services/ocr_bridge.py
class OCRBridge:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.checkpoint_path = checkpoint_path
        self.engine = None  # Lazy loading

    def predict(self, image) -> tuple:
        # Returns (boxes, scores) tuple
        ...
```

**After** (Shared Package):
```python
# apps/shared/backend_shared/inference/engine.py
from apps.shared.backend_shared.inference import InferenceEngine

engine = InferenceEngine()
engine.load_model(checkpoint_path)
result = engine.predict_array(image_array)
# Returns dict with 'polygons', 'confidences', 'texts' keys
```

### Key Differences

1. **Return Format**:
   - Old: tuple `(boxes: list[np.ndarray], scores: list[float])`
   - New: dict `{"polygons": str, "confidences": list[float], "texts": list[str], ...}`

2. **Coordinate System**:
   - Old: Always original image coordinates
   - New: Configurable via `return_preview` parameter (original or 640x640 preview)

3. **Lazy Loading**:
   - Old: Model loaded on first predict() call
   - New: Explicit `load_model()` call required

4. **Type Safety**:
   - Old: No Pydantic models, manual dict construction
   - New: Pydantic v2 models for all requests/responses

---

## Implementation Checklist

To implement the shared backend package, complete these tasks:

- [ ] Create `apps/shared/backend_shared/__init__.py` with exports
- [ ] Create `apps/shared/backend_shared/inference/__init__.py`
- [ ] Create `apps/shared/backend_shared/inference/engine.py` (re-export from ocr.inference.engine)
- [ ] Create `apps/shared/backend_shared/models/__init__.py`
- [ ] Create `apps/shared/backend_shared/models/inference.py` with Pydantic models
- [ ] Create `apps/shared/backend_shared/README.md` with usage examples
- [ ] Update OCR backend to import from shared package
- [ ] Update Playground backend to import from shared package
- [ ] Add smoke tests for shared package imports

---

## References

- **Implementation Plan**: [2025-12-14_1746_implementation_plan_domain-driven-backends.md](../implementation_plans/2025-12-14_1746_implementation_plan_domain-driven-backends.md)
- **Source InferenceEngine**: [ocr/inference/engine.py](../../../ocr/inference/engine.py)
- **Archived OCR Bridge**: [docs/archive/archive_code/deprecated/apps-backend/services/ocr_bridge.py](../../archive/archive_code/deprecated/apps-backend/services/ocr_bridge.py)
- **Archived Playground Router**: [docs/archive/archive_code/deprecated/apps-backend/services/playground_api/routers/inference.py](../../archive/archive_code/deprecated/apps-backend/services/playground_api/routers/inference.py)
- **Backend Setup Guide**: [docs/guides/setting-up-app-backends.md](../../guides/setting-up-app-backends.md)

---

**Document Status**: Draft
**Last Updated**: 2025-12-14
**Next Review**: After Phase 1 implementation
