# Shared Backend Package

**Package**: `apps.shared.backend_shared`
**Version**: 1.0.0
**Status**: Active

Shared backend components for domain-specific FastAPI applications (OCR Console, Playground Console).

## Installation

Package is part of the monorepo and auto-discoverable via Python path.

```bash
# No installation needed - use direct imports
from apps.shared.backend_shared.inference import InferenceEngine
```

## Quick Start

### Basic Inference

```python
from apps.shared.backend_shared.inference import InferenceEngine
import cv2

# Initialize engine
engine = InferenceEngine()

# Load model
if not engine.load_model("outputs/experiments/train/ocr/checkpoint.ckpt"):
    raise RuntimeError("Failed to load model")

# Run inference
image = cv2.imread("test.jpg")  # BGR format
result = engine.predict_array(image)

# Access predictions
polygons_str = result["polygons"]  # "x1 y1 x2 y2 ... | ..."
confidences = result["confidences"]  # [0.95, 0.87, ...]
```

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from apps.shared.backend_shared.inference import InferenceEngine
from apps.shared.backend_shared.models.inference import (
    InferenceRequest,
    InferenceResponse,
    TextRegion,
)
import base64
import cv2
import numpy as np

app = FastAPI()
engine = InferenceEngine()

@app.on_event("startup")
async def startup():
    if not engine.load_model("checkpoint.ckpt"):
        raise RuntimeError("Failed to load model")

@app.post("/api/v1/inference/preview", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    # Decode base64 image
    if "base64," in request.image_base64:
        request.image_base64 = request.image_base64.split("base64,", 1)[1]

    image_bytes = base64.b64decode(request.image_base64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run inference
    result = engine.predict_array(
        image_array=image,
        binarization_thresh=request.confidence_threshold,
        box_thresh=request.nms_threshold,
        return_preview=True
    )

    # Parse to response model
    regions = []
    if result:
        polygons_str = result.get("polygons", "")
        confidences = result.get("confidences", [])
        texts = result.get("texts", [])

        for idx, polygon_str in enumerate(polygons_str.split("|")):
            if not polygon_str.strip():
                continue
            coords = [float(c) for c in polygon_str.strip().split()]
            polygon = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]

            regions.append(TextRegion(
                polygon=polygon,
                text=texts[idx] if idx < len(texts) else None,
                confidence=confidences[idx] if idx < len(confidences) else 0.0
            ))

    return InferenceResponse(
        status="success",
        regions=regions,
        preview_image_base64=result.get("preview_image_base64"),
        meta=result.get("meta")
    )
```

## API Reference

### InferenceEngine

**Import**: `from apps.shared.backend_shared.inference import InferenceEngine`

**Primary Methods**:
- `load_model(checkpoint_path, config_path=None) -> bool`
- `predict_array(image_array, **kwargs) -> dict | None`
- `predict_image(image_path, **kwargs) -> dict | None`

**See**: [Shared Backend Contract](../../../docs/artifacts/specs/shared-backend-contract.md) for full API documentation.

### Pydantic Models

**Import**: `from apps.shared.backend_shared.models.inference import ...`

**Models**:
- `InferenceRequest` - Request validation
- `InferenceResponse` - Response serialization
- `TextRegion` - Detected region
- `InferenceMetadata` - Coordinate metadata
- `Padding` - Preprocessing padding

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_CHECKPOINT_PATH` | None | Default OCR checkpoint path |
| `MODEL_DEVICE` | "auto" | Device: "cuda", "cpu", or "auto" |
| `PROJECT_ROOT` | Auto | Project root directory |

## Architecture

```
apps/shared/backend_shared/
├── __init__.py              # Package exports
├── inference/
│   ├── __init__.py          # Re-exports InferenceEngine
│   └── engine.py            # → ocr.inference.engine
├── models/
│   ├── __init__.py          # Model exports
│   └── inference.py         # Pydantic v2 models
└── README.md                # This file
```

## References

- **Contract**: [docs/artifacts/specs/shared-backend-contract.md](../../../docs/artifacts/specs/shared-backend-contract.md)
- **Implementation Plan**: [docs/artifacts/implementation_plans/2025-12-14_1746_implementation_plan_domain-driven-backends.md](../../../docs/artifacts/implementation_plans/2025-12-14_1746_implementation_plan_domain-driven-backends.md)
- **Backend Setup Guide**: [docs/guides/setting-up-app-backends.md](../../../docs/guides/setting-up-app-backends.md)
- **Source Engine**: [ocr/inference/engine.py](../../../ocr/inference/engine.py)

## Version History

- **1.0.0** (2025-12-14): Initial release with InferenceEngine re-export and Pydantic v2 models
