# API Data Contracts

## POST /api/inference/preview

Runs OCR inference on an uploaded image and returns predictions with metadata.

### Request

**Endpoint**: `POST http://127.0.0.1:8000/api/inference/preview`

**Content-Type**: `application/json`

**Body**:
```typescript
interface InferencePreviewRequest {
  image_base64: string;  // Base64-encoded image
  checkpoint_path: string;
}
```

**Example**:
```bash
curl -X POST http://127.0.0.1:8000/api/inference/preview \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "...", "checkpoint_path": "/path/to/checkpoint.ckpt"}'
```

### Response

**Content-Type**: `application/json`

**TypeScript Interface**:
```typescript
interface InferencePreviewResponse {
  status: string;
  regions: TextRegion[];
  processing_time_ms: number;
  preview_image_base64?: string | null;  // Preprocessed image in processed_size space
  meta?: InferenceMetadata;
}

interface TextRegion {
  polygon: number[][];  // [[x1, y1], [x2, y2], ...] in processed_size space
  confidence: number;    // 0.0 - 1.0
  text?: string | null;
}

interface InferenceMetadata {
  original_size: [number, number];  // [width, height] of source image
  processed_size: [number, number]; // [width, height] of preprocessed image (typically 640x640)
  padding: {
    top: number;
    bottom: number;
    left: number;
    right: number;
  };
  scale: number;                     // target_size / max(original_h, original_w)
  coordinate_system: "pixel" | "normalized";
}

interface InferenceResponse {
  filename: string;
  predictions: Prediction[];
  meta?: InferenceMetadata;
  preview_image_base64?: string | null;
}

interface Prediction {
  points: number[][];  // [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  confidence: number;   // 0.0 - 1.0
  label?: string;
}
```

**Python Model** (Pydantic):
```python
from typing import List
from pydantic import BaseModel

class Prediction(BaseModel):
    points: List[List[float]]  # [[x1, y1], [x2, y2], ...]
    confidence: float

class InferenceResponse(BaseModel):
    filename: str
    predictions: List[Prediction]
```

**Example Response**:
```json
{
  "filename": "invoice.jpg",
  "predictions": [
    {
      "points": [[100, 50], [300, 50], [300, 100], [100, 100]],
      "confidence": 0.95
    },
    {
      "points": [[100, 150], [250, 150], [250, 180], [100, 180]],
      "confidence": 0.87
    }
  ]
}
```

### Coordinate System

**Important**: Coordinates are in `processed_size` space, not original image space.

- **Origin**: Top-left corner of processed image (0, 0)
- **Points**: Pixel coordinates relative to `processed_size` (typically 640x640)
- **Polygon**: 4+ points defining a quadrilateral or polygon
- **Order**: Points are ordered clockwise or counter-clockwise
- **Preview Image**: Use `preview_image_base64` for display - it matches the coordinate system

### Padding and Content Area

The preprocessed image includes black padding. The content area (actual image without padding) can be calculated:

```typescript
// For top-left padding (default)
const contentW = processed_size[0] - padding.right;
const contentH = processed_size[1] - padding.bottom;
const contentArea = { x: 0, y: 0, w: contentW, h: contentH };
```

See [Annotation Rendering](./annotation-rendering.md) for details on trimming padding during display.

### Error Responses

**503 Service Unavailable**: Model not loaded
```json
{
  "detail": "Model not loaded or checkpoint missing"
}
```

**500 Internal Server Error**: Inference failed
```json
{
  "detail": "Error message describing the failure"
}
```

### Validation

The frontend performs runtime validation to ensure the response matches the contract:
```typescript
if (!data.predictions || !Array.isArray(data.predictions)) {
    throw new Error('Backend returned unexpected format');
}
```

### Related

- Backend implementation: [`apps/backend/services/playground_api/routers/inference.py`](../../backend/services/playground_api/routers/inference.py)
- Frontend client: [`apps/ocr-inference-console/src/api/ocrClient.ts`](../src/api/ocrClient.ts)
- Annotation rendering: [`apps/ocr-inference-console/docs/annotation-rendering.md`](./annotation-rendering.md)
- Pipeline data contracts: [`docs/pipeline/inference-data-contracts.md`](../../docs/pipeline/inference-data-contracts.md)
