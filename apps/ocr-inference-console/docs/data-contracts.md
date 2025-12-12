# API Data Contracts

## POST /ocr/predict

Runs OCR inference on an uploaded image.

### Request

**Endpoint**: `POST http://localhost:8000/ocr/predict`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `file` (required): Image file (JPG, PNG, BMP)
- `checkpoint_path` (optional): Override default checkpoint path

**Example**:
```bash
curl -X POST http://localhost:8000/ocr/predict \
  -F "file=@image.jpg" \
  -F "checkpoint_path=/path/to/checkpoint.ckpt"
```

### Response

**Content-Type**: `application/json`

**TypeScript Interface**:
```typescript
interface InferenceResponse {
  filename: string;
  predictions: Prediction[];
}

interface Prediction {
  points: number[][];  // [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  confidence: number;   // 0.0 - 1.0
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

- **Origin**: Top-left corner of image (0, 0)
- **Points**: Pixel coordinates in original image space
- **Polygon**: 4+ points defining a quadrilateral or polygon
- **Order**: Points are ordered clockwise or counter-clockwise

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

- Backend implementation: [`apps/backend/services/ocr_bridge.py`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/backend/services/ocr_bridge.py)
- Frontend client: [`apps/ocr-inference-console/src/api/ocrClient.ts`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/src/api/ocrClient.ts)
- Data contract docs: [`docs/pipeline/inference-data-contracts.md`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/pipeline/inference-data-contracts.md)
