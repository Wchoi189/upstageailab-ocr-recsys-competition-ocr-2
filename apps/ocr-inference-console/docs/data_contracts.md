# OCR Inference Console Data Contracts

> [!NOTE]
> This document specifically addresses the data contracts relevant to the Inference Console. The core pipeline contracts are defined in [Data Contracts](../../../../docs/pipeline/data_contracts.md).

## Interface Contracts

### Backend API Bridge

The backend bridge exposes the OCR pipeline via REST API.

#### POST /predict

**Input**:
- `file`: Multipart file upload (image, JPEG/PNG).

**Output**:
```json
{
  "polygons": "x1,y1 x2,y2 ... | x1,y1 ...", // String representation
  "texts": ["text1", ...],
  "confidences": [0.99, ...],
  "preview_image_base64": "...",
  "meta": {
      "padding_position": "top_left",
      "content_area": [x, y, w, h],
      "original_size": [w, h],
      "processed_size": [w, h]
  }
}
```

See `docs/pipeline/data_contracts.md#inference-engine-contract` for detailed field definitions.
