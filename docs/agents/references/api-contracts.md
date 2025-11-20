# API Contracts Quick Reference

## Overview

**Full Documentation:** [`docs/api/pipeline-contract.md`](../../api/pipeline-contract.md)

API contracts define the expected request/response formats, endpoints, and validation rules for the OCR pipeline API. **Always review API contracts before modifying API code to prevent API violations.**

## When to Review

**REQUIRED before:**
- Adding new API endpoints
- Modifying request/response formats
- Changing API validation logic
- Updating error responses
- Modifying authentication/authorization

## Critical Endpoints

### POST /predict
**Request:**
```json
{
  "image": "base64_encoded_image_string",
  "options": {
    "preprocessing": true,
    "format": "json"
  }
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "recognized text",
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "confidence": 0.95
    }
  ],
  "metadata": {
    "processing_time_ms": 150,
    "model_version": "1.0.0"
  }
}
```

### POST /batch_predict
**Request:**
```json
{
  "images": ["base64_1", "base64_2", ...],
  "options": {...}
}
```

**Response:** Array of prediction results

### GET /health
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600
}
```

## Request Validation

**Required fields:**
- `image` or `images` (depending on endpoint)
- Valid base64 encoding
- Image size limits (check docs for current limits)

**Optional fields:**
- `options`: Configuration for preprocessing/postprocessing
- `format`: Response format (json, simplified, etc.)

## Error Responses

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Image decoding failed",
    "details": "..."
  }
}
```

**Error Codes:**
- `INVALID_INPUT`: Bad request format
- `PROCESSING_ERROR`: OCR pipeline error
- `MODEL_ERROR`: Model inference error
- `SERVER_ERROR`: Internal server error

## Common Pitfalls

❌ **Missing validation** (no input checks)
❌ **Wrong response format** (doesn't match contract)
❌ **Missing error handling** (no error responses)
❌ **Breaking changes** (incompatible updates)
❌ **Missing version** (no API versioning)

## Quick Validation Pattern

```python
from pydantic import BaseModel, validator

class PredictRequest(BaseModel):
    image: str
    options: dict = {}

    @validator('image')
    def validate_image(cls, v):
        try:
            # Validate base64
            import base64
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 image")
        return v
```

## Prevention Checklist

Before modifying API code:
- [ ] Read relevant API contract section
- [ ] Understand expected request/response formats
- [ ] Add validation for all inputs
- [ ] Test with sample requests
- [ ] Verify contract compliance
- [ ] Update API documentation if needed

## Links

- **Full API Contracts**: [`docs/api/pipeline-contract.md`](../../api/pipeline-contract.md)
- **API Documentation**: [`docs/api/`](../../api/)
- **FastAPI Endpoints**: `services/api/routes.py`
- **Request Validation**: `services/api/validation.py`

## Examples

### Good: Contract-Compliant Endpoint
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class PredictRequest(BaseModel):
    image: str
    options: dict = {}

@router.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Decode image
        image = decode_base64_image(request.image)

        # Run OCR
        result = ocr_pipeline(image, **request.options)

        # Return contract-compliant response
        return {
            "predictions": result.predictions,
            "metadata": result.metadata
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Processing error")
```

### Bad: Contract Violation
```python
@app.post("/predict")
def predict(data):
    # ❌ No request validation
    # ❌ No error handling
    # ❌ Wrong response format
    result = model(data["image"])
    return result
```

## API Versioning

When making breaking changes:
1. Create new API version (e.g., `/v2/predict`)
2. Maintain old version for backward compatibility
3. Document migration path
4. Set deprecation timeline

## Additional Resources

- [API Architecture](../../api/architecture.md)
- [FastAPI Guide](../../api/fastapi-guide.md)
- [Request Validation](../../api/validation.md)
