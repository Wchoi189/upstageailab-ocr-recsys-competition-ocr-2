---
title: "Perspective Correction API Integration"
date: "2025-12-15 12:00 (KST)"
type: "design"
category: "architecture"
status: "completed"
version: "1.0"
ads_version: "1.0"
component: "perspective_correction"
---



# Perspective Correction API Integration

**Purpose**: User-activated rembg-based perspective correction for OCR inference; API flag `enable_perspective_correction: true` enables feature.

---

## Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Perspective Utilities** | `ocr/utils/perspective_correction.py` | `remove_background_and_mask()`, `fit_mask_rectangle()`, `four_point_transform()`, `correct_perspective_from_mask()` |
| **Preprocessing Integration** | `ocr/inference/preprocess.py` | `apply_optional_perspective_correction()` wrapper |
| **Inference Engine** | `ocr/inference/engine.py` | Runtime parameter support in `predict_array()`, `predict_image()`, `_predict_from_array()` |
| **API Models** | `apps/shared/backend_shared/models/inference.py` | `enable_perspective_correction: bool = False` field |
| **Backend Endpoints** | `apps/ocr-inference-console/backend/main.py`, `apps/playground-console/backend/routers/inference.py` | API integration |

---

## Data Flow

| Step | Component | Action |
|------|-----------|--------|
| 1 | **User Request** | `enable_perspective_correction: true` |
| 2 | **Backend** | Receives `InferenceRequest` |
| 3 | **InferenceEngine** | `predict_array(enable_perspective_correction=True)` |
| 4 | **Preprocessing** | `apply_optional_perspective_correction()` |
| 5 | **Rembg** | `remove_background_and_mask()` |
| 6 | **Edge Detection** | `fit_mask_rectangle()` (100% success rate) |
| 7 | **Warp** | `four_point_transform()` (Max-Edge rule, Lanczos4 interpolation) |
| 8 | **Inference** | Standard preprocessing and OCR on corrected image |

---

## API Usage

### Basic Request (Corrected Display)

**Request**:
```json
{
  "checkpoint_path": "outputs/experiments/train/ocr/checkpoint.ckpt",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "confidence_threshold": 0.3,
  "nms_threshold": 0.5,
  "enable_perspective_correction": true
}
```

**Response**: Standard `InferenceResponse`
- `status`: "success"
- `regions`: Detected text regions with polygons (in corrected image coordinate space)
- `meta`: Coordinate transformation metadata
- `preview_image_base64`: Corrected image (if enabled)

### Display Mode Control (Phase 2 - ✅ Implemented)

**Request with Original Display**:
```json
{
  "checkpoint_path": "outputs/experiments/train/ocr/checkpoint.ckpt",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "confidence_threshold": 0.3,
  "nms_threshold": 0.5,
  "enable_perspective_correction": true,
  "perspective_display_mode": "original"
}
```

**Display Modes**:
- `"corrected"` (default): Returns corrected image with annotations in corrected coordinate space
- `"original"`: Returns original image with annotations inverse-transformed to original coordinate space

**InferenceRequest Model**:
```python
class InferenceRequest(BaseModel):
    enable_perspective_correction: bool = Field(default=False)
    perspective_display_mode: str = Field(
        default="corrected",
        pattern="^(corrected|original)$",
        description="Display mode: 'corrected' or 'original'"
    )
```

---

## Coordinate Space Behavior

### Corrected Mode (Default)
**When `perspective_display_mode: "corrected"`**:
- Inference runs on corrected image
- Annotations in corrected image coordinate space
- Preview shows corrected image with annotations

### Original Mode (Phase 2)
**When `perspective_display_mode: "original"`**:
- Inference runs on corrected image (for accuracy)
- Annotations inverse-transformed back to original coordinate space
- Preview shows original image with transformed annotations
- Uses stored perspective transform matrix for inverse transformation

---

## Configuration

| Method | Implementation | Precedence |
|--------|----------------|------------|
| **Runtime Parameter** (recommended) | `engine.predict_array(enable_perspective_correction=True)` | High (takes precedence) |
| **Config File** (legacy) | `enable_perspective_correction: true` in YAML | Low |

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| **rembg** | >= 2.0.67 | Background removal, mask generation |
| **opencv-python** | Default | Image transformations |
| **numpy** | Default | Array operations |

---

## Constraints

- **Processing Time**: +1-3s per image (rembg processing)
- **Memory**: rembg model load (one-time cost)
- **Success Rate**: 100% for documents with clear backgrounds
- **Interpolation**: Lanczos4 ensures high-quality text preservation

---

## Backward Compatibility

**Status**: Maintained (default `false`)

**Breaking Changes**: None

**Compatibility Matrix**:

| Interface | v1.0 | Notes |
|-----------|------|-------|
| InferenceRequest API | ✅ Compatible | New optional field `enable_perspective_correction: bool = False` |
| Inference Engine | ✅ Compatible | Runtime parameter optional |
| Config YAML | ✅ Compatible | Legacy config still supported |

**Precedence**: Runtime parameter > Config file

---

## Implementation Status

| Phase | Status | Features |
|-------|--------|----------|
| **Phase 1: Basic Correction** | ✅ Completed | Runtime API flag, corrected image display |
| **Phase 2: Original Display** | ✅ Completed | Transform matrix storage, inverse transformation, display mode toggle |

**Files Modified**:
- `apps/shared/backend_shared/models/inference.py` - Added `perspective_display_mode` field with validation
- `apps/ocr-inference-console/backend/main.py` - Parameter passing from request to engine
- `ocr/inference/engine.py` - Parameter delegation to orchestrator
- `ocr/inference/orchestrator.py` - Inverse transformation logic, original image handling
- `ocr/inference/preprocessing_pipeline.py` - Original image capture and matrix storage

## Future Enhancements

**Phase 3** (Potential):
- Batch processing support with perspective correction
- Custom perspective transformation parameters
- Perspective quality metrics and validation

---

## References

- [Inference Data Contracts](../../pipeline/inference-data-contracts.md)
- [Backend Pipeline Contract](../../backend/api/backend-pipeline-contract.md)
- [System Architecture](../../architecture/system-architecture.md)
