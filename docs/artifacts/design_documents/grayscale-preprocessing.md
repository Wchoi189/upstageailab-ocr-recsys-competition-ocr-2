---
title: "Grayscale Preprocessing API Integration"
date: "2025-12-16 (KST)"
type: "specification"
category: "architecture"
status: "completed"
version: "1.0"
component: "grayscale_preprocessing"
---

# Grayscale Preprocessing API Integration

**Purpose**: User-activated grayscale conversion for OCR inference; API flag `enable_grayscale: true` enables feature.

---

## Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Preprocessing Pipeline** | `ocr/inference/preprocessing_pipeline.py` | Grayscale conversion (BGR→GRAY→BGR) after perspective correction |
| **Inference Engine** | `ocr/inference/engine.py` | Runtime parameter support in `predict_array()`, `predict_image()` |
| **Orchestrator** | `ocr/inference/orchestrator.py` | Parameter delegation to preprocessing pipeline |
| **API Models** | `apps/shared/backend_shared/models/inference.py` | `enable_grayscale: bool = False` field |
| **Backend Endpoints** | `apps/ocr-inference-console/backend/main.py` | API integration |
| **Frontend UI** | `apps/ocr-inference-console/src/components/Sidebar.tsx` | "Enable Grayscale" checkbox |

---

## Data Flow

| Step | Component | Action |
|------|-----------|--------|
| 1 | **User Request** | `enable_grayscale: true` |
| 2 | **Backend** | Receives `InferenceRequest` |
| 3 | **InferenceEngine** | `predict_array(enable_grayscale=True)` |
| 4 | **Orchestrator** | Delegates to preprocessing pipeline |
| 5 | **Preprocessing** | Applies perspective correction (if enabled) |
| 6 | **Grayscale Conversion** | BGR → GRAY → BGR (maintains 3-channel) |
| 7 | **Standard Pipeline** | Resize, pad, normalize |
| 8 | **Inference** | Model processes grayscale image |

---

## API Usage

### Basic Request

**Request**:
```json
{
  "checkpoint_path": "outputs/experiments/train/ocr/checkpoint.ckpt",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "confidence_threshold": 0.3,
  "nms_threshold": 0.5,
  "enable_grayscale": true
}
```

**Response**: Standard `InferenceResponse`
- `status`: "success"
- `regions`: Detected text regions with polygons
- `meta`: Coordinate transformation metadata
- `preview_image_base64`: Grayscale preview image

### Combined with Perspective Correction

**Request**:
```json
{
  "checkpoint_path": "outputs/experiments/train/ocr/checkpoint.ckpt",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "confidence_threshold": 0.3,
  "nms_threshold": 0.5,
  "enable_perspective_correction": true,
  "perspective_display_mode": "corrected",
  "enable_grayscale": true
}
```

**Processing Order**:
1. Perspective correction (if enabled)
2. Grayscale conversion (if enabled)
3. Resize and padding
4. Normalization

---

## Implementation Details

### Grayscale Conversion Logic

```python
# In preprocessing_pipeline.py (Stage 2)
if enable_grayscale:
    import cv2
    # Convert BGR → GRAY
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert GRAY → BGR (maintain 3-channel input for model)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
```

**Rationale**: Convert back to BGR to maintain 3-channel input expected by the OCR model.

---

## Feature Compatibility

| Feature | Works with Grayscale | Notes |
|---------|---------------------|-------|
| **Perspective Correction** | ✅ Yes | Grayscale applied AFTER perspective correction |
| **Display Modes** | ✅ Yes | Works with both "corrected" and "original" modes |
| **Preview Image** | ✅ Yes | Preview shows grayscale when enabled |
| **All Models** | ✅ Yes | Model-agnostic (3-channel input maintained) |

---

## Configuration

| Method | Implementation | Precedence |
|--------|----------------|------------|
| **Runtime Parameter** (recommended) | `engine.predict_array(enable_grayscale=True)` | High |
| **Config File** | Not supported (runtime-only parameter) | N/A |

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| **opencv-python** | Default | Color space conversion (cvtColor) |
| **numpy** | Default | Array operations |

---

## Constraints

- **Processing Time**: Minimal overhead (~1-5ms per image)
- **Memory**: No additional memory required
- **Channel Requirement**: Maintains 3-channel BGR format for model compatibility

---

## Backward Compatibility

**Status**: Maintained (default `false`)

**Breaking Changes**: None

**Compatibility Matrix**:

| Interface | v1.0 | Notes |
|-----------|------|-------|
| InferenceRequest API | ✅ Compatible | New optional field `enable_grayscale: bool = False` |
| Inference Engine | ✅ Compatible | Runtime parameter optional |
| Preprocessing Pipeline | ✅ Compatible | Applied after perspective correction |

---

## User Insight

> "Grayscale images are incredibly effective at converting zero prediction images to full prediction capable images."

This feature enables users to leverage grayscale preprocessing for improved OCR accuracy on certain document types, particularly those with color artifacts or poor contrast.

---

## Frontend Integration

### UI Control (Sidebar)

```tsx
<label className="flex items-center gap-2 text-sm">
  <input
    type="checkbox"
    checked={enableGrayscale}
    onChange={(e) => onGrayscaleChange(e.target.checked)}
    className="rounded border-gray-300"
  />
  <span>Enable Grayscale</span>
</label>
```

### State Flow

1. **Sidebar** → `enableGrayscale` state
2. **App.tsx** → State management
3. **Workspace** → Pass to API client
4. **ocrClient.ts** → Include in request body
5. **Backend** → Process with grayscale conversion

---

## Implementation Status

| Phase | Status | Features |
|-------|--------|----------|
| **Phase 1: Basic Implementation** | ✅ Completed | Runtime API flag, grayscale conversion, preview support |

**Files Modified**:
- `apps/shared/backend_shared/models/inference.py` - Added `enable_grayscale` field
- `apps/ocr-inference-console/backend/main.py` - Parameter passing from request to engine
- `apps/ocr-inference-console/src/components/Sidebar.tsx` - UI checkbox control
- `apps/ocr-inference-console/src/App.tsx` - State management
- `apps/ocr-inference-console/src/components/Workspace.tsx` - Prop passing
- `apps/ocr-inference-console/src/api/ocrClient.ts` - API parameter
- `ocr/inference/engine.py` - Parameter delegation
- `ocr/inference/orchestrator.py` - Pipeline coordination
- `ocr/inference/preprocessing_pipeline.py` - Grayscale conversion implementation

---

## References

- [Perspective Correction API Integration](./perspective-correction-api-integration.md)
- [Inference Data Contracts](../../pipeline/inference-data-contracts.md)
- [Backend Pipeline Contract](../../backend/api/backend-pipeline-contract.md)
