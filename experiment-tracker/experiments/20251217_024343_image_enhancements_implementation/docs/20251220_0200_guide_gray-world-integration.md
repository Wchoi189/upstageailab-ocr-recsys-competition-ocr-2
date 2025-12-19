---
ads_version: "1.0"
type: "implementation_guide"
experiment_id: "20251217_024343_image_enhancements_implementation"
status: "active"
created: "2025-12-20T02:00:00Z"
updated: "2025-12-20T02:00:00Z"
tags: ['background-normalization', 'gray-world', 'pipeline-integration', 'preprocessing']
priority: "high"
validated_performance:
  tint_reduction: "75% (58.1 → 14.6)"
  processing_time: "~20ms per image"
  user_confirmed: true
target_pipelines:
  - inference
  - training
  - ui_apps
---

# Gray-World Background Normalization Integration Guide

## Overview

This guide explains how to integrate the gray-world background normalization enhancement into:
1. **Inference pipeline** (production OCR)
2. **Training dataset** (data preprocessing)
3. **UI applications** (console apps)

**Validated Performance**:
- Tint reduction: 75% improvement (58.1 → 14.6)
- Processing time: ~20ms per image
- User-confirmed effective for zero predictions

---

## Step 1: Create Shared Utility Module

**Create: `ocr/utils/background_normalization.py`**

```python
"""Background normalization utilities for document preprocessing."""

import numpy as np


def normalize_gray_world(img: np.ndarray) -> np.ndarray:
    """
    Apply gray-world white balance assumption.

    Assumes average color of scene should be neutral gray.
    Scales each channel so its mean equals the global average.

    Args:
        img: Input image (BGR format, uint8)

    Returns:
        Normalized image (BGR format, uint8)
    """
    # Calculate channel means
    b_avg, g_avg, r_avg = img.mean(axis=(0, 1))

    # Calculate gray average
    gray_avg = (b_avg + g_avg + r_avg) / 3

    # Avoid division by zero
    if b_avg == 0 or g_avg == 0 or r_avg == 0:
        return img.copy()

    # Calculate scale factors
    scale_factors = np.array([
        gray_avg / b_avg,
        gray_avg / g_avg,
        gray_avg / r_avg
    ])

    # Apply scaling
    result = img.astype(np.float32) * scale_factors
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
```

---

## Step 2: Inference Pipeline Integration

### 2a. Update Configuration Schema

**File: `ocr/inference/config_loader.py`**

**Add to `PreprocessSettings` dataclass** (around line 30):

```python
@dataclass(slots=True)
class PreprocessSettings:
    image_size: tuple[int, int]
    normalization: NormalizationSettings
    enable_background_normalization: bool = False  # ADD THIS LINE
```

### 2b. Update Preprocessing Function

**File: `ocr/inference/preprocess.py`**

**Import the utility** (add to top imports):

```python
from ocr.utils.background_normalization import normalize_gray_world
```

**Modify `preprocess_image()` function** (around line 60-70):

Find this section:
```python
def preprocess_image(
    image: Any, transform: Callable[[Any], Any], target_size: int = 640, return_processed_image: bool = False
) -> Any | tuple[Any, Any]:
```

Add normalization **before** LongestMaxSize resizing:

```python
def preprocess_image(
    image: Any,
    transform: Callable[[Any], Any],
    target_size: int = 640,
    return_processed_image: bool = False,
    enable_background_normalization: bool = False  # ADD THIS PARAMETER
) -> Any | tuple[Any, Any]:
    """Apply preprocessing transform to an image and return a batched tensor.

    Args:
        image: BGR numpy array (OpenCV convention)
        transform: Torchvision transform pipeline
        target_size: Target size for LongestMaxSize and PadIfNeeded
        return_processed_image: If True, also return processed BGR image
        enable_background_normalization: If True, apply gray-world normalization

    Returns:
        Batched tensor ready for model inference, or tuple of (tensor, processed_image_bgr)
    """
    # Work on a copy to avoid modifying the input image in place
    processed_image = image.copy()

    # STEP 1: Background normalization (NEW - before any resizing)
    if enable_background_normalization:
        processed_image = normalize_gray_world(processed_image)

    original_h, original_w = processed_image.shape[:2]

    # STEP 2: LongestMaxSize - scale longest side to target_size
    # ... rest of existing code ...
```

### 2c. Update YAML Configuration

**File: `configs/_base/preprocessing.yaml`** (or wherever preprocessing configs are):

```yaml
preprocessing:
  image_size: [640, 640]
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  enable_background_normalization: true  # ADD THIS
```

### 2d. Update High-Level Preprocessing Caller

**File: `ocr/inference/preprocess.py`**

Find where `preprocess_image()` is called from `apply_optional_perspective_correction()` or similar wrappers:

```python
def apply_optional_perspective_correction(
    image_bgr: Any,
    enable_perspective_correction: bool,
    enable_background_normalization: bool = False,  # ADD THIS
    return_matrix: bool = False,
) -> Any | tuple[Any, Any]:
    """
    Optionally apply rembg-based perspective correction and background normalization.

    Args:
        image_bgr: Input image in BGR format
        enable_perspective_correction: If False, image returned unchanged
        enable_background_normalization: If True, apply gray-world normalization
        return_matrix: If True, return tuple of (corrected_image, transform_matrix)

    Returns:
        Potentially corrected and normalized BGR image
    """

    if not enable_perspective_correction:
        if return_matrix:
            import numpy as np
            return image_bgr, np.eye(3, dtype=np.float32)
        return image_bgr

    try:
        image_no_bg, mask = remove_background_and_mask(image_bgr)
        if return_matrix:
            corrected, _result, matrix = correct_perspective_from_mask(image_no_bg, mask, return_matrix=True)
        else:
            corrected, _result = correct_perspective_from_mask(image_no_bg, mask)

        # Apply background normalization AFTER perspective correction
        if enable_background_normalization:
            corrected = normalize_gray_world(corrected)

        if return_matrix:
            return corrected, matrix
        else:
            return corrected
    except Exception as exc:
        LOGGER.warning("Perspective correction failed: %s", exc)
        if return_matrix:
            import numpy as np
            return image_bgr, np.eye(3, dtype=np.float32)
        return image_bgr
```

---

## Step 3: Training Dataset Integration

**File: `ocr/datasets/base.py`**

### 3a. Add Configuration Field

**Find `DatasetConfig` class** (in `ocr/datasets/schemas.py`):

```python
class DatasetConfig(BaseModel):
    image_path: Path
    annotation_path: Path | None = None
    image_extensions: list[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    preload_images: bool = False
    preload_maps: bool = False
    load_maps: bool = False
    prenormalize_images: bool = False
    enable_background_normalization: bool = False  # ADD THIS LINE
    cache_config: CacheConfig = CacheConfig()
    image_loading_config: ImageLoadingConfig = ImageLoadingConfig()
```

### 3b. Apply in `__getitem__`

**File: `ocr/datasets/base.py`**

**Find `__getitem__()` method** (around line 410):

Add normalization **after** image loading but **before** transform:

```python
def __getitem__(self, idx: int) -> dict[str, Any]:
    """Get a sample from the dataset by index."""

    # ... existing cache check code ...

    # Step 2 - Image Loading
    image_filename = list(self.anns.keys())[idx]
    image_data = self._load_image_data(image_filename)
    image_array = image_data.image_array

    # NEW: Apply background normalization before transform
    if self.config.enable_background_normalization:
        from ocr.utils.background_normalization import normalize_gray_world
        image_array = normalize_gray_world(image_array)

    # ... rest of existing code (polygon processing, transform, etc.) ...
```

### 3c. Update Training Config

**File: `configs/data/train.yaml`** (or similar):

```yaml
dataset:
  image_path: ${paths.data_dir}/train/images
  annotation_path: ${paths.data_dir}/train/annotations.json
  preload_images: false
  enable_background_normalization: true  # ADD THIS
  cache_config:
    cache_transformed_tensors: false
```

---

## Step 4: UI Apps Integration

### 4a. Shared Inference Engine

**File: `apps/shared/backend_shared/inference.py`** (create if doesn't exist)

```python
"""Shared inference engine for UI applications."""

import cv2
import numpy as np
from ocr.utils.background_normalization import normalize_gray_world


class InferenceEngine:
    """Unified inference engine with configurable preprocessing."""

    def __init__(
        self,
        checkpoint_path: str,
        enable_perspective_correction: bool = True,
        enable_background_normalization: bool = True,  # ADD THIS
    ):
        self.checkpoint_path = checkpoint_path
        self.enable_perspective_correction = enable_perspective_correction
        self.enable_background_normalization = enable_background_normalization
        # ... existing initialization ...

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply configured preprocessing steps."""
        processed = image.copy()

        # Step 1: Background normalization (NEW)
        if self.enable_background_normalization:
            processed = normalize_gray_world(processed)

        # Step 2: Perspective correction
        if self.enable_perspective_correction:
            from ocr.utils.perspective_correction import correct_perspective_from_mask, remove_background_and_mask
            try:
                no_bg, mask = remove_background_and_mask(processed)
                processed, _ = correct_perspective_from_mask(no_bg, mask)
            except Exception as e:
                LOGGER.warning(f"Perspective correction failed: {e}")

        return processed

    def run_inference(self, image: np.ndarray) -> dict:
        """Run full inference pipeline."""
        # Preprocess
        processed = self.preprocess_image(image)

        # ... existing inference code ...
```

### 4b. Playground Console

**File: `apps/playground-console/backend/routers/inference.py`**

Update initialization to pass flag:

```python
@router.post("/preview", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run OCR inference on an image."""

    if _inference_engine is None:
        raise HTTPException(status_code=503, detail="InferenceEngine not initialized")

    # Decode image
    image_bytes = base64.b64decode(request.image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Enable background normalization from request (default True)
    enable_bg_norm = request.enable_background_normalization if hasattr(request, 'enable_background_normalization') else True

    # Update engine setting
    _inference_engine.enable_background_normalization = enable_bg_norm

    # Run inference
    result = _inference_engine.run_inference(image)

    # ... rest of existing code ...
```

### 4c. OCR Inference Console

**File: `apps/ocr-inference-console/backend/main.py`**

Similar updates for initialization and toggle control.

---

## Step 5: Testing & Validation

### Test Inference Pipeline

```bash
# Test with background normalization enabled
python runners/predict.py \
    checkpoint=epoch-18_step-001957.ckpt \
    data.test_path=data/zero_prediction_worst_performers \
    preprocessing.enable_background_normalization=true

# Compare with baseline (disabled)
python runners/predict.py \
    checkpoint=epoch-18_step-001957.ckpt \
    data.test_path=data/zero_prediction_worst_performers \
    preprocessing.enable_background_normalization=false
```

### Test Training Dataset

```python
# Test script
from ocr.datasets.base import ValidatedOCRDataset
from ocr.datasets.schemas import DatasetConfig

config = DatasetConfig(
    image_path=Path("data/train/images"),
    annotation_path=Path("data/train/annotations.json"),
    enable_background_normalization=True
)

dataset = ValidatedOCRDataset(config, transform=your_transform)
sample = dataset[0]

# Verify normalization was applied
print(f"Image shape: {sample['image'].shape}")
```

### Test UI Apps

```bash
# Start playground console
cd apps/playground-console
npm run dev

# Test with sample tinted background images
# Verify "Enable Background Normalization" toggle works
```

---

## Configuration Matrix

| Pipeline | Config File | Parameter | Default |
|----------|------------|-----------|---------|
| **Inference** | `configs/_base/preprocessing.yaml` | `enable_background_normalization` | `true` |
| **Training** | `configs/data/train.yaml` | `dataset.enable_background_normalization` | `false` |
| **Validation** | `configs/data/val.yaml` | `dataset.enable_background_normalization` | `false` |
| **UI Apps** | Runtime flag | `InferenceEngine(..., enable_background_normalization=True)` | `true` |

**Important**:
- **Training/validation should default to FALSE** to avoid data augmentation conflicts
- **Inference should default to TRUE** for production benefit
- **UI apps should provide toggle** for A/B testing

---

## Performance Metrics

### Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Tint** | 58.1 | 14.6 | -75% ✅ |
| **Processing Time** | - | ~20ms | Acceptable ✅ |
| **Zero Predictions** | Baseline | Reduced | User-confirmed ✅ |

### Monitoring

Add metrics logging:

```python
import time

def normalize_gray_world(img: np.ndarray) -> np.ndarray:
    """Apply gray-world normalization with timing."""
    start = time.time()

    # ... normalization code ...

    elapsed_ms = (time.time() - start) * 1000
    LOGGER.debug(f"Background normalization: {elapsed_ms:.1f}ms")

    return result
```

---

## Rollback Plan

If issues arise:

1. **Inference**: Set `enable_background_normalization: false` in YAML config
2. **Training**: Already defaults to `false`, no action needed
3. **UI Apps**: Toggle off in UI or set engine parameter to `false`

No code removal required - feature is gated by configuration flags.

---

## Summary

**Files to Create**:
- `ocr/utils/background_normalization.py` (new utility)

**Files to Modify**:
- `ocr/inference/config_loader.py` (add config field)
- `ocr/inference/preprocess.py` (apply normalization)
- `ocr/datasets/schemas.py` (add config field)
- `ocr/datasets/base.py` (apply in `__getitem__`)
- `apps/shared/backend_shared/inference.py` (add to engine)
- `apps/playground-console/backend/routers/inference.py` (pass flag)
- `apps/ocr-inference-console/backend/main.py` (pass flag)

**Config Files to Update**:
- `configs/_base/preprocessing.yaml` (inference)
- `configs/data/train.yaml` (training - keep false)
- `configs/data/val.yaml` (validation - keep false)

**Estimated Effort**:
- Implementation: 2-3 hours
- Testing: 1-2 hours
- Total: **3-5 hours**

**Risk**: Low (feature-flagged, validated performance)
