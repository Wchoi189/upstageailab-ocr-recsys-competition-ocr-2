---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "draft"
version: "1.0"
tags: ['sepia-enhancement', 'clahe', 'preprocessing', 'pipeline-integration', 'zero-prediction-fix']
title: "Sepia Enhancement (CLAHE) Integration for OCR Pipeline"
date: "2025-12-21 04:10 (KST)"
branch: "main"
validated_performance:
  edge_improvement: "+164.0%"
  contrast_boost: "+8.2"
  processing_time: "~25ms"
  target_images: "000712, 000732 (zero-prediction cases)"
reference_experiment: "experiment-tracker/experiments/20251220_154834_zero_prediction_images_debug"
---

# Sepia Enhancement (CLAHE) Integration

## Problem

Zero-prediction failures on low-contrast/aged document images (e.g., `000712`, `000732`) persist despite gray-world normalization. Experiment validated sepia+CLAHE achieves **+164% edge improvement** vs baseline, superior to gray-world alone.

## Scope

**Pipelines**: Inference, Training, UI
**Files Modified**: 8+ (config schemas, preprocessing, datasets, UI engines)
**New Files**: 1 (`ocr/utils/sepia_enhancement.py`)
**Estimated Effort**: 3-5 hours
**Risk**: Low (feature-flagged, validated)

---

## Implementation Phases

### Phase 1: Core Utility

**File**: `ocr/utils/sepia_enhancement.py` [NEW]

```python
"""Sepia enhancement with CLAHE for OCR preprocessing."""
from __future__ import annotations

import cv2
import numpy as np


def enhance_sepia_clahe(img: np.ndarray) -> np.ndarray:
    """Apply sepia + CLAHE enhancement for optimal OCR contrast.

    Best-performing method from zero-prediction experiment.
    +164% edge improvement, +8.2 contrast boost.

    Args:
        img: BGR numpy array (OpenCV convention), shape (H, W, 3)

    Returns:
        Enhanced BGR numpy array with same shape and dtype uint8
    """
    # Step 1: Warm sepia transformation (enhanced red/yellow channels)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    warm_matrix = np.array([
        [0.450, 0.850, 0.200],  # Red (strong boost)
        [0.350, 0.750, 0.150],  # Green (boosted)
        [0.200, 0.450, 0.100],  # Blue (reduced)
    ])
    sepia = cv2.transform(img_rgb, warm_matrix)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    sepia_bgr = cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)

    # Step 2: CLAHE on L channel (LAB space)
    lab = cv2.cvtColor(sepia_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])

    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
```

---

### Phase 2: Inference Integration

#### 2.1 Config Schema

**File**: [config_loader.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/inference/config_loader.py) `:32-36`

```diff
 @dataclass(slots=True)
 class PreprocessSettings:
     image_size: tuple[int, int]
     normalization: NormalizationSettings
     enable_background_normalization: bool = False
+    enable_sepia_enhancement: bool = False  # ADD
```

#### 2.2 Preprocessing Pipeline

**File**: [preprocess.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/inference/preprocess.py)

**Import** (after line 10):
```python
from ocr.utils.sepia_enhancement import enhance_sepia_clahe
```

**Function signature** (~line 45):
```diff
 def preprocess_image(
     image: Any,
     transform: Callable[[Any], Any],
     target_size: int = 640,
     return_processed_image: bool = False,
     enable_background_normalization: bool = False,
+    enable_sepia_enhancement: bool = False,  # ADD
 ) -> Any | tuple[Any, Any]:
```

**Apply enhancement** (after gray-world normalization block, ~line 72):
```python
# Apply sepia enhancement AFTER gray-world (if both enabled, sepia last)
if enable_sepia_enhancement:
    processed_image = enhance_sepia_clahe(processed_image)
```

#### 2.3 YAML Config

**File**: `configs/_base/preprocessing.yaml`

```yaml
preprocessing:
  image_size: [640, 640]
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  enable_background_normalization: true
  enable_sepia_enhancement: false  # ADD (default FALSE, enable per-image)
```

---

### Phase 3: Training Integration

#### 3.1 Dataset Schema

**File**: [schemas.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/datasets/schemas.py) (DatasetConfig)

```diff
 class DatasetConfig(BaseModel):
     # ... existing fields ...
     enable_background_normalization: bool = False
+    enable_sepia_enhancement: bool = False  # ADD
```

#### 3.2 Dataset `__getitem__`

**File**: [base.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/datasets/base.py) `:~410`

```python
# Apply sepia enhancement BEFORE transform (after gray-world if enabled)
if self.config.enable_sepia_enhancement:
    from ocr.utils.sepia_enhancement import enhance_sepia_clahe
    image_array = enhance_sepia_clahe(image_array)
```

#### 3.3 Training Config

**Files**: `configs/data/train.yaml`, `configs/data/val.yaml`

```yaml
dataset:
  # ... existing fields ...
  enable_sepia_enhancement: false  # DEFAULT FALSE
```

> [!IMPORTANT]
> Training/validation MUST default to `false` to prevent data distribution shift.

---

### Phase 4: UI Integration

#### 4.1 InferenceEngine (Main)

**File**: [engine.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/inference/engine.py)

Functions `predict_array` and `predict_image` already accept preprocessing flags. Add:

```diff
 def predict_array(
     ...
     enable_grayscale: bool = False,
     enable_background_normalization: bool = False,
+    enable_sepia_enhancement: bool = False,  # ADD
 ):
```

Pass to orchestrator/preprocessing calls.

#### 4.2 OCR Console Backend

**File**: [main.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/backend/main.py)

Update inference endpoint to accept `enable_sepia_enhancement` parameter:

```python
@router.post("/inference")
async def run_inference(...):
    # Add flag extraction and passing
    enable_sepia = request.get("enable_sepia_enhancement", False)
    result = engine.predict_array(..., enable_sepia_enhancement=enable_sepia)
```

#### 4.3 OCR Console Frontend

**File**: [App.tsx](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/src/App.tsx)

Add toggle in Sidebar (similar to existing grayscale/normalization toggles):

```tsx
<ToggleSwitch
  id="sepia-toggle"
  label="Sepia Enhancement"
  checked={inferenceOptions.enableSepiaEnhancement}
  onChange={(checked) => setInferenceOptions(prev => ({
    ...prev,
    enableSepiaEnhancement: checked
  }))}
/>
```

---

## Configuration Matrix

| Pipeline | Config Location | Parameter | Default | Rationale |
|----------|----------------|-----------|---------|-----------|
| Inference | `configs/_base/preprocessing.yaml` | `enable_sepia_enhancement` | `false` | Per-image toggle via UI |
| Training | `configs/data/train.yaml` | `dataset.enable_sepia_enhancement` | `false` | Avoid distribution shift |
| Validation | `configs/data/val.yaml` | `dataset.enable_sepia_enhancement` | `false` | Match training |
| UI Apps | Runtime parameter | `InferenceEngine(..., enable_sepia_enhancement=True)` | `false` | User toggle |

---

## Integration Order

```
Sepia vs Gray-World:
  Gray-world: neutralizes color tints
  Sepia: adds warm tones + adaptive contrast

Pipeline Position (if BOTH enabled):
  normalize_gray_world() → enhance_sepia_clahe() → Resize → Transform

Perspective Integration:
  RemoveBackground → CorrectPerspective → normalize_gray_world() → enhance_sepia_clahe()
```

---

## Validation Plan

### Unit Test

```bash
# Test sepia enhancement module
python -c "
import cv2
import numpy as np
from ocr.utils.sepia_enhancement import enhance_sepia_clahe

# Create test image
test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
result = enhance_sepia_clahe(test_img)
assert result.shape == test_img.shape
assert result.dtype == np.uint8
print('✓ Sepia enhancement module works')
"
```

### Integration Test

```bash
# Test inference with sepia enabled
python runners/predict.py \
    checkpoint=epoch-18_step-001957.ckpt \
    data.test_path=data/zero_prediction_worst_performers \
    preprocessing.enable_sepia_enhancement=true
```

### UI Manual Test

1. Start OCR Console: `make serve-ocr-console`
2. Upload problematic image (000712 or 000732)
3. Enable "Sepia Enhancement" toggle in Sidebar
4. Click "Run Inference"
5. Verify predictions appear (vs zero predictions without sepia)

---

## Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| Training distribution shift | Medium | Default `false` for training/validation |
| Interaction with gray-world | Low | Apply sepia AFTER gray-world |
| Performance overhead | Low | ~25ms validated, acceptable |

---

## Rollback Plan

**All components feature-flagged**:
1. **Inference**: Set `enable_sepia_enhancement: false` in YAML
2. **Training**: Already defaults to `false`
3. **UI**: Toggle off in interface

**No code removal required** - disable via configuration.

---

## File Change Summary

### New Files
- [sepia_enhancement.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/utils/sepia_enhancement.py) (~40 lines)

### Modified Files

| File | Changes |
|------|---------|
| [config_loader.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/inference/config_loader.py) | +1 field to `PreprocessSettings` |
| [preprocess.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/inference/preprocess.py) | +param, +import, +enhancement call |
| [schemas.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/datasets/schemas.py) | +1 field to `DatasetConfig` |
| [base.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/datasets/base.py) | +enhancement in `__getitem__` |
| [engine.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/inference/engine.py) | +param to `predict_array`, `predict_image` |
| [main.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/backend/main.py) | +flag passing |
| [App.tsx](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/src/App.tsx) | +toggle component |
| `configs/_base/preprocessing.yaml` | +enable_sepia_enhancement: false |
| `configs/data/train.yaml` | +enable_sepia_enhancement: false |
| `configs/data/val.yaml` | +enable_sepia_enhancement: false |

**Total**: 1 new file, 10 modified files

---

## References

- **Experiment**: [sepia experiment](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment-tracker/experiments/20251220_154834_zero_prediction_images_debug/README.md)
- **Reference Plan**: [gray-world implementation](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/artifacts/implementation_plans/2025-12-20_0226_implementation_plan_gray-world-normalization.md)
- **Sepia Script**: [sepia_enhancement.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment-tracker/experiments/20251220_154834_zero_prediction_images_debug/scripts/sepia_enhancement.py)
