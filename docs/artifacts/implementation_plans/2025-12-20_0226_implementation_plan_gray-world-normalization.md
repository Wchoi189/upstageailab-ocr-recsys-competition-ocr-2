---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['background-normalization', 'gray-world', 'preprocessing', 'pipeline-integration']
title: "Gray-World Background Normalization Integration"
date: "2025-12-20 02:26 (KST)"
branch: "main"
validated_performance:
  tint_reduction: "75% (58.1 → 14.6)"
  processing_time: "~20ms"
  zero_prediction_improvement: "user_confirmed"
reference_guide: "experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/docs/20251220_0200_guide_gray-world-integration.md"
---

# Gray-World Background Normalization Integration

## Problem

Production OCR pipeline loses predictions on tinted-background documents. Gray-world normalization validated at 75% tint reduction, requires integration across 3 pipelines.

## Scope

**Pipelines**: Inference, Training, UI
**Files Modified**: 7+ (config schemas, preprocessing, datasets, UI engines)
**New Files**: 1 (`ocr/utils/background_normalization.py`)
**Estimated Effort**: 3-5 hours
**Risk**: Low (feature-flagged, validated)

---

## Implementation Phases

### Phase 1: Core Utility

**File**: `ocr/utils/background_normalization.py` (NEW)

```python
import numpy as np

def normalize_gray_world(img: np.ndarray) -> np.ndarray:
    """Gray-world normalization. Scales channels to neutral gray average."""
    b_avg, g_avg, r_avg = img.mean(axis=(0, 1))
    gray_avg = (b_avg + g_avg + r_avg) / 3

    if b_avg == 0 or g_avg == 0 or r_avg == 0:
        return img.copy()

    scale_factors = np.array([gray_avg / b_avg, gray_avg / g_avg, gray_avg / r_avg])
    result = img.astype(np.float32) * scale_factors
    return np.clip(result, 0, 255).astype(np.uint8)
```

---

### Phase 2: Inference Integration

#### 2.1 Config Schema

**File**: `ocr/inference/config_loader.py:33-35`

```python
@dataclass(slots=True)
class PreprocessSettings:
    image_size: tuple[int, int]
    normalization: NormalizationSettings
    enable_background_normalization: bool = False  # ADD
```

#### 2.2 Preprocessing Pipeline

**File**: `ocr/inference/preprocess.py`

**Import**:
```python
from ocr.utils.background_normalization import normalize_gray_world
```

**Function signature** (~line 60):
```python
def preprocess_image(
    image: Any,
    transform: Callable[[Any], Any],
    target_size: int = 640,
    return_processed_image: bool = False,
    enable_background_normalization: bool = False,  # ADD
) -> Any | tuple[Any, Any]:
```

**Apply normalization** (BEFORE LongestMaxSize):
```python
processed_image = image.copy()

if enable_background_normalization:
    processed_image = normalize_gray_world(processed_image)

# ... existing resize/transform code ...
```

#### 2.3 YAML Config

**File**: `configs/_base/preprocessing.yaml`

```yaml
preprocessing:
  image_size: [640, 640]
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  enable_background_normalization: true  # ADD
```

---

### Phase 3: Training Integration

#### 3.1 Dataset Schema

**File**: `ocr/datasets/schemas.py` (DatasetConfig)

```python
class DatasetConfig(BaseModel):
    # ... existing fields ...
    enable_background_normalization: bool = False  # ADD (default FALSE for training)
```

#### 3.2 Dataset __getitem__

**File**: `ocr/datasets/base.py:~410`

```python
def __getitem__(self, idx: int) -> dict[str, Any]:
    # ... load image ...
    image_array = image_data.image_array

    # Apply normalization BEFORE transform
    if self.config.enable_background_normalization:
        from ocr.utils.background_normalization import normalize_gray_world
        image_array = normalize_gray_world(image_array)

    # ... transform, return ...
```

#### 3.3 Training Config

**File**: `configs/data/train.yaml`

```yaml
dataset:
  # ... existing fields ...
  enable_background_normalization: false  # DEFAULT FALSE (avoid augmentation conflicts)
```

---

### Phase 4: UI Integration

#### 4.1 Shared InferenceEngine

**File**: `apps/shared/backend_shared/inference.py`

```python
class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        enable_perspective_correction: bool = True,
        enable_background_normalization: bool = True,  # ADD
    ):
        self.enable_background_normalization = enable_background_normalization
        # ... init ...

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        processed = image.copy()

        if self.enable_background_normalization:
            processed = normalize_gray_world(processed)

        if self.enable_perspective_correction:
            # ... perspective code ...

        return processed
```

#### 4.2 Playground Console

**File**: `apps/playground-console/backend/routers/inference.py`

```python
async def run_inference(request: InferenceRequest):
    enable_bg_norm = getattr(request, 'enable_background_normalization', True)
    _inference_engine.enable_background_normalization = enable_bg_norm
    # ... run inference ...
```

---

## Configuration Matrix

| Pipeline | Config Location | Parameter | Default | Rationale |
|----------|----------------|-----------|---------|-----------|
| Inference | `configs/_base/preprocessing.yaml` | `enable_background_normalization` | `true` | Production benefit |
| Training | `configs/data/train.yaml` | `dataset.enable_background_normalization` | `false` | Avoid augmentation conflicts |
| Validation | `configs/data/val.yaml` | `dataset.enable_background_normalization` | `false` | Match training |
| UI Apps | Runtime parameter | `InferenceEngine(..., enable_background_normalization=True)` | `true` | User toggle available |

**Critical**: Training/validation MUST default to `false` to prevent data distribution shift and augmentation conflicts.

---

## Validation Plan

### Test Suite

```bash
# Inference with/without normalization
python runners/predict.py \
    checkpoint=epoch-18_step-001957.ckpt \
    data.test_path=data/zero_prediction_worst_performers \
    preprocessing.enable_background_normalization={true|false}

# Training dataset loading
python -c "
from ocr.datasets.base import ValidatedOCRDataset
from ocr.datasets.schemas import DatasetConfig
from pathlib import Path

config = DatasetConfig(
    image_path=Path('data/train/images'),
    annotation_path=Path('data/train/annotations.json'),
    enable_background_normalization=True
)
dataset = ValidatedOCRDataset(config, transform=None)
sample = dataset[0]
print(f'Loaded: {sample[\"image\"].shape}')
"

# UI toggle
cd apps/playground-console && npm run dev
# Test Enable Background Normalization checkbox
```

### Success Criteria

- Inference flag toggles normalization
- Training defaults to `false`, no augmentation conflicts
- UI toggle functions correctly
- Processing time ≤25ms per image
- Zero prediction rate reduced (baseline comparison)

---

## Integration Order

**Critical**: Normalization must occur at specific pipeline positions:

```
Inference:
  normalize_gray_world() → LongestMaxSize → PadIfNeeded → ModelTransform

Training:
  LoadImage → normalize_gray_world() → Augmentation → Transform

Perspective:
  RemoveBackground → CorrectPerspective → normalize_gray_world()
```

---

## Risk Mitigation

### Risk 1: Training Distribution Shift
**Severity**: Medium
**Mitigation**: Default `false` for training/validation
**Status**: Addressed in Phase 3.3

### Risk 2: Augmentation Conflicts
**Severity**: Medium
**Mitigation**: Apply normalization BEFORE augmentation transforms
**Status**: Addressed in Phase 3.2

### Risk 3: Performance Regression
**Severity**: Low
**Mitigation**: Validated at ~20ms, monitoring added
**Status**: Acceptable

---

## Rollback Plan

**All components feature-flagged**:

1. **Inference**: Set `enable_background_normalization: false` in YAML
2. **Training**: Already defaults to `false`
3. **UI**: Toggle off in interface or set engine parameter to `false`

**No code removal required** - disable via configuration.

---

## File Change Summary

### New Files
- `ocr/utils/background_normalization.py` (~25 lines)

### Modified Files
1. `ocr/inference/config_loader.py` (+1 field to PreprocessSettings)
2. `ocr/inference/preprocess.py` (+param, +import, +normalization call)
3. `ocr/datasets/schemas.py` (+1 field to DatasetConfig)
4. `ocr/datasets/base.py` (+normalization in `__getitem__`)
5. `apps/shared/backend_shared/inference.py` (+param, +preprocessing step)
6. `apps/playground-console/backend/routers/inference.py` (+flag passing)
7. `configs/_base/preprocessing.yaml` (+enable_background_normalization: true)
8. `configs/data/train.yaml` (+enable_background_normalization: false)
9. `configs/data/val.yaml` (+enable_background_normalization: false)

**Total**: 1 new file, 9 modified files

---

## References

- **Integration Guide**: [`gray-world-integration.md`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/docs/20251220_0200_guide_gray-world-integration.md)
- **Config Loader**: [`config_loader.py`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/inference/config_loader.py)
- **Dataset Base**: [`base.py`](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/datasets/base.py)
- **Performance Data**: Experiment 20251217_024343

---

## Next Steps

1. Create `ocr/utils/background_normalization.py`
2. Update config schemas (inference + datasets)
3. Integrate into preprocessing pipelines
4. Update YAML configs with proper defaults
5. Run validation test suite
6. Deploy with monitoring
