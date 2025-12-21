# Preprocessing Enhancement Architecture

## Overview

The OCR preprocessing pipeline supports optional image enhancement techniques to improve text detection accuracy on challenging document images (aged, low-contrast, tinted backgrounds).

**Module:** `ocr.utils.sepia_enhancement`
**Integration Point:** `ocr.inference.preprocess.preprocess_image()`
**Feature Flags:** `enable_sepia_enhancement`, `enable_clahe`

---

## Enhancement Methods

### 1. Sepia Tone Transformation

**Function:** `enhance_sepia(img: np.ndarray) -> np.ndarray`

**Purpose:** Apply classic sepia color transformation to normalize document tint and improve text-background separation.

**Algorithm:**
```python
# Classic sepia matrix (Albumentations standard)
R' = 0.393*R + 0.769*G + 0.189*B
G' = 0.349*R + 0.686*G + 0.168*B
B' = 0.272*R + 0.534*G + 0.131*B
```

**Color Space:** RGB → Sepia RGB → BGR (OpenCV convention)

**Performance:**
- Processing time: ~5-10ms per image
- Best for: Aged documents, yellowish/bluish tints, faded text
- Validated improvement: +164% edge detection on zero-prediction images

**When to Use:**
- Enable by default for production inference
- Particularly effective on historical documents, scanned receipts, faded forms

---

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Function:** `enhance_clahe(img: np.ndarray, clip_limit=2.0, tile_size=(8,8)) -> np.ndarray`

**Purpose:** Adaptive local contrast enhancement while preventing noise over-amplification.

**Algorithm:**
1. Convert BGR → LAB color space
2. Apply CLAHE to L (luminance) channel only
   - `clip_limit=2.0`: Moderate contrast limiting threshold
   - `tile_grid_size=(8,8)`: Local 8×8 pixel grid for histogram equalization
3. Preserve A/B (chrominance) channels unchanged
4. Convert LAB → BGR

**Performance:**
- Processing time: ~15-20ms per image
- **Caution:** Can amplify paper grain/noise on clean documents

**When to Use:**
- **Disable by default** (empirically poor performance on most documents)
- Consider for extreme low-contrast cases only
- May benefit very dark scans or heavily shadowed regions

**Why it underperforms:**
- Documents typically have binary text (high inherent contrast)
- Tile-based processing can create artificial boundaries in text regions
- Amplifies paper texture more than improving text clarity

---

## Pipeline Integration

### Processing Order

```
Input Image (BGR)
    ↓
[Optional] Gray-world background normalization
    ↓
[Optional] Sepia enhancement ← enable_sepia_enhancement
    ↓
[Optional] CLAHE enhancement ← enable_clahe
    ↓
Resize + Pad to target size
    ↓
ToTensor + Normalize
    ↓
Model Input
```

**Rationale:**
1. **Background normalization first**: Removes color casts before sepia
2. **Sepia before CLAHE**: Sepia standardizes color, then CLAHE adjusts contrast
3. **Both before resize**: Enhancement works better on full-resolution images

### Configuration Schema

**File:** `config.yaml` (experiment/training configs)
```yaml
preprocessing:
  enable_sepia_enhancement: true   # Recommended: true
  enable_clahe: false               # Recommended: false
```

**Runtime Override:** API requests can override config defaults
```python
# Backend API
POST /api/inference/preview
{
  "enable_sepia_enhancement": true,
  "enable_clahe": false
}
```

---

## Code Locations

### Core Implementation
| File | Purpose |
|------|---------|
| `ocr/utils/sepia_enhancement.py` | Enhancement functions |
| `ocr/inference/preprocess.py` | Integration into preprocessing |
| `ocr/inference/config_loader.py` | Config schema definitions |

### Pipeline Propagation
| File | Purpose |
|------|---------|
| `ocr/inference/preprocessing_pipeline.py` | Pipeline orchestration |
| `ocr/inference/orchestrator.py` | Inference orchestration |
| `ocr/inference/engine.py` | Public inference API |

### Backend API
| File | Purpose |
|------|---------|
| `apps/shared/backend_shared/models/inference.py` | Request model schema |
| `apps/ocr-inference-console/backend/services/inference_service.py` | Inference service |
| `apps/ocr-inference-console/backend/main.py` | REST API endpoint |

### Frontend UI
| File | Purpose |
|------|---------|
| `apps/ocr-inference-console/src/contexts/InferenceContext.tsx` | State management |
| `apps/ocr-inference-console/src/components/Sidebar.tsx` | Toggle checkboxes |
| `apps/ocr-inference-console/src/components/Workspace.tsx` | Request orchestration |
| `apps/ocr-inference-console/src/api/ocrClient.ts` | API client |

**Total files modified:** 16

---

## Experimental Validation

**Experiment ID:** `20251220_154834_zero_prediction_images_debug`

**Problem Statement:** OCR model produces zero predictions on aged, low-contrast document images (samples: 000712, 000732)

**Methods Tested:**
1. `sepia_classic` - Standard sepia transformation ✅ **Winner**
2. `sepia_adaptive` - Intensity-weighted sepia blending
3. `sepia_warm` - Enhanced warm tones (original implementation, too dark)
4. `sepia_clahe` - Sepia + CLAHE combined ❌ **Underperforms classic alone**
5. `sepia_linear_contrast` - Sepia + global contrast

**Results:**
| Method | Edge Improvement | Contrast Boost | Verdict |
|--------|-----------------|----------------|---------|
| `sepia_classic` | **+164%** | +8.2 | ✅ Production |
| `sepia_clahe` | +164% | +8.2 | ❌ No benefit over classic |

**Key Finding:** Classic sepia alone provides optimal results. CLAHE adds no measurable benefit and risks noise amplification.

---

## Deployment Configuration

### Recommended Settings

**Production Inference:**
```yaml
preprocessing:
  enable_sepia_enhancement: true
  enable_clahe: false
```

**Training Pipeline:**
- Not yet integrated (blocked: gitignore on `ocr/datasets/`)
- Future work: Add as data augmentation option

### UI Defaults

**Frontend Initial State:**
```typescript
enableSepiaEnhancement: false  // Off by default, user-controlled
enableClahe: false             // Off by default
```

**Reasoning:** Conservative defaults allow users to A/B test on their specific documents.

---

## Performance Characteristics

### Computational Cost
```
Baseline (no enhancement):        ~30ms
+ Sepia:                          ~35-40ms  (+16% overhead)
+ CLAHE:                          ~45-50ms  (+50% overhead)
+ Sepia + CLAHE:                  ~50-60ms  (+66% overhead)
```

### Memory Footprint
- In-place operations where possible
- Temporary LAB conversion for CLAHE: +1x image size (HWC)
- No persistent state

### Thread Safety
- All functions are stateless and thread-safe
- Safe for concurrent inference workers

---

## Future Enhancements

### Potential Improvements
1. **Adaptive CLAHE parameters**: Detect image statistics and adjust `clip_limit`/`tile_size`
2. **Selective enhancement**: Apply different methods per region based on contrast analysis
3. **Training integration**: Add sepia as augmentation to training dataset pipeline
4. **Benchmark suite**: Automated A/B testing on diverse document types

### Known Limitations
1. **Not optimized for color documents**: Sepia loses color information (acceptable for most OCR)
2. **Fixed parameters**: No auto-tuning based on image characteristics
3. **CPU-only**: No GPU acceleration (acceptable given low overhead)

---

## References

- Albumentations sepia implementation: Standard computer vision transformation
- CLAHE: Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
- Experiment manifest: `experiment-tracker/experiments/20251220_154834_zero_prediction_images_debug/README.md`
