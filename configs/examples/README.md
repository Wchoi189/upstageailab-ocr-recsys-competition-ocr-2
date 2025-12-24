# Configuration Examples

This directory contains example configurations for common use cases.

## Preprocessing & Enhancement

### 1. Basic Prediction (Default)
```bash
uv run python runners/predict.py \
  --config-name predict \
  checkpoint_path=path/to/checkpoint.ckpt \
  image_dir=data/test_images
```

### 2. With Perspective Correction
```bash
uv run python runners/predict.py \
  --config-name predict_with_perspective \
  checkpoint_path=path/to/checkpoint.ckpt \
  image_dir=data/test_images
```

### 3. Full Enhancement Pipeline
```bash
uv run python runners/predict.py \
  --config-name predict_full_enhancement \
  checkpoint_path=path/to/checkpoint.ckpt \
  image_dir=data/test_images
```

### 4. Command-Line Override
```bash
uv run python runners/predict.py \
  --config-name predict \
  checkpoint_path=path/to/checkpoint.ckpt \
  image_dir=data/test_images \
  preprocessing.enable_perspective_correction=true \
  preprocessing.enable_background_normalization=true
```

## Available Preprocessing Options

Based on experiment `20251217_024343_image_enhancements_implementation`:

| Option | Status | Description | Performance Impact |
|--------|--------|-------------|-------------------|
| `enable_perspective_correction` | ✅ Production-ready | Rembg mask + 4-point transform | One extreme case: -83° → 0.88° |
| `enable_background_normalization` | ✅ Recommended | Gray-world method | 75% tint reduction, 100% success rate |
| `enable_sepia_enhancement` | ⚠️ Testing | Alternative to normalization | User reports promising results |
| `enable_grayscale` | ✅ Available | Grayscale conversion | Simple preprocessing |
| `enable_clahe` | ✅ Available | Contrast enhancement | Adaptive histogram equalization |
| ~~`enable_deskewing`~~ | ❌ Excluded | Hough lines text rotation | **No OCR improvement** (excluded from integration) |

### Display Modes

- **`perspective_display_mode`**:
  - `"corrected"` (default): Display inference results on corrected image
  - `"original"`: Transform polygons back to original image space

- **`sepia_display_mode`**:
  - `"enhanced"` (default): Show sepia-enhanced image
  - `"original"`: Show original image

## Configuration Hierarchy

```
configs/
├── predict.yaml              # Base prediction config
├── _base/
│   └── preprocessing.yaml    # Default preprocessing settings (all disabled)
└── examples/
    ├── predict_with_perspective.yaml      # Enable perspective only
    └── predict_full_enhancement.yaml      # Full enhancement pipeline
```

## Experiment Reference

All preprocessing options are based on validated experiments:
- **Experiment**: `experiment-tracker/experiments/20251217_024343_image_enhancements_implementation`
- **Documentation**: See experiment README and state.yml for detailed results
- **Key Finding**: Background normalization + perspective correction = best results
- **Exclusion**: Deskewing excluded from integration (no performance benefit)
