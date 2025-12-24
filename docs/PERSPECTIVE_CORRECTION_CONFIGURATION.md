# Perspective Correction Configuration - Quick Reference

## What Was Added

### 1. Base Configuration (`configs/_base/preprocessing.yaml`)
Added comprehensive preprocessing options:
- ✅ `enable_perspective_correction`: Rembg-based mask detection + 4-point transform
- ✅ `perspective_display_mode`: "corrected" or "original"
- ✅ `enable_grayscale`: Grayscale conversion
- ✅ `enable_background_normalization`: Gray-world method (75% tint reduction)
- ✅ `enable_sepia_enhancement`: Alternative enhancement method
- ✅ `enable_clahe`: Contrast enhancement

**Default**: All options disabled (false) for backward compatibility

### 2. Prediction Config (`configs/predict.yaml`)
- Added preprocessing config to defaults
- Added commented override section as usage reference

### 3. Example Configs (`configs/examples/`)
Created ready-to-use example configurations:
- `predict_with_perspective.yaml`: Enable perspective correction only
- `predict_full_enhancement.yaml`: Full pipeline (perspective + background normalization)
- `README.md`: Complete usage guide and command examples

### 4. Test Script (`scripts/test_perspective_inference.py`)
Created standalone test script to validate the integration:
- Tests without perspective correction
- Tests with perspective correction
- Tests with full enhancement pipeline
- Saves comparison visualizations

## Usage Examples

### Method 1: Using Example Configs

```bash
# With perspective correction only
uv run python runners/predict.py \
  --config-name predict_with_perspective \
  checkpoint_path=outputs/.../checkpoint.ckpt \
  image_dir=data/test_images

# With full enhancement pipeline
uv run python runners/predict.py \
  --config-name predict_full_enhancement \
  checkpoint_path=outputs/.../checkpoint.ckpt \
  image_dir=data/test_images
```

### Method 2: Command-Line Override

```bash
uv run python runners/predict.py \
  --config-name predict \
  checkpoint_path=outputs/.../checkpoint.ckpt \
  image_dir=data/test_images \
  preprocessing.enable_perspective_correction=true \
  preprocessing.enable_background_normalization=true
```

### Method 3: Direct API (Python)

```python
from ocr.inference.engine import InferenceEngine

engine = InferenceEngine()
engine.load_model(checkpoint_path="...", config_path="...")

# Enable perspective correction
predictions = engine.predict_image(
    image_path="test.jpg",
    enable_perspective_correction=True,
    enable_background_normalization=True,
    perspective_display_mode="corrected"
)
```

### Method 4: Test Script

```bash
# Test on single image with visualizations
python scripts/test_perspective_inference.py \
  --image data/test_images/sample.jpg \
  --checkpoint outputs/.../checkpoint.ckpt \
  --output outputs/perspective_test
```

## Configuration Options

### Perspective Correction
```yaml
preprocessing:
  enable_perspective_correction: true
  perspective_display_mode: "corrected"  # or "original"
```

- **`enable_perspective_correction`**: Uses rembg for background removal, mask-based rectangle fitting, and 4-point transform
- **`perspective_display_mode`**:
  - `"corrected"`: Display results on corrected image (default)
  - `"original"`: Transform polygons back to original image space

### Full Enhancement Pipeline
```yaml
preprocessing:
  enable_perspective_correction: true
  perspective_display_mode: "corrected"
  enable_background_normalization: true  # Gray-world method
  enable_sepia_enhancement: false
  enable_grayscale: false
  enable_clahe: false
```

## Experiment Validation

Based on: `experiment-tracker/experiments/20251217_024343_image_enhancements_implementation`

### Results Summary
- **Perspective Correction**: ✅ One extreme case: -83° → 0.88° skew correction
- **Background Normalization**: ✅ 75% tint reduction, 100% success rate (6/6 images)
- **Sepia Enhancement**: ⚠️ Under testing as alternative
- **Deskewing**: ❌ Excluded from integration (no OCR improvement)

### Recommended Configuration
```yaml
preprocessing:
  enable_perspective_correction: true
  enable_background_normalization: true
  enable_sepia_enhancement: false
```

## Files Modified

1. `configs/_base/preprocessing.yaml` - Added all preprocessing options
2. `configs/predict.yaml` - Included preprocessing config + override examples
3. `configs/examples/predict_with_perspective.yaml` - New example config
4. `configs/examples/predict_full_enhancement.yaml` - New example config
5. `configs/examples/README.md` - Complete usage guide
6. `scripts/test_perspective_inference.py` - Validation test script

## Next Steps

1. **Test the integration**:
   ```bash
   python scripts/test_perspective_inference.py \
     --image data/test_images/sample.jpg \
     --output outputs/test
   ```

2. **Run on worst performers**:
   ```bash
   uv run python runners/predict.py \
     --config-name predict_full_enhancement \
     checkpoint_path=outputs/.../checkpoint.ckpt \
     image_dir=data/zero_prediction_worst_performers
   ```

3. **Enable in UI**: The UI configs already have perspective correction fields ([`configs/ui/inference.yaml`](configs/ui/inference.yaml#L79))

4. **Benchmark performance**: Compare OCR accuracy with/without perspective correction

## Related Files

- **Core Implementation**: [`ocr/utils/perspective_correction/`](ocr/utils/perspective_correction/)
- **Preprocessing Pipeline**: [`ocr/inference/preprocessing_pipeline.py`](ocr/inference/preprocessing_pipeline.py)
- **Inference Engine**: [`ocr/inference/engine.py`](ocr/inference/engine.py)
- **Experiment Scripts**: [`experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/scripts/`](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/scripts/)
