# 🚀 Quick Start: Perspective Correction

## 1-Minute Setup

### Enable in Your Config
Edit `configs/predict.yaml` and uncomment:
```yaml
preprocessing:
  enable_perspective_correction: true
  enable_background_normalization: true
```

### Or Use Command Line
```bash
uv run python runners/predict.py \
  --config-name predict \
  checkpoint_path=YOUR_CHECKPOINT.ckpt \
  image_dir=YOUR_IMAGES \
  preprocessing.enable_perspective_correction=true
```

### Or Use Python API
```python
from ocr.inference.engine import InferenceEngine

engine = InferenceEngine()
engine.load_model("checkpoint.ckpt")
result = engine.predict_image(
    "image.jpg",
    enable_perspective_correction=True
)
```

## Test It Now

```bash
# Test on single image
python scripts/test_perspective_inference.py \
  --image data/test_images/sample.jpg \
  --checkpoint YOUR_CHECKPOINT.ckpt \
  --output outputs/test
```

## Full Documentation

- 📖 Complete Guide: `docs/PERSPECTIVE_CORRECTION_CONFIGURATION.md`
- 📋 Summary: `docs/PERSPECTIVE_CORRECTION_SUMMARY.md`
- 📁 Examples: `configs/examples/`

## Available Options

| Option | Recommended |
|--------|-------------|
| `enable_perspective_correction` | ✅ Yes |
| `enable_background_normalization` | ✅ Yes |
| `enable_sepia_enhancement` | ⚠️ Testing |
| `enable_grayscale` | ❌ Optional |
| `enable_clahe` | ❌ Optional |

**Based on**: Experiment `20251217_024343_image_enhancements_implementation`
**Result**: -83° → 0.88° skew correction, 75% tint reduction
