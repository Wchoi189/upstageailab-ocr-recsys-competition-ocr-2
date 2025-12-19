# Border Removal Integration Context

## Preprocessing Stack Location

**Target location**: `ocr/datasets/transforms.py`

**Integration pattern**: Add as Albumentations custom transform (like `ConditionalNormalize`)

## Existing Transform Architecture

```python
# Current structure in ocr/datasets/transforms.py
class ConditionalNormalize(A.ImageOnlyTransform):
    """Example of custom Albumentations transform"""

class ValidatedDBTransforms:
    """Main transform pipeline orchestrator"""
    def __call__(self, data):
        # Applies transform pipeline
        # Returns TransformOutput (Pydantic model)
```

## Border Removal Transform Template

```python
class BorderRemoval(A.DualTransform):  # DualTransform handles both image + keypoints
    """
    Remove black borders from scanned documents.

    Args:
        method: Detection method ['canny', 'morph', 'hough']
        enabled: Enable conditional on skew threshold
        skew_threshold: Only apply if estimated skew > threshold
        min_area_ratio: Minimum crop area vs original (safety check)
        confidence_threshold: Minimum detection confidence to crop
    """

    def __init__(self,
                 method='canny',
                 enabled=True,
                 skew_threshold=20.0,
                 min_area_ratio=0.75,
                 confidence_threshold=0.8,
                 always_apply=False,
                 p=1.0):
        super().__init__(always_apply, p)
        self.method = method
        self.enabled = enabled
        self.skew_threshold = skew_threshold
        self.min_area_ratio = min_area_ratio
        self.confidence_threshold = confidence_threshold

    def apply(self, img, **params):
        """Apply to image (required by Albumentations)"""
        if not self.enabled:
            return img

        # Your border detection logic here
        cropped_img, metrics = self._detect_and_remove_border(img)

        # Safety: only crop if confident
        if metrics['confidence'] < self.confidence_threshold:
            return img
        if metrics['area_ratio'] < self.min_area_ratio:
            return img

        return cropped_img

    def apply_to_keypoint(self, keypoint, **params):
        """Transform polygon keypoints (required for DualTransform)"""
        # Adjust keypoint coordinates based on crop
        # Return (x, y, angle, scale) tuple
        pass

    def get_transform_init_args_names(self):
        """For serialization"""
        return ('method', 'enabled', 'skew_threshold',
                'min_area_ratio', 'confidence_threshold')
```

## Pipeline Integration Points

### Option A: Pre-augmentation (RECOMMENDED)
```python
# In ValidatedDBTransforms.__init__()
self.transform = A.Compose([
    BorderRemoval(method='canny', enabled=config.border_removal.enabled),  # FIRST
    A.Resize(height=config.img_h, width=config.img_w),
    # ... existing augmentations ...
    ConditionalNormalize(),
    ToTensorV2(),
])
```

### Option B: Conditional gating (for Option C)
```python
# Add to transform pipeline with conditional logic
if estimated_skew > 20:  # Option C threshold
    self.border_removal = BorderRemoval(method='canny')
else:
    self.border_removal = None
```

## Data Contract

**Input**: `TransformInput` (Pydantic model)
- `image: np.ndarray` (H, W, 3)
- `polygons: List[PolygonData]` (bounding boxes to adjust)

**Output**: `TransformOutput` (Pydantic model)
- `image: np.ndarray` (H', W', 3) - cropped
- `word_bboxes: np.ndarray` - adjusted coordinates
- `transformation_matrix: np.ndarray` - for inverse mapping

## Metrics Schema

```python
{
    "border_removed": bool,
    "confidence": float,  # 0.0-1.0
    "area_ratio": float,  # cropped_area / original_area
    "method": str,  # 'canny', 'morph', 'hough'
    "crop_box": [x1, y1, x2, y2],  # detected border bounds
    "processing_time_ms": float
}
```

## Testing Strategy

1. **Unit test**: `tests/ocr/datasets/test_border_removal.py`
   ```python
   def test_border_removal_preserves_keypoints():
       """Verify polygon coordinates adjusted correctly"""
   ```

2. **Integration test**: Add to `ocr/datasets/base.py` test dataset
   ```python
   # Test with actual dataset loading
   dataset = TextDataset(config, border_removal_enabled=True)
   ```

3. **Benchmark**: Use existing `experiment-tracker/` structure
   ```python
   # scripts/benchmark_border_removal.py
   # Measure latency on 1000 images
   ```

## Hardware Constraints

**Target hardware**: Check `configs/base.yaml` for device config
```yaml
devices: 1  # GPU count
precision: 32  # or 16 for mixed precision
```

**Latency budget**: <50ms per image (p95)
- Canny edge detection: ~10ms
- Contour detection: ~15ms
- Crop operation: ~5ms
- **Total**: ~30ms (leaves 20ms headroom)

## Failure Case Dataset Query

```python
# Use experiment_registry.py to find border cases
from ocr.experiment_registry import ExperimentRegistry

registry = ExperimentRegistry()

# Query strategy (pseudo-code - adapt to actual registry API)
border_candidates = [
    exp for exp in registry.list_experiments()
    if exp.metadata.get('skew_deg_abs', 0) > 20
    and exp.metadata.get('predicted_text', '').strip() == ''
]

# Primary test case
test_case = "data/zero_prediction_worst_performers/drp.en_ko.in_house.selectstar_000732.jpg"
```

## VLM Tool Usage (Offline Only)

**Constraint**: VLM latency (3-5s) incompatible with <50ms requirement

**Use cases**:
1. **Dataset curation** (offline): Identify border severity
2. **Failure analysis** (offline): Debug detection failures
3. **Validation** (offline): Verify crop quality

**DO NOT** use VLM in runtime pipeline - only OpenCV methods.

## Config Schema

Add to `configs/data/` directory:
```yaml
# configs/data/border_removal.yaml
border_removal:
  enabled: true
  method: canny  # or morph, hough
  skew_threshold: 20.0  # Option C gating
  min_area_ratio: 0.75
  confidence_threshold: 0.8

  canny:
    low_threshold: 50
    high_threshold: 150

  morph:
    kernel_size: 5
    iterations: 2
```

## Integration Test Protocol (Option C)

```python
def test_option_c_gating():
    """Test conditional border removal based on skew"""

    # Case 1: High skew (>20°) - should apply border removal
    img_high_skew = load_image("000732.jpg")  # -83° skew
    result = transform(img_high_skew, enable_border_removal=True)
    assert result.metadata['border_removed'] == True

    # Case 2: Low skew (<20°) - should skip border removal
    img_low_skew = load_image("normal.jpg")  # 2° skew
    result = transform(img_low_skew, enable_border_removal=True)
    assert result.metadata['border_removed'] == False

    # Case 3: Fallback safety - low confidence
    img_ambiguous = load_image("complex_border.jpg")
    result = transform(img_ambiguous, confidence_threshold=0.9)
    if result.metadata['confidence'] < 0.9:
        assert result.metadata['border_removed'] == False
```
