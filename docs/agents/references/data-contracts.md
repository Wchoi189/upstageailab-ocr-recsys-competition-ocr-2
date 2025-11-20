# Data Contracts Quick Reference

## Overview

**Full Documentation:** [`docs/pipeline/data_contracts.md`](../../pipeline/data_contracts.md)

Data contracts define the expected shape, types, and validation rules for data flowing through the OCR pipeline. **Always review data contracts before modifying pipeline code to prevent shape errors.**

## When to Review

**REQUIRED before:**
- Modifying dataset loaders
- Changing preprocessing functions
- Updating model input/output shapes
- Modifying postprocessing logic
- Adding new data transforms

## Critical Contracts

### Image Input Contract
- **Shape**: `(H, W, C)` where C=3 (RGB) or C=1 (grayscale)
- **Type**: `numpy.ndarray` or `torch.Tensor`
- **Range**: `[0, 255]` (uint8) or `[0.0, 1.0]` (float32)
- **Format**: RGB color order (not BGR)

### Annotation Contract
- **Bounding Boxes**: `[[x1, y1, x2, y2, x3, y3, x4, y4], ...]`
- **Format**: Polygon with 4 points (quadrilateral)
- **Coordinates**: Absolute pixel coordinates
- **Order**: Clockwise from top-left

### Model Output Contract
- **Detection**: `{boxes: [...], scores: [...], labels: [...]}`
- **Recognition**: `{text: str, confidence: float}`
- **Batch**: First dimension is batch size

### Dataset Contract
- **Train**: `{image: Tensor, target: dict}`
- **Val**: Same as train
- **Test**: `{image: Tensor, image_id: str}`

## Common Pitfalls

❌ **Wrong color order** (BGR vs RGB)
❌ **Wrong coordinate format** (x1,y1,x2,y2 vs polygon)
❌ **Wrong normalization** (0-255 vs 0-1)
❌ **Wrong shape** (HWC vs CHW)
❌ **Missing validation** (no shape checks)

## Quick Validation Pattern

```python
def validate_input(image: np.ndarray) -> None:
    """Validate image input contract."""
    assert isinstance(image, np.ndarray), "Image must be numpy array"
    assert image.ndim == 3, f"Image must be 3D, got {image.ndim}D"
    assert image.shape[2] in [1, 3], f"Invalid channels: {image.shape[2]}"
    assert image.dtype in [np.uint8, np.float32], f"Invalid dtype: {image.dtype}"

    if image.dtype == np.float32:
        assert image.min() >= 0.0 and image.max() <= 1.0, "Float images must be [0, 1]"
```

## Prevention Checklist

Before modifying pipeline code:
- [ ] Read relevant data contract section
- [ ] Understand expected input/output shapes
- [ ] Add validation checks for shape/type
- [ ] Test with sample data
- [ ] Verify contract compliance

## Links

- **Full Data Contracts**: [`docs/pipeline/data_contracts.md`](../../pipeline/data_contracts.md)
- **Pipeline Documentation**: [`docs/pipeline/`](../../pipeline/)
- **Dataset Loaders**: `ocr/data/dataset.py`
- **Preprocessing**: `ocr/preprocessing/`

## Examples

### Good: Contract-Compliant Code
```python
def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess image following input contract."""
    # Validate contract
    assert image.shape[-1] == 3, "Expect RGB image"
    assert image.dtype == np.uint8, "Expect uint8 image"

    # Convert to float [0, 1]
    image = image.astype(np.float32) / 255.0

    # Convert HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    return torch.from_numpy(image)
```

### Bad: Contract Violation
```python
def preprocess_image(image):
    # ❌ No validation
    # ❌ Assumes BGR (wrong!)
    # ❌ No shape check
    return torch.tensor(image)
```

## Additional Resources

- [Pipeline Architecture](../../pipeline/architecture.md)
- [Data Loading Guide](../../pipeline/data-loading.md)
- [Preprocessing Guide](../../pipeline/preprocessing.md)
