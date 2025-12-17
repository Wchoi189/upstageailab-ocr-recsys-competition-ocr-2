---
type: data_contract
component: training_pipeline
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Training Data Contracts

**Purpose**: Pydantic V2-validated data contracts for OCR pipeline; defines shapes, types, formats for dataset, transforms, collate, model, loss.

---

## Pipeline Contracts

| Contract | Producer | Consumer | Validation Model |
|----------|----------|----------|------------------|
| **Dataset Sample** | `ValidatedOCRDataset.__getitem__()` | Transform Pipeline | `DatasetSample` (Pydantic V2) |
| **Transform Output** | Transform Pipeline | Collate Function | `TransformOutput` (Pydantic V2) |
| **Batch Output** | Collate Function | Model | `CollateOutput` (Pydantic V2) |
| **Model Output** | Model | Loss Function | Tensor dict (validated) |
| **Loss Output** | Loss Function | Trainer | Scalar tensor |

---

## Dataset Sample Contract

**Model**: `DatasetSample` (Pydantic V2)

| Field | Type | Shape | Range | Purpose |
|-------|------|-------|-------|---------|
| `image` | np.ndarray | (H, W, 3) | uint8 or float32 | Input image |
| `polygons` | List[np.ndarray] | Each (N, 2) | float32 | Text region polygons |
| `metadata` | Optional[dict] | N/A | - | Optional metadata |
| `prob_maps` | np.ndarray | (H, W) | [0, 1], float32 | Probability maps |
| `thresh_maps` | np.ndarray | (H, W) | [0, 1], float32 | Threshold maps |
| `image_filename` | str | N/A | - | Relative path |
| `image_path` | str | N/A | - | Absolute path |
| `inverse_matrix` | np.ndarray | (3, 3) | float32 | Affine transform (640x640 → original) |
| `shape` | Tuple[int, int] | (H, W) | - | Original shape |

**Critical**: `inverse_matrix` must correctly map coordinates from transformed space (640x640) to original image space.

---

## Transform Output Contract

**Model**: `TransformOutput` (Pydantic V2)

| Field | Type | Shape | Range | Purpose |
|-------|------|-------|-------|---------|
| `image` | torch.Tensor | (3, 640, 640) | [-1, 1], float32 | Normalized image |
| `polygons` | List[np.ndarray] | Each (N, 2) | float32 | Transformed polygons |
| `prob_maps` | torch.Tensor | (1, 640, 640) | [0, 1], float32 | Probability maps |
| `thresh_maps` | torch.Tensor | (1, 640, 640) | [0, 1], float32 | Threshold maps |

---

## Batch Output Contract (Collate)

**Model**: `CollateOutput` (Pydantic V2)

| Field | Type | Shape | Range | Purpose |
|-------|------|-------|-------|---------|
| `images` | torch.Tensor | (B, 3, 640, 640) | [-1, 1], float32 | Batch images |
| `prob_maps` | torch.Tensor | (B, 1, 640, 640) | [0, 1], float32 | Batch probability maps |
| `thresh_maps` | torch.Tensor | (B, 1, 640, 640) | [0, 1], float32 | Batch threshold maps |
| `polygons` | List[List[np.ndarray]] | B x [Each (N, 2)] | float32 | Batch polygons |

---

## Model Input/Output Contract

**Input**: `CollateOutput` (from collate function)

**Output**:
```python
{
    'prob_map': torch.Tensor,    # Shape: (B, 1, 640, 640), range: [0, 1]
    'thresh_map': torch.Tensor,  # Shape: (B, 1, 640, 640), range: [0, 1]
}
```

---

## Loss Function Contract

**Input**: Model output dict + ground truth

**Output**: Scalar tensor (loss value)

---

## Preprocessing Pipeline Contract

**Model**: `PreprocessingResultContract` (Pydantic V2)

| Field | Type | Purpose |
|-------|------|---------|
| `image` | np.ndarray | Preprocessed image |
| `metadata` | dict | Preprocessing metadata |

**Input Contract**: `ImageInputContract` (numpy arrays, dimensions, channels)

**Detection Result**: `DetectionResultContract` (corners, confidence, method)

---

## Polygon Validation

**Model**: `ValidatedPolygonData` (Pydantic V2)

**Rules**:
- All x-coordinates within image width
- All y-coordinates within image height
- Minimum 3 points per polygon
- No NaN/Inf values

---

## Tensor Validation

**Model**: `ValidatedTensorData` (Pydantic V2)

**Rules**:
- Shape matches expected dimensions
- Device (CPU/CUDA) specified
- Dtype (float32, uint8, etc.)
- Value range [min, max]
- No NaN/Inf values

---

## Validation Strategy

| Strategy | Implementation |
|----------|----------------|
| **Strict Mode** | No automatic coercion |
| **Error Messages** | Clear, actionable |
| **Runtime Checks** | At pipeline boundaries (dataset, collate, model) |
| **Type Safety** | Full type hints and validation |

---

## Common Pydantic Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ValidationError: 1 validation error for DatasetSample` | Field types/shapes mismatch | Check field types |
| `Field required` | Missing required fields | Add missing fields |
| `Input should be a valid tensor` | Wrong tensor types/shapes | Verify tensor creation |
| `Polygon has out-of-bounds x-coordinates` | Coordinates exceed image dimensions | Clip coordinates |
| `Tensor shape mismatch` | Tensor doesn't match expected shape | Check transform pipeline |
| `Tensor contains NaN values` | Invalid numerical values | Debug computation |

---

## Dependencies

| Component | Validation Model | Upstream | Downstream |
|-----------|------------------|----------|------------|
| **Dataset** | `DatasetSample` | Raw data | Transform Pipeline |
| **Transform** | `TransformOutput` | Dataset | Collate Function |
| **Collate** | `CollateOutput` | Transform | Model |
| **Model** | Tensor dict | Collate | Loss Function |
| **Loss** | Scalar tensor | Model | Trainer |

---

## Constraints

- **Strict Validation**: No automatic type coercion
- **Shape Invariance**: All tensors must match expected shapes
- **Value Ranges**: prob_maps, thresh_maps in [0, 1]; images normalized
- **Polygon Bounds**: All coordinates within image dimensions

---

## Backward Compatibility

**Status**: Maintained for Pydantic V2 models

**Breaking Changes**: None (internal validation only)

**Compatibility Matrix**:

| Interface | v1.0 | Notes |
|-----------|------|-------|
| DatasetSample | ✅ Compatible | Pydantic V2 model |
| TransformOutput | ✅ Compatible | Pydantic V2 model |
| CollateOutput | ✅ Compatible | Pydantic V2 model |

---

## References

- [Preprocessing Data Contracts](preprocessing-data-contracts.md)
- [Inference Data Contracts](inference-data-contracts.md)
- [System Architecture](../architecture/system-architecture.md)
