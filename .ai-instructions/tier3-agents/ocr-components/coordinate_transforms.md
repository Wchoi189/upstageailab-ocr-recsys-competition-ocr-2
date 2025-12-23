# Component: Coordinate Transforms (CoordinateManager)

## Role
Handles the bidirectional transformation of polygon coordinates between the **Original Image Space** (variable size) and the **Processed Image Space** (fixed 640x640 model input). This is critical for accurate bounding box overlays.

## Critical Logic

### 1. Transformation Algorithm
The pipeline uses a **Longest Max Size** strategy with **Bottom/Right Padding**.

1.  **Scale Calculation**: `scale = target_size / max(original_h, original_w)`
2.  **Resize**: Image is resized to `(original_h * scale, original_w * scale)`.
3.  **Padding**:
    - `pad_w = target_size - resized_w`
    - `pad_h = target_size - resized_h`
    - Padding is applied to the **Right** and **Bottom** only (Top-Left alignment).

### 2. Inverse Transformation (Model -> Original)
To map a point $(x', y')$ from model output back to $(x, y)$ in original image:
$$ x = x' / scale $$
$$ y = y' / scale $$
*(Note: No translation offset is needed because padding is right/bottom and content starts at 0,0)*

### 3. Forward Transformation (Original -> Model)
$$ x' = x * scale $$
$$ y' = y * scale $$
*(Note: Used for creating ground-truth debug overlays or metrics)*

## Data Contract

### TransformMetadata
This component produces the `TransformMetadata` (subset of InferenceMetadata) used by the PreviewGenerator.

```python
class TransformMetadata(NamedTuple):
    original_h: int
    original_w: int
    resized_h: int
    resized_w: int
    target_size: int  # e.g., 640
    scale: float      # e.g., 0.5
    pad_h: int        # e.g., 100
    pad_w: int        # e.g., 0
```
