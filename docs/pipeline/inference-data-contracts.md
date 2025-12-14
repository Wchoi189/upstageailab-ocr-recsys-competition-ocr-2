# Inference Pipeline Data Contracts

**Purpose**: Coordinate transformation and padding contracts for inference pipeline.

## InferenceMetadata

```python
{
    "original_size": Tuple[int, int],      # (width, height) source image
    "processed_size": Tuple[int, int],      # (width, height) preprocessed image (typically 640x640)
    "padding": {
        "top": int,
        "bottom": int,
        "left": int,
        "right": int
    },
    "padding_position": Literal["top_left", "center"],  # REQUIRED: Padding alignment
    "content_area": Tuple[int, int, int, int],          # REQUIRED: (x, y, width, height) content bounds in processed_size
    "scale": float,                        # target_size / max(original_h, original_w)
    "coordinate_system": Literal["pixel", "normalized"]
}
```

## Coordinate Transformation

### Top-Left Padding
- Content at `(0, 0)` in processed_size frame
- Padding on right/bottom only
- No translation offset: `x_original = x_processed * (original_w / resized_w)`

### Centered Padding
- Content centered in processed_size frame
- Padding distributed evenly (left/right, top/bottom)
- Translation required: `x_original = (x_processed - pad_left) * (original_w / resized_w)`

## Polygon Coordinate Space

**Pixel coordinates**: Absolute pixels relative to `processed_size` frame.
- Top-left: Coordinates in `[0, processed_w] x [0, processed_h]` range
- Centered: Coordinates in `[pad_left, pad_left+content_w] x [pad_top, pad_top+content_h]` range

**Content area**: Actual image content bounds within processed_size frame.
- Top-left: `(0, 0, resized_w, resized_h)`
- Centered: `(pad_left, pad_top, resized_w, resized_h)`

## Frontend Contract

**Input**: `InferencePreviewResponse` with `meta` field
**Requirements**:
- Verify `displayBitmap` dimensions match `meta.processed_size`
- Use `meta.padding_position` to determine coordinate handling
- Apply display centering offsets (dx, dy) only
- For centered padding: coordinates already include padding offset

## Validation Rules

- `padding_position` must be present (no default assumption)
- `content_area` must be present and valid
- `padding` values must match `padding_position` (top-left: top=0, left=0)
- Coordinates must be within content_area bounds

## Implementation Notes

### OCR Inference Console

The OCR Inference Console uses `preview_image_base64` to display annotations aligned with the coordinate system:

- **Preview Image**: Preprocessed image in `processed_size` space (matches coordinates)
- **Padding Trim**: Uses SVG `viewBox` to crop black padding using `content_area` calculated from padding metadata
- **No Coordinate Transformation**: Since image and coordinates are in the same space, direct mapping is used

See [`apps/ocr-inference-console/docs/annotation-rendering.md`](../../apps/ocr-inference-console/docs/annotation-rendering.md) for implementation details.

## Related Contracts

- [Pipeline Data Contracts](data_contracts.md#inference-engine-contract)
- [Coordinate Transformation](data_contracts.md#critical-areas---do-not-modify-without-tests)
- [OCR Inference Console Annotation Rendering](../../apps/ocr-inference-console/docs/annotation-rendering.md)
