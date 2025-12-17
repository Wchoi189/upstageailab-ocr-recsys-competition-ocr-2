---
type: data_reference
component: null
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Inference Data Contracts

**Purpose**: Coordinate transformation and padding contracts for inference pipeline.

## Component Mapping

| Data Contract | Source | Consumers |
|---------------|--------|-----------|
| InferenceMetadata | [preprocessing_metadata.py](../api/inference/preprocessing_metadata.md) | PreprocessingPipeline, PostprocessingPipeline, PreviewGenerator |
| PreprocessingResult | [PreprocessingPipeline](../api/inference/preprocessing_pipeline.md) | InferenceOrchestrator, PostprocessingPipeline |
| PostprocessingResult | [PostprocessingPipeline](../api/inference/postprocessing_pipeline.md) | InferenceOrchestrator |
| LoadedImage | [ImageLoader](../api/inference/image_loader.md) | InferenceEngine |

## InferenceMetadata

**Source**: preprocessing_metadata.py
**Version**: 1.0
**Backward Compatible**: ✅ Yes

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| original_size | Tuple[int, int] | Yes | (width, height) source image |
| processed_size | Tuple[int, int] | Yes | (width, height) preprocessed (typically 640x640) |
| padding | Dict[str, int] | Yes | Keys: top, bottom, left, right |
| padding_position | Literal["top_left", "center"] | Yes | Padding alignment strategy |
| content_area | Tuple[int, int, int, int] | Yes | (x, y, width, height) content bounds |
| scale | float | Yes | target_size / max(original_h, original_w) |
| coordinate_system | Literal["pixel", "normalized"] | Yes | Always "pixel" in current impl |

**Invariants**:
- `padding_position="top_left"` implies `padding.top=0` and `padding.left=0`
- `sum(processed_size) >= sum(original_size)`
- `content_area` fits within `processed_size`

## PreprocessingResult

**Source**: PreprocessingPipeline
**Version**: 1.0
**Backward Compatible**: ✅ Yes

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| batch | torch.Tensor | Yes | Preprocessed batch tensor for model |
| preview_image | np.ndarray | Yes | BGR image matching processed_size |
| original_shape | Tuple[int, int, int] | Yes | (H, W, C) before preprocessing |
| metadata | Dict[str, Any] | Yes | InferenceMetadata dict |
| perspective_matrix | np.ndarray \| None | No | Transform matrix if correction applied |
| original_image | np.ndarray \| None | No | Stored for "original" display mode |

## PostprocessingResult

**Source**: PostprocessingPipeline
**Version**: 1.0
**Backward Compatible**: ✅ Yes

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| polygons | str | Yes | Pipe-delimited polygon coordinates |
| texts | list[str] | Yes | Detected text regions |
| confidences | list[float] | Yes | Confidence scores per detection |

## LoadedImage

**Source**: ImageLoader
**Version**: 1.0
**Backward Compatible**: ✅ Yes

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| image | np.ndarray | Yes | BGR format, EXIF-normalized |
| orientation | int | Yes | EXIF orientation code (1-8) |
| raw_width | int | Yes | Width before EXIF rotation |
| raw_height | int | Yes | Height before EXIF rotation |
| canonical_width | int | Yes | Width after EXIF rotation |
| canonical_height | int | Yes | Height after EXIF rotation |

## Coordinate Transformation

### Top-Left Padding
- Content at (0, 0) in processed_size frame
- Padding on right/bottom only
- No translation: `x_original = x_processed * (original_w / resized_w)`

### Centered Padding
- Content centered in processed_size frame
- Padding distributed evenly
- Translation required: `x_original = (x_processed - pad_left) * (original_w / resized_w)`

## Polygon Coordinate Space

**Pixel coordinates**: Absolute pixels relative to processed_size frame.
- Top-left: Range [0, processed_w] x [0, processed_h]
- Centered: Range [pad_left, pad_left+content_w] x [pad_top, pad_top+content_h]

**Content area**: Actual image content bounds within processed_size frame.
- Top-left: (0, 0, resized_w, resized_h)
- Centered: (pad_left, pad_top, resized_w, resized_h)

## Frontend Contract

**Input**: InferencePreviewResponse with meta field
**Requirements**:
- Verify displayBitmap dimensions match meta.processed_size
- Use meta.padding_position for coordinate handling
- Apply display centering offsets (dx, dy) only
- Centered padding: coordinates already include padding offset

## Validation Rules

- padding_position must be present
- content_area must be present and valid
- padding values must match padding_position
- Coordinates must be within content_area bounds

## Related Contracts

- [Module Structure](module-structure.md)
- [Architecture Overview](../architecture/inference-overview.md)
- [OCR Inference Console Annotation Rendering](../../apps/ocr-inference-console/docs/annotation-rendering.md)
