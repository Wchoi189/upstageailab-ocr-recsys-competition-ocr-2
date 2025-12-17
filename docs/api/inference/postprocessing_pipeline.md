---
type: api_reference
component: postprocessing_pipeline
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# PostprocessingPipeline

## Purpose

Decodes model predictions, transforms coordinates to original image space, and formats output in competition format.

## Interface

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| `__init__` | `__init__(settings: PostprocessSettings \| None = None)` | None | - |
| `process` | `process(model: Any, processed_tensor: Any, predictions: dict[str, Any], original_shape: Tuple[int, int, int])` | PostprocessingResult \| None | - |
| `set_settings` | `set_settings(settings: PostprocessSettings)` | None | - |

## Dependencies

### Imports
- logging
- dataclasses
- typing

### Internal Components
- config_loader (PostprocessSettings)
- postprocess (decode_polygons_with_head, fallback_postprocess)

### External Dependencies
- Model head: For primary decoding method
- OpenCV contours: For fallback detection method

## State

- **Stateful**: Yes (maintains postprocessing settings)
- **Thread-safe**: No
- **Lifecycle**: uninitialized → configured → ready

### State Variables

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| _settings | PostprocessSettings \| None | None | Thresholds and limits for postprocessing |

### PostprocessSettings Fields

| Field | Type | Purpose |
|-------|------|---------|
| binarization_thresh | float | Threshold for binarizing prediction maps |
| box_thresh | float | Confidence threshold for box filtering |
| max_candidates | int | Maximum number of candidate detections |
| min_detection_size | int | Minimum detection size in pixels |

## Constraints

- Primary method: decode using model head (returns structured output)
- Fallback method: contour-based detection (used if head decode fails)
- Requires settings configured for fallback method
- Coordinate transformation uses original_shape for inverse mapping
- Output format: pipe-separated polygon strings (competition format)
- Result includes method indicator: "head" or "fallback"

## Backward Compatibility

✨ **New Component**: Introduced in v2.0
- Extracted from InferenceEngine postprocessing logic
- No public API changes to InferenceEngine
- Return type `PostprocessingResult` wraps previous internal data structures
- Method indicator field added for debugging transparency
