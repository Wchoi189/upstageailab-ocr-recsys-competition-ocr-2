---
type: api_reference
component: coordinate_manager
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# CoordinateManager

## Purpose

Transforms polygon coordinates between original image space and processed image space (640x640) with padding-aware calculations.

## Interface

### Functional API (Stateless)

| Function | Signature | Returns | Raises |
|----------|-----------|---------|--------|
| `calculate_transform_metadata` | `calculate_transform_metadata(original_shape: Sequence[int], target_size: int = 640)` | TransformMetadata | ValueError |
| `compute_inverse_matrix` | `compute_inverse_matrix(original_shape: Sequence[int], target_size: int = 640)` | np.ndarray | - |
| `compute_forward_scales` | `compute_forward_scales(original_shape: Sequence[int], target_size: int = 640)` | Tuple[float, float] | - |
| `transform_polygon_to_processed_space` | `transform_polygon_to_processed_space(polygon: np.ndarray, original_shape: Sequence[int], target_size: int = 640)` | np.ndarray | - |
| `transform_polygons_string_to_processed_space` | `transform_polygons_string_to_processed_space(polygons_str: str, original_shape: Sequence[int], target_size: int = 640, tolerance: float = 2.0)` | str | - |

### Class API (Stateful)

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| `__init__` | `__init__(target_size: int = 640)` | None | - |
| `set_original_shape` | `set_original_shape(original_shape: Sequence[int])` | None | - |
| `get_inverse_matrix` | `get_inverse_matrix(original_shape: Sequence[int] \| None = None)` | np.ndarray | ValueError |
| `get_forward_scales` | `get_forward_scales(original_shape: Sequence[int] \| None = None)` | Tuple[float, float] | ValueError |
| `transform_polygon_forward` | `transform_polygon_forward(polygon: np.ndarray, original_shape: Sequence[int] \| None = None)` | np.ndarray | ValueError |
| `transform_polygons_string_forward` | `transform_polygons_string_forward(polygons_str: str, original_shape: Sequence[int] \| None = None, tolerance: float = 2.0)` | str | ValueError |

## Dependencies

### Imports
- logging
- numpy
- collections.abc (Sequence)
- typing (NamedTuple)

### Internal Components
- None (standalone utility module)

## State

- **Functional API**: Stateless (pure functions)
- **Class API (CoordinateTransformationManager)**: Stateful (caches metadata)
- **Thread-safe**: Yes (functional API), No (class API)
- **Lifecycle**: N/A (functional), initialized → configured → ready (class)

### TransformMetadata (NamedTuple)

| Field | Type | Purpose |
|-------|------|---------|
| original_h | int | Original image height |
| original_w | int | Original image width |
| resized_h | int | Content height after resize |
| resized_w | int | Content width after resize |
| target_size | int | Target processed size (640) |
| scale | float | Resize scale factor |
| pad_h | int | Bottom padding pixels |
| pad_w | int | Right padding pixels |

## Constraints

- Padding convention: top_left position (content at 0,0, padding bottom/right only)
- No translation offset in transforms (padding position fixed)
- Target size default: 640x640
- Transformation algorithm: LongestMaxSize + PadIfNeeded
- Inverse matrix: processed space (640x640) → original space
- Forward scales: original space → processed space (separate x/y)
- Polygon format: pipe-separated coordinate strings ("x1 y1 x2 y2 ... | x1 y1 ...")
- Tolerance parameter: bounds checking tolerance in pixels (default 2.0)
- Critical function: `compute_inverse_matrix` - BUG-20251116-001 fix implemented

## Backward Compatibility

✨ **New Component**: Introduced in v2.0
- Consolidates duplicate logic from engine.py and postprocess.py
- No public API changes to existing inference functions
- BUG-20251116-001 fix: Corrected padding position handling in inverse transforms
