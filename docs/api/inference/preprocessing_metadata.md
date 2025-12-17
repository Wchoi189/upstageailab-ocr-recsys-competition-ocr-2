---
type: api_reference
component: preprocessing_metadata
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# PreprocessingMetadata

## Purpose

Creates preprocessing metadata dictionaries for inference pipeline with consistent transformation calculations.

## Interface

### Functions (All Stateless)

| Function | Signature | Returns | Raises |
|----------|-----------|---------|--------|
| `create_preprocessing_metadata` | `create_preprocessing_metadata(original_shape: Sequence[int], target_size: int = 640)` | dict[str, Any] | ValueError |
| `calculate_resize_dimensions` | `calculate_resize_dimensions(original_shape: Sequence[int], target_size: int = 640)` | Tuple[int, int, float] | ValueError |
| `calculate_padding` | `calculate_padding(original_shape: Sequence[int], target_size: int = 640)` | Tuple[int, int] | ValueError |
| `get_content_area` | `get_content_area(original_shape: Sequence[int], target_size: int = 640)` | Tuple[int, int] | ValueError |

## Dependencies

### Imports
- logging
- typing

### Internal Components
- coordinate_manager (calculate_transform_metadata)

### External Dependencies
- None

## State

- **Stateful**: No (all pure functions)
- **Thread-safe**: Yes
- **Lifecycle**: N/A (functional module)

## Constraints

- All functions build on coordinate_manager for consistency
- Default target size: 640
- Padding position: top_left (top=0, left=0, padding at bottom/right)
- Coordinate system: "pixel" (absolute pixel coordinates)
- Metadata contract: Matches InferenceMetadata format (see [contracts.md](contracts.md))
- Input validation: Raises ValueError for invalid shapes

## Metadata Dictionary Structure

Output from `create_preprocessing_metadata()`:

| Field | Type | Value |
|-------|------|-------|
| original_size | Tuple[int, int] | (width, height) before preprocessing |
| processed_size | Tuple[int, int] | (width, height) after preprocessing |
| padding | Dict[str, int] | Keys: top, bottom, left, right |
| padding_position | str | Always "top_left" |
| content_area | Tuple[int, int] | (width, height) of resized content |
| scale | float | Resize scale factor |
| coordinate_system | str | Always "pixel" |

## Backward Compatibility

✨ **New Component**: Introduced in v2.0
- Convenience wrapper around coordinate_manager
- Provides metadata dictionaries for pipeline use
- No public API changes to existing inference functions
