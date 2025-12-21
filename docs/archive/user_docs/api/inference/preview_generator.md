---
type: api_reference
component: preview_generator
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# PreviewGenerator

## Purpose

Encodes preview images as base64 JPEG and attaches them to prediction payloads with coordinate transformation metadata.

## Interface

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| `__init__` | `__init__(jpeg_quality: int = 85)` | None | ValueError |
| `encode_preview_image` | `encode_preview_image(preview_image: np.ndarray, format: str = "jpg")` | str \| None | ValueError |
| `attach_preview_to_payload` | `attach_preview_to_payload(payload: dict, preview_image: np.ndarray, metadata: dict \| None = None, transform_polygons: bool = True, original_shape: Tuple[int, int] \| None = None, target_size: int = 640)` | dict[str, Any] | - |

### Module-Level Function

| Function | Signature | Returns |
|----------|-----------|---------|
| `create_preview_with_metadata` | `create_preview_with_metadata(payload: dict, preview_image: np.ndarray, metadata: dict \| None = None, original_shape: Tuple[int, int] \| None = None, target_size: int = 640, jpeg_quality: int = 85)` | dict[str, Any] |

## Dependencies

### Imports
- base64
- logging
- numpy
- cv2

### Internal Components
- coordinate_manager (transform_polygons_string_to_processed_space)

### External Dependencies
- cv2: JPEG/PNG encoding with `imencode()`

## State

- **Stateful**: Yes (maintains jpeg_quality config)
- **Thread-safe**: Yes (no mutable shared state)
- **Lifecycle**: initialized → ready

### State Variables

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| jpeg_quality | int | 85 | JPEG encoding quality (0-100) |
| _encode_params | List[int] | [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] | cv2 encoding parameters |

## Constraints

- JPEG quality range: 0-100 (validated in `__init__`, raises ValueError if invalid)
- Supported formats: "jpg" (default) or "png"
- JPEG encoding reduces file size ~10x vs PNG while maintaining preview quality
- Transform polygons: maps coordinates from original space to preview space (640x640)
- Metadata attachment: warns if metadata is None (coordinate system incomplete)
- Input image format: BGR numpy array (OpenCV format)
- Output format: base64-encoded ASCII string

## Backward Compatibility

✨ **New Component**: Introduced in v2.0
- Extracted from InferenceEngine preview generation logic
- No public API changes to InferenceEngine
- JPEG encoding optimization (was PNG in v1.x, BUG-001 reference)
