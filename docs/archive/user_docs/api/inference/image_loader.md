---
type: api_reference
component: image_loader
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# ImageLoader

## Purpose

Handles image loading from file paths, PIL images, or numpy arrays with EXIF normalization and BGR format conversion.

## Interface

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| `__init__` | `__init__(use_turbojpeg: bool = True, turbojpeg_fallback: bool = True)` | None | - |
| `load_from_path` | `load_from_path(image_path: str \| Path)` | LoadedImage \| None | - |
| `load_from_pil` | `load_from_pil(pil_image: Image.Image)` | LoadedImage | - |
| `load_from_array` | `load_from_array(image_array: np.ndarray, color_space: str = "BGR")` | LoadedImage | ValueError |

## Dependencies

### Imports
- logging
- pathlib
- cv2
- numpy
- PIL (Image)

### Internal Components
- ocr.utils.image_loading (load_image_optimized)
- ocr.utils.orientation (get_exif_orientation, normalize_pil_image)

### External Dependencies
- TurboJPEG: Optional JPEG loading optimization (faster than PIL)
- PIL: Image loading and EXIF handling
- OpenCV: Color space conversion (RGB → BGR)

## State

- **Stateful**: Yes (maintains TurboJPEG config)
- **Thread-safe**: Yes (no mutable shared state)
- **Lifecycle**: initialized → ready

### State Variables

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| use_turbojpeg | bool | True | Enable TurboJPEG for JPEG files |
| turbojpeg_fallback | bool | True | Fallback to PIL if TurboJPEG fails |

## Constraints

- EXIF orientation normalization: Applied to file and PIL inputs only (not arrays)
- EXIF orientation values: 1-8 (1 = normal, 2-8 = rotated/flipped)
- Output format: BGR numpy array (H, W, C) for OpenCV compatibility
- TurboJPEG: Only used for JPEG files, falls back to PIL for other formats
- Color space parameter for arrays: "BGR", "RGB", or "GRAY" (raises ValueError if invalid)
- Array input assumes no EXIF orientation (orientation = 1)
- PIL image cleanup: Normalized images closed if different from input

## Backward Compatibility

✨ **New Component**: Introduced in v2.0
- Extracted from InferenceEngine image loading logic
- Return type `LoadedImage` wraps previous internal data structures
- No public API changes to external callers
