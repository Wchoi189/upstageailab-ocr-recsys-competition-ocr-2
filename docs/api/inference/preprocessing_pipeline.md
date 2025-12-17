---
type: api_reference
component: preprocessing_pipeline
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# PreprocessingPipeline

## Purpose

Coordinates image preprocessing stages including optional perspective correction, resize, padding, normalization, and metadata calculation.

## Interface

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| `__init__` | `__init__(transform: Callable \| None = None, target_size: int = 640)` | None | - |
| `process` | `process(image: np.ndarray, enable_perspective_correction: bool = False, perspective_display_mode: str = "corrected")` | PreprocessingResult \| None | - |
| `process_for_original_display` | `process_for_original_display(original_image: np.ndarray)` | Tuple[np.ndarray, dict] \| None | - |
| `set_transform` | `set_transform(transform: Callable)` | None | - |
| `set_target_size` | `set_target_size(target_size: int)` | None | - |
| `from_settings` | `from_settings(settings: PreprocessSettings)` (classmethod) | PreprocessingPipeline | - |

## Dependencies

### Imports
- numpy
- logging
- dataclasses

### Internal Components
- config_loader (PreprocessSettings)
- preprocess (apply_optional_perspective_correction, build_transform, preprocess_image)
- preprocessing_metadata (create_preprocessing_metadata)

### External Dependencies
- Perspective correction: rembg module (optional)
- Transform pipeline: torchvision transforms (ToTensor, Normalize)
- Image processing: albumentations (LongestMaxSize, PadIfNeeded)

## State

- **Stateful**: Yes (maintains transform pipeline and target_size)
- **Thread-safe**: No
- **Lifecycle**: uninitialized → configured → ready

### State Variables

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| _transform | Callable \| None | None | Torchvision transform (ToTensor + Normalize) |
| _target_size | int | 640 | Target size for resize and padding |

## Constraints

- Requires transform set before calling `process()` (returns None if not set)
- Target size must be specified (default 640)
- Perspective correction mode: "corrected" or "original"
  - "corrected": Display corrected image with corrected polygons
  - "original": Display original image with inverse-transformed polygons
- Padding position fixed to "top_left" (content at 0,0, padding bottom/right)
- Input image format: BGR numpy array (H, W, C)
- Output tensor format: (1, C, H, W) normalized for model input

## Backward Compatibility

✨ **New Component**: Introduced in v2.0
- Extracted from InferenceEngine preprocessing logic
- No public API changes to InferenceEngine
- Return type `PreprocessingResult` wraps previous internal data structures
