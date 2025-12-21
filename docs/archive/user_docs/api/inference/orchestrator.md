---
type: api_reference
component: orchestrator
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# InferenceOrchestrator

## Purpose

Coordinates OCR inference pipeline workflow between ModelManager, PreprocessingPipeline, PostprocessingPipeline, and PreviewGenerator.

## Interface

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| `__init__` | `__init__(device: str \| None = None)` | None | - |
| `load_model` | `load_model(checkpoint_path: str, config_path: str \| None = None)` | bool | - |
| `predict` | `predict(image: np.ndarray, return_preview: bool = True, enable_perspective_correction: bool = False, perspective_display_mode: str = "corrected")` | dict[str, Any] \| None | - |
| `update_postprocessor_params` | `update_postprocessor_params(binarization_thresh: float \| None = None, box_thresh: float \| None = None, max_candidates: int \| None = None, min_detection_size: int \| None = None)` | None | - |
| `cleanup` | `cleanup()` | None | - |

## Dependencies

### Imports
- numpy
- logging

### Internal Components
- ModelManager
- PreprocessingPipeline
- PostprocessingPipeline
- PreviewGenerator
- config_loader (ModelConfigBundle, PostprocessSettings)
- dependencies (OCR_MODULES_AVAILABLE)

### Component Initialization
- ModelManager: Eager (created in `__init__`)
- PreprocessingPipeline: Lazy (created in `load_model` from config)
- PostprocessingPipeline: Lazy (created in `load_model` from config)
- PreviewGenerator: Eager (created in `__init__`)

## State

- **Stateful**: Yes (maintains references to 4 components)
- **Thread-safe**: No
- **Lifecycle**: initialized → loaded → ready → cleaned_up

### State Transitions

| From | To | Trigger |
|------|-----------|---------|
| initialized | loaded | Successful `load_model()` |
| loaded | ready | First `predict()` call |
| Any | cleaned_up | `cleanup()` call |

## Constraints

- Requires `load_model()` called before `predict()`
- Requires pipelines initialized (non-None) before inference
- Single concurrent inference per instance (not thread-safe)
- Device (GPU/CPU) determined at initialization, cannot change
- Perspective correction requires rembg module available

## Backward Compatibility

✨ **New Component**: Introduced in v2.0
- Replaces internal InferenceEngine logic
- InferenceEngine now delegates to InferenceOrchestrator
- No public API changes to InferenceEngine

See [contracts.md](contracts.md) for orchestrator pattern details.
