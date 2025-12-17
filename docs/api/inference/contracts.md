---
type: api_reference
component: null
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Inference API Contracts

## Purpose

Documents orchestrator pattern, component initialization order, and data contracts between inference components.

## Orchestrator Pattern

### Coordination Model

InferenceOrchestrator delegates to 4 primary components without performing domain logic itself.

| Component | Initialization | Ownership | Lifecycle |
|-----------|---------------|-----------|-----------|
| ModelManager | Eager (in `__init__`) | Owned by orchestrator | Cleanup on orchestrator cleanup |
| PreprocessingPipeline | Lazy (in `load_model`) | Owned by orchestrator | Created from model config |
| PostprocessingPipeline | Lazy (in `load_model`) | Owned by orchestrator | Created from model config |
| PreviewGenerator | Eager (in `__init__`) | Owned by orchestrator | No cleanup needed |

### Initialization Order

1. `InferenceOrchestrator.__init__(device)` → Creates ModelManager, PreviewGenerator
2. `orchestrator.load_model(checkpoint, config)` → ModelManager loads model and config
3. After successful load → Create PreprocessingPipeline, PostprocessingPipeline from config
4. `orchestrator.predict(image)` → Execute full pipeline

## Component Initialization

### Required Sequence

```python
# Step 1: Create orchestrator (eager components initialized)
orchestrator = InferenceOrchestrator(device="cuda")

# Step 2: Load model (lazy components initialized from config)
success = orchestrator.load_model("checkpoint.pth", "config.yaml")

# Step 3: Execute inference (all components ready)
result = orchestrator.predict(image_array)
```

### Initialization Dependencies

| Step | Component Created | Requires | Config Source |
|------|------------------|----------|---------------|
| 1 | ModelManager | Device string | Constructor arg |
| 1 | PreviewGenerator | JPEG quality (default 85) | Constructor default |
| 2 | PreprocessingPipeline | Model config loaded | `config.preprocess` |
| 2 | PostprocessingPipeline | Model config loaded | `config.postprocess` |

## State Management

### Lifecycle States

| State | Condition | Available Methods |
|-------|-----------|-------------------|
| initialized | After `__init__` | `load_model` |
| loaded | After successful `load_model` | `predict`, `update_postprocessor_params` |
| ready | After first `predict` call | All methods |
| cleaned_up | After `cleanup` | None (must reinitialize) |

### State Transitions

- `initialized` → `loaded`: Successful `load_model()` call
- `loaded` → `ready`: First `predict()` execution
- Any state → `cleaned_up`: `cleanup()` call

## Error Handling

### Per-Component Error Behavior

| Component | Method | Returns on Error | Logs | Raises |
|-----------|--------|------------------|------|--------|
| ModelManager | `load_model` | `False` | ERROR | No |
| PreprocessingPipeline | `process` | `None` | EXCEPTION | No |
| PostprocessingPipeline | `process` | `None` | EXCEPTION | No |
| PreviewGenerator | `encode_preview_image` | `None` | WARNING | No |
| PreviewGenerator | `attach_preview_to_payload` | Original payload | EXCEPTION | No |
| ImageLoader | `load_from_path` | `None` | ERROR | No |
| CoordinateManager | Functions | Identity/empty | No | Yes (ValueError) |

### Orchestrator Error Propagation

`InferenceOrchestrator.predict()` returns `None` if any pipeline stage fails. Caller must check return value before using result.

## Data Contracts

### PreprocessingResult

Output from `PreprocessingPipeline.process()`.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| batch | torch.Tensor | Yes | Shape (1, C, H, W), normalized |
| preview_image | np.ndarray | Yes | BGR uint8 (H, W, C) for preview overlay |
| original_shape | Tuple[int, int, int] | Yes | (H, W, C) before preprocessing |
| metadata | Dict[str, Any] | Yes | Contains transformation parameters |
| perspective_matrix | np.ndarray \| None | No | 3x3 matrix if perspective correction applied |
| original_image | np.ndarray \| None | No | Original before perspective correction |

### PostprocessingResult

Output from `PostprocessingPipeline.process()`.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| polygons | str | Yes | Pipe-separated polygon coordinates |
| texts | List[str] | Yes | Text labels for each detection |
| confidences | List[float] | Yes | Confidence scores for each detection |
| method | str | Yes | "head" or "fallback" |

### LoadedImage

Output from `ImageLoader.load_from_path()`.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| image | np.ndarray | Yes | BGR uint8 (H, W, C) |
| orientation | int | Yes | EXIF orientation value (1-8) |
| raw_width | int | Yes | Width before EXIF normalization |
| raw_height | int | Yes | Height before EXIF normalization |
| canonical_width | int | Yes | Width after EXIF normalization |
| canonical_height | int | Yes | Height after EXIF normalization |

### InferenceMetadata

Embedded in final payload under `"meta"` key.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| original_size | Tuple[int, int] | Yes | (width, height) before preprocessing |
| processed_size | Tuple[int, int] | Yes | (width, height) after preprocessing |
| padding | Dict[str, int] | Yes | Keys: top, bottom, left, right |
| padding_position | str | Yes | Always "top_left" |
| content_area | Tuple[int, int] | Yes | (width, height) of resized content |
| scale | float | Yes | Resize scale factor |
| coordinate_system | str | Yes | Always "pixel" |

## Coordinate Transformations

### Transformation Contexts

| Context | From Space | To Space | Used By |
|---------|------------|----------|---------|
| Postprocessing | Model output (640x640) | Original image | PostprocessingPipeline |
| Preview overlay | Original image | Processed (640x640) | PreviewGenerator |
| Inverse perspective | Corrected image | Original image | InferenceOrchestrator |

### Padding Convention

**Fixed**: top_left position
- Content starts at (0, 0)
- Padding added to bottom and right only
- No translation offset needed for coordinate transforms

See [CoordinateManager](coordinate_manager.md) for transformation details.

## Backward Compatibility

✅ **Maintained**: No breaking changes
- All public method signatures unchanged
- Return types identical
- Error behavior preserved
- Data contract formats unchanged

See [inference-data-contracts.md](../reference/inference-data-contracts.md) for complete data specifications.
