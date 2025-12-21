---
type: data_reference
component: null
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Inference Module Structure

**Purpose**: Component dependency graph and data flow for modular inference architecture.

## Component Dependency Graph

```
InferenceEngine (298L) ← thin wrapper
    └─→ InferenceOrchestrator (274L) ← coordination layer
          ├─→ ModelManager (248L)
          │     ├─→ model_loader.py
          │     └─→ config_loader.py
          ├─→ PreprocessingPipeline (264L)
          │     ├─→ ImageLoader (273L)
          │     ├─→ preprocessing_metadata.py (163L)
          │     └─→ preprocess.py
          ├─→ PostprocessingPipeline (149L)
          │     ├─→ postprocess.py
          │     └─→ CoordinateManager (410L)
          └─→ PreviewGenerator (239L)
                └─→ CoordinateManager (410L)
```

## Data Flow

```
Input (file path or np.ndarray)
    │
    ├─→ [ImageLoader] → LoadedImage (BGR, EXIF-normalized)
    │
    └─→ [PreprocessingPipeline]
          │
          ├─→ perspective_correction (optional)
          ├─→ resize + pad → PreprocessingResult
          │                   ├─ batch (torch.Tensor)
          │                   ├─ preview_image (np.ndarray)
          │                   ├─ original_shape
          │                   └─ metadata (InferenceMetadata)
          │
          └─→ [ModelManager]
                │
                └─→ model.forward(batch) → predictions (torch.Tensor)
                      │
                      └─→ [PostprocessingPipeline]
                            │
                            ├─→ decode (head or fallback)
                            └─→ coordinate_transform → PostprocessingResult
                                                        ├─ polygons (str)
                                                        ├─ texts (list)
                                                        └─ confidences (list)
                                  │
                                  └─→ [PreviewGenerator] (if return_preview=True)
                                        │
                                        ├─→ transform_polygons_to_preview_space
                                        ├─→ encode_preview_image (JPEG base64)
                                        └─→ attach_metadata → Final Response
                                                               ├─ polygons
                                                               ├─ texts
                                                               ├─ confidences
                                                               ├─ preview_image_base64
                                                               └─ meta
```

## Component Size Metrics

| Component | Lines | Responsibility |
|-----------|-------|---------------|
| engine.py | 298 | Backward-compatible wrapper |
| orchestrator.py | 274 | Pipeline coordination |
| model_manager.py | 248 | Model lifecycle |
| preprocessing_pipeline.py | 264 | Image preprocessing |
| postprocessing_pipeline.py | 149 | Prediction decoding |
| preview_generator.py | 239 | Preview encoding |
| image_loader.py | 273 | Image I/O + EXIF |
| coordinate_manager.py | 410 | Transformations |
| preprocessing_metadata.py | 163 | Metadata calculation |
| **Total (new components)** | **2020** | Modular architecture |

## Code Reduction

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| engine.py lines | 899 | 298 | -67% (-601L) |
| Responsibilities | 10+ | 1 (delegation) | -90% |
| Method complexity | High | Low | Simplified |

## Module Relationships

**Stateless**:
- preprocessing_metadata.py (pure functions)
- coordinate_manager.py (stateless transformations)

**Stateful**:
- ModelManager (model + config lifecycle)
- PreprocessingPipeline (transform state)
- PostprocessingPipeline (settings state)
- InferenceOrchestrator (component composition)

**I/O Boundary**:
- ImageLoader (file system)
- PreviewGenerator (base64 encoding)

## Import Dependencies

```python
# Primary Entry Point
ocr.inference.engine.InferenceEngine
    → ocr.inference.orchestrator.InferenceOrchestrator

# Core Components (used by orchestrator)
ocr.inference.model_manager.ModelManager
ocr.inference.preprocessing_pipeline.PreprocessingPipeline
ocr.inference.postprocessing_pipeline.PostprocessingPipeline
ocr.inference.preview_generator.PreviewGenerator

# Utilities (used by pipelines)
ocr.inference.image_loader.ImageLoader
ocr.inference.coordinate_manager.CoordinateTransformationManager
ocr.inference.preprocessing_metadata (functions)
```

## Test Coverage

| Component | Unit Tests | Status |
|-----------|-----------|--------|
| coordinate_manager | 45 | ✅ Pass |
| preprocessing_metadata | 30 | ✅ Pass |
| preview_generator | 31 | ✅ Pass |
| image_loader | 26 | ✅ Pass |
| preprocessing_pipeline | 12 | 🟡 Skip (requires torch) |
| postprocessing_pipeline | 9 | ✅ Pass |
| model_manager | 13 | ✅ Pass |
| orchestrator | 10 | ✅ Pass |
| **Total** | **176** | **164 Pass (93%)** |

## Related Documentation

- [Data Contracts](inference-data-contracts.md)
- [Backward Compatibility](../architecture/backward-compatibility.md)
- [Component APIs](../api/inference/)
