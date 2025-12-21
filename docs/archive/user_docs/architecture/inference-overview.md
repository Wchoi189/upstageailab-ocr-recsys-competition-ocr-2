---
type: architecture
component: null
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Inference Module Architecture Overview

## Purpose

Modular OCR inference system using orchestrator pattern to coordinate 8 specialized components.

## Component Breakdown

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| InferenceOrchestrator | `ocr/inference/orchestrator.py` | 274 | Coordinates pipeline workflow between components |
| ModelManager | `ocr/inference/model_manager.py` | 248 | Manages model lifecycle and caching |
| PreprocessingPipeline | `ocr/inference/preprocessing_pipeline.py` | 264 | Processes images with metadata calculation |
| PostprocessingPipeline | `ocr/inference/postprocessing_pipeline.py` | 149 | Decodes predictions to competition format |
| PreviewGenerator | `ocr/inference/preview_generator.py` | 239 | Encodes preview images with overlays |
| ImageLoader | `ocr/inference/image_loader.py` | 273 | Handles image I/O and EXIF normalization |
| CoordinateManager | `ocr/inference/coordinate_manager.py` | 410 | Transforms coordinates between image spaces |
| PreprocessingMetadata | `ocr/inference/preprocessing_metadata.py` | 163 | Creates metadata dictionaries for pipeline |

## Orchestrator Pattern

```
InferenceEngine (thin wrapper, 298 lines)
    ↓ delegates to
InferenceOrchestrator (coordinator)
    ├─→ ModelManager (model lifecycle)
    ├─→ PreprocessingPipeline (image preprocessing)
    ├─→ PostprocessingPipeline (prediction decoding)
    └─→ PreviewGenerator (preview encoding)

Helper Components (used by pipelines):
    ├─→ ImageLoader (used externally for file loading)
    ├─→ CoordinateManager (used by PreviewGenerator)
    └─→ PreprocessingMetadata (used by PreprocessingPipeline)
```

## Dependencies

### Component Relationships

| Component | Depends On | Used By |
|-----------|------------|---------|
| InferenceOrchestrator | ModelManager, PreprocessingPipeline, PostprocessingPipeline, PreviewGenerator | InferenceEngine |
| ModelManager | config_loader, model_loader, torch | InferenceOrchestrator |
| PreprocessingPipeline | preprocess, preprocessing_metadata, config_loader | InferenceOrchestrator |
| PostprocessingPipeline | postprocess, config_loader | InferenceOrchestrator |
| PreviewGenerator | coordinate_manager, cv2, base64 | InferenceOrchestrator |
| ImageLoader | PIL, cv2, image_loading utils | External callers |
| CoordinateManager | numpy | PreviewGenerator, PostprocessingPipeline |
| PreprocessingMetadata | coordinate_manager | PreprocessingPipeline |

### External Dependencies

- **torch**: Model execution and tensor operations
- **numpy**: Array operations and coordinate transformations
- **cv2**: Image encoding and color space conversion
- **PIL**: Image loading and EXIF handling

## Data Flow

```
1. Image Input (file path / PIL / numpy array)
       ↓
2. [Optional: ImageLoader] → LoadedImage (BGR array + metadata)
       ↓
3. PreprocessingPipeline.process()
   - Resize (LongestMaxSize)
   - Pad (PadIfNeeded, top_left position)
   - Normalize
   - Calculate metadata (via PreprocessingMetadata)
       ↓
   PreprocessingResult (batch tensor, preview image, metadata)
       ↓
4. ModelManager.model() → predictions dict
       ↓
5. PostprocessingPipeline.process()
   - Decode polygons (head or fallback)
   - Transform to original space (via CoordinateManager)
   - Format to competition format
       ↓
   PostprocessingResult (polygons string, texts, confidences)
       ↓
6. PreviewGenerator.attach_preview_to_payload()
   - Transform polygons to preview space (via CoordinateManager)
   - Encode preview image (JPEG base64)
   - Attach metadata
       ↓
7. Final Payload (polygons, texts, confidences, preview_image_base64, meta)
```

## Backward Compatibility

✅ **Maintained**: No breaking changes to public APIs
- `InferenceEngine` public methods unchanged
- Return types identical
- Exception behavior preserved
- Test coverage: 164/176 passing (93%)

See [backward-compatibility.md](backward-compatibility.md) for detailed compatibility statement.
