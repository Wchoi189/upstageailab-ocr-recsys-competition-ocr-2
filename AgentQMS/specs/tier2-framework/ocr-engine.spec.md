---
ads_version: '2.0'
id: 'FW-OCR-ENGINE'
type: 'rule_set'
tier: 2
priority: 'high'
spec_version: '1.0.0'
updated: '2026-02-03'
description: 'OCR Engine standards consolidated from pipeline contracts, preprocessing, postprocessing, model management, orchestration, coordinates, and image loading.'
---

# OCR Engine Standards

> Consolidated orchestration, preprocessing, postprocessing, pipeline contracts, model management, coordinate transforms, and image loading standards.

## Overview

This specification consolidates all OCR engine component standards into a unified document for efficient agent consumption.

---

## Pipeline Contracts (from FW-026)

> Defines the canonical data structures and contracts used to pass data between pipeline stages.

> Defines the canonical data structures and contracts used to pass data between pipeline stages.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: pipeline_contracts
role: 'Single Source of Truth. These dataclasses/TypedDicts are the hard contracts
  between looseley coupled components.

  Immutability: Metadata objects (Padding, InferenceMetadata) should be treated as
  immutable once created.

  '
data_contract:
  types:
  - name: PreprocessingResult
    path: PreprocessingPipeline -> InferenceOrchestrator -> ModelManager
    fields:
    - name: tensor
      type: torch.Tensor
      description: Shape (1, 3, 640, 640), normalized, float32
    - name: original_image
      type: np.ndarray
      description: Shape (H, W, 3), BGR
    - name: meta
      type: InferenceMetadata
      description: Transformation metadata for coordinate mapping
  - name: InferenceMetadata
    path: PreprocessingPipeline -> ... -> InferenceResponse
    description: Contains all logic needed to map coordinates back to the original
      image.
    fields:
    - name: original_size
      type: Tuple[int, int]
      description: (width, height) of raw input image
    - name: processed_size
      type: Tuple[int, int]
      description: (width, height) of image fed to model (typ. 640x640)
    - name: padding
      type: Dict[str, int]
      description: 'Keys: top, bottom, left, right'
    - name: padding_position
      type: str
      description: Always 'top_left'
    - name: content_area
      type: Tuple[int, int]
      description: (width, height) of the image content within the canvas
    - name: scale
      type: float
    - name: coordinate_system
      type: str
      description: Default 'pixel'
  - name: TextRegion
    path: PostprocessingPipeline -> InferenceOrchestrator
    description: Represents a single detected text entity.
    fields:
    - name: polygon
      type: List[List[float]]
      description: shape (N, 2), coordinates in Original Image Space
    - name: confidence
      type: float
      description: 0.0 to 1.0
    - name: text
      type: Optional[str]
  - name: InferenceResponse
    path: InferenceOrchestrator -> API
    description: Final payload returned to the client.
    fields:
    - name: status
      type: str
      description: success or error
    - name: regions
      type: List[TextRegion]
    - name: meta
      type: InferenceMetadata
    - name: preview_image_base64
      type: Optional[str]

```

---

## Orchestration Flow (from FW-024)

> The central controller for the OCR inference pipeline.

> The central controller for the OCR inference pipeline.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: orchestration_flow
role: 'The central ''brain'' that wires together the individual components. It ensures
  the data flows correctly from ImageLoader -> Preprocessing -> Model -> Postprocessing
  -> Preview.

  '
critical_logic:
- id: request-parsing
  description: Receives 'InferenceRequest' object.
- id: resource-check
  description: Verifies ModelManager has the correct checkpoint loaded. If not, triggers
    load (blocking).
- id: pipeline-execution
  description: '1. Load and Fix Orientation (ImageLoader)

    2. Preprocess (Resize/Pad) (PreprocessingPipeline)

    3. Inference (ModelManager)

    4. Postprocess (Decode Polygons) (PostprocessingPipeline)

    5. Preview (Optional Visualization) (PreviewGenerator)

    '
- id: error-handling
  description: Catches internal decoding errors and wraps them in 'OCRBackendError'.
data_contract:
  input:
  - field: InferenceRequest
    type: object
  output:
  - field: InferenceResponse
    type: Dict

```

---

## Preprocessing Logic (from FW-028)

> Prepares raw images for inference.

> Prepares raw images for inference.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: preprocessing_logic
role: 'Applies resolution standardization, padding, normalization, and optional geometric
  corrections.

  '
critical_logic:
- id: pipeline-stages
  description: "1. Decode: Convert Base64/Bytes to BGR Numpy Array (Opencv format).\n\
    2. Perspective Correction (Optional):\n   - If 'enable_perspective_correction=True'.\n\
    \   - Uses 'rembg' or corner detection logic to find document corners.\n   - Warps\
    \ image to 'flat' view.\n   - Note: This changes the 'original_size' effective\
    \ for downstream coordinate mapping.\n3. Resize & Pad:\n   - Resizes to 'target_size'\
    \ (default 640) preserving aspect ratio.\n   - Pads right/bottom to reach exact\
    \ 640x640 square.\n4. Normalize:\n   - Standard ImageNet mean/std (if generic\
    \ model) or custom stats.\n   - Converts to Float32 Tensor (1, C, H, W).\n"
- id: configuration
  description: 'Configuration is loaded from Hydra V5.0 domain configs (domain/detection.yaml,
    etc.) or can be overridden per request.

    - target_size: int (default 640)

    - transform_pipeline: List[str] (names of albumentations transforms)

    See: AgentQMS/standards/tier2-framework/hydra-v5-rules.yaml

    '
- id: display-modes
  description: 'Corrected: The preview image shows the ''After Warp'' view.

    Original: The preview image shows the raw input, requiring inverse-inverse mapping
    (complex). Currently defaults to Corrected view for simplicity.

    '
data_contract:
  input:
  - field: image
    type: np.ndarray
  - field: settings
    type: PreprocessSettings
  output:
  - field: result
    type: PreprocessingResult
    description: Tensor + Metadata + OriginalImage

```

---

## Postprocessing Logic (from FW-027)

> Decodes raw model output into structured TextRegion objects.

> Decodes raw model output into structured TextRegion objects.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: postprocessing_logic
role: 'Decodes the raw model output (binarized segmentation maps) into structured
  ''TextRegion'' objects containing polygon coordinates and confidence scores.

  '
critical_logic:
- id: binarization
  description: 'Input: Probability map from model (0.0 to 1.0).

    Operation: binary_map = prob_map > binarization_threshold (default 0.2).

    Output: Binary mask (0 or 255).

    '
- id: contour-extraction
  description: 'Uses ''cv2.findContours'' on the binary mask.

    Filters contours by size (''min_detection_size'', default 3.0).

    '
- id: box-estimation
  description: 'For each contour, computes the minimum bounding box (''cv2.minAreaRect'').

    Filters boxes by ''box_threshold'' (default 0.6) to remove low-confidence noise.

    Unclips the box (expands it slightly) using the ''Vatti clipping algorithm'' (via
    ''pyclipper'') to recover the full text area, counteracting the shrink-training
    of DBNet.

    Logic: offset = area * unclip_ratio / perimeter.

    '
- id: coordinate-transformation
  description: 'The resulting polygons are in Processed Space (640x640).

    Uses ''CoordinateManager.transform_polygon_to_original_space'' to map them back
    to the original image coordinates (reversing padding and resizing).

    '
data_contract:
  input:
  - field: predictions_dict
  - field: metadata
  output:
  - field: regions
    type: List[TextRegion]

```

---

## Model Management (from FW-023)

> Manages the lifecycle of models (detection, recognition, etc.) for loading, inference, and unloading.

> Manages the lifecycle of models (detection, recognition, etc.) for loading, inference, and unloading.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: model_management
role: 'Abstracts away the PyTorch Lightning / Hydra complexity. Provides a simple
  ''predict(tensor)'' interface to the orchestrator.

  '
critical_logic:
- id: checkpoint-loading
  description: 'Loads weights from `.ckpt` file.

    Optimization: Caches the model in memory (Singleton pattern) to avoid reloading
    per request.

    Reason: Serverless-like behavior (cold start vs warm start).

    '
- id: configuration-inference
  description: 'Models require 4 components: ''Encoder'', ''Decoder'', ''Head'', ''Transforms''.

    These are defined in Hydra V5.0 domain configs (domain/detection.yaml, domain/recognition.yaml,
    etc.).

    Auto-Discovery: If ''checkpoint.pth'' is at ''/foo/bar/checkpoints/epoch=1.ckpt'',
    manager looks for ''.hydra/config.yaml'' or ''config.yaml'' in parent directories.

    See: AgentQMS/standards/tier2-framework/hydra-v5-rules.yaml

    '
- id: resource-management
  description: 'Cleanup: Calls ''torch.cuda.empty_cache()'' on model swap to prevent
    OutOfMemory (OOM) errors on 8GB GPUs.

    Device: Defaults to ''cuda'' if available, else ''cpu''.

    '
- id: initialization-stability
  description: '[BUG-001] DBHead biases must be initialized to background prior (p=0.01)
    to prevent "Red Line" saturation failure during scratch training.

    Uninitialized biases (~0.0) result in ~0.5 probability, causing all-positive masks.

    '
data_contract:
  input:
  - field: checkpoint_path
    description: API/Config provided path
  output:
  - field: LightingModule
    description: Loaded model on device

```

---

## Coordinate Transforms (from FW-012)

> Handles mapping between resized/padded images and original coordinates.

> Handles mapping between resized/padded images and original coordinates.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: coordinate_transforms
role: 'Ensures that bounding boxes detected on the 640x640 model input are accurately
  mapped back to the original full-resolution image, accounting for padding and resizing.

  '
critical_logic:
- id: padding-calculation
  description: 'Calculates padding (top, bottom, left, right) to maintain aspect ratio
    while fitting into target square.

    Logic: scale = target / max(h, w). new_size = older * scale. pad = (target - new)
    / 2.

    '
- id: inverse-mapping
  description: 'Applies the inverse of the affine transformation matrix used in preprocessing.

    Must handle both ''resize'' and ''pad'' operations.

    '
- id: coordinate-clamping
  description: 'Ensures transformed coordinates do not exceed original image dimensions
    (0, 0, W, H).

    '
data_contract:
  input:
  - field: polygons
    type: List[List[float]]
    description: Coordinates in processed space (640x640)
  - field: metadata
    type: InferenceMetadata
    description: Contains scale factor and padding values
  output:
  - field: original_polygons
    type: List[List[float]]
    description: Coordinates in original image space

```

---

## Image Loading Standards (from FW-020)

> Handles loading and validation of input images.

> Handles loading and validation of input images.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: image_loading
role: 'The entry point for raw data. Responsible for reading image bytes, validating
  formats, and correcting orientation (EXIF).

  '
critical_logic:
- id: format-support
  description: 'Must support: JPG, PNG, BMP, TIFF, WebP.

    Must reject: PDF (handled by separate PDF pipeline), GIF (unless first frame).

    '
- id: safe-loading
  description: 'Use ''Pillow'' or ''OpenCV'' with strict limits on image dimensions
    (Decompression Bomb protection).

    Max dims default: 10000x10000.

    '
- id: exif-correction
  description: 'Must apply EXIF orientation tag (e.g., rotate 90 deg for phone photos)
    BEFORE any processing.

    '
- id: channel-standardization
  description: 'Convert all inputs to BGR (OpenCV standard) or RGB (Pillow standard).

    Current Standard: BGR (np.array).

    '
data_contract:
  input:
  - field: image_source
    type: Union[str, bytes, Path]
    description: File path, url, or raw bytes
  output:
  - field: image_array
    type: np.ndarray
    description: BGR image array (H, W, 3)

```

---

