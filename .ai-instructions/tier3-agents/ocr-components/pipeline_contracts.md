# Component: Pipeline Contracts (Shared Data Models)

## Role
Defines the canonical data structures and contracts used to pass data between the stages of the OCR Inference Pipeline (Orchestrator -> Preprocessing -> Model -> Postprocessing).

## Critical Logic
- **Single Source of Truth**: These dataclasses/TypedDicts are the hard contracts between looseley coupled components.
- **Immutability**: Metadata objects (Padding, InferenceMetadata) should be treated as immutable once created.

## Data Contract

### 1. PreprocessingResult
**Passed from**: `PreprocessingPipeline` -> `InferenceOrchestrator` -> `ModelManager`
```python
@dataclass
class PreprocessingResult:
    tensor: torch.Tensor          # Shape: (1, 3, 640, 640), normalized, float32
    original_image: np.ndarray    # Shape: (H, W, 3), BGR
    meta: InferenceMetadata       # Transformation metadata for coordinate mapping
```

### 2. InferenceMetadata
**Passed from**: `PreprocessingPipeline` -> ... -> `InferenceResponse`
**Purpose**: Contains all logic needed to map coordinates back to the original image.

| Field | Type | Description |
|-------|------|-------------|
| `original_size` | `Tuple[int, int]` | (width, height) of raw input image |
| `processed_size` | `Tuple[int, int]` | (width, height) of image fed to model (typ. 640x640) |
| `padding` | `Dict[str, int]` | Keys: top, bottom, left, right |
| `padding_position` | `str` | **Constraint**: Always "top_left" (content at 0,0) |
| `content_area` | `Tuple[int, int]` | (width, height) of the image content within the canvas |
| `scale` | `float` | Resize scale factor applied during formatting |
| `coordinate_system` | `str` | Default: "pixel" |

### 3. TextRegion
**Passed from**: `PostprocessingPipeline` -> `InferenceOrchestrator`
**Purpose**: Represents a single detected text entity.

| Field | Type | Description |
|-------|------|-------------|
| `polygon` | `List[List[float]]` | shape (N, 2), coordinates in **Original Image Space** |
| `confidence` | `float` | 0.0 to 1.0 |
| `text` | `Optional[str]` | Detected text content (if recognition enabled) |

### 4. InferenceResponse
**Passed from**: `InferenceOrchestrator` -> `API`
**Purpose**: Final payload returned to the client.

| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | "success" or "error" |
| `regions` | `List[TextRegion]` | Detected regions |
| `meta` | `InferenceMetadata` | Context for client-side rendering |
| `preview_image_base64` | `Optional[str]` | Visual debug of detection (if requested) |
