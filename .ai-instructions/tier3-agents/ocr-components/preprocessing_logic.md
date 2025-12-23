# Component: Preprocessing Logic (PreprocessingPipeline)

## Role
Prepares raw images for inference by applying resolution standardization, padding, normalization, and optional geometric corrections.

## Critical Logic

### 1. Pipeline Stages
1.  **Decode**: Convert Base64/Bytes to BGR Numpy Array (Opencv format).
2.  **Perspective Correction (Optional)**:
    - If `enable_perspective_correction=True`:
    - Uses `rembg` or corner detection logic to find document corners.
    - Warps image to "flat" view.
    - **Note**: This changes the `original_size` effective for downstream coordinate mapping.
3.  **Resize & Pad**:
    - Resizes to `target_size` (default 640) preserving aspect ratio.
    - Pads right/bottom to reach exact 640x640 square.
4.  **Normalize**:
    - Standard ImageNet mean/std (if generic model) or custom stats.
    - Converts to Float32 Tensor `(1, C, H, W)`.

### 2. Configuration (`PreprocessSettings`)
Configuration is loaded from `configs/predict/default_predict.yaml` but can be overridden per request.

- `target_size`: int (default 640)
- `transform_pipeline`: List[str] (names of albumentations transforms)

### 3. Display Modes
- **Corrected**: The preview image shows the "After Warp" view.
- **Original**: The preview image shows the raw input, requiring inverse-inverse mapping (complex). *Currently defaults to Corrected view for simplicity.*

## Data Contract
**Input**: `np.ndarray` (Image) + `PreprocessSettings`
**Output**: `PreprocessingResult` (Tensor + Metadata + OriginalImage)
