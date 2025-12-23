# Component: Postprocessing Logic (PostprocessingPipeline)

## Role
Decodes the raw model output (binarized segmentation maps) into structured `TextRegion` objects containing polygon coordinates and confidence scores.

## Critical Logic

### 1. Binarization
- **Input**: Probability map from model (0.0 to 1.0).
- **Operation**: `binary_map = prob_map > binarization_threshold` (default 0.2).
- **Output**: Binary mask (0 or 255).

### 2. Contour Extraction
- Uses `cv2.findContours` on the binary mask.
- Filters contours by size (`min_detection_size`, default 3.0).

### 3. Box Estimation
- For each contour, computes the minimum bounding box (`cv2.minAreaRect`).
- Filters boxes by `box_threshold` (default 0.6) to remove low-confidence noise.
- Unclips the box (expands it slightly) using the `Vatti clipping algorithm` (via `pyclipper`) to recover the full text area, counteracting the shrink-training of DBNet.
    - `offset = area * unclip_ratio / perimeter`

### 4. Coordinate Transformation
- The resulting polygons are in **Processed Space** (640x640).
- Uses `CoordinateManager.transform_polygon_to_original_space` to map them back to the original image coordinates (reversing padding and resizing).

## Data Contract
**Input**: `predictions_dict` (from Model) + `metadata`
**Output**: `List[TextRegion]`
