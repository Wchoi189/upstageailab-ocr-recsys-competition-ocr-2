# Component: Visualization Logic (PreviewGenerator)

## Role
Draws detected bounding boxes and class labels onto the image for instant visual verification by the user.

## Critical Logic

### 1. Drawing Context
- **Tool**: OpenCV (`cv2.polylines`, `cv2.putText`).
- **Space**: Draws on the **Original Image** (using the original coordinates from `TextRegion`).

### 2. Optimization
- **Compression**: Encodes output as **JPEG** (quality 85) instead of PNG or raw bytes.
- **Reason**: Reduces payload size from ~10MB (Raw) to ~200KB (JPEG), critical for low-latency HTTP responses.

### 3. Visual Style
- **Color**: Green `(0, 255, 0)` for boxes.
- **Thickness**: Adaptive based on image resolution (so lines aren't invisible on 4K images).

## Data Contract
**Input**: `Original Image` + `List[TextRegion]`
**Output**: `Base64 String` (JPEG)
