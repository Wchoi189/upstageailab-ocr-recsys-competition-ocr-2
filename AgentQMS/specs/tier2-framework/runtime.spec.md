# Runtime Specification

**Tier**: 2 (Framework)
**Scope**: Execution Flow, Image Handling, and Coordinate Systems.

## 1. Orchestration Flow

**Pattern**: `Saga / Linear Pipeline`

### Execution Order
1.  **Load Config**: Merge CLI + Hydra.
2.  **Mount Config**: Resolve paths.
3.  **Init Pipeline**: Instantiate classes.
4.  **Load Data**: `DataLoader` (Batching).
5.  **Process**: `run_batch()`.
6.  **Teardown**: Save metrics.

## 2. Image Loading

**Standard**: `OpenCV (BGR)`
*   **Loader**: `cv2.imread`
*   **Tensor format**: `NCHW` (Batch, Channels, Height, Width).
*   **Normalization**: `0-1` float for Models, `0-255` uint8 for IO.

### Constraints
*   **Large Images**: Must use tiled loading for > 4k res.
*   **Formats**: `.jpg`, `.png`, `.pdf` (via pdf2image).

## 3. Coordinate Systems
*   **Box Format**: `[x_min, y_min, x_max, y_max]` (XYXY).
*   **Polygons**: `[[x1,y1], [x2,y2], ...]` (Clockwise).
*   **Normalization**: Normalized coordinates `[0, 1]` preferred for storage.
