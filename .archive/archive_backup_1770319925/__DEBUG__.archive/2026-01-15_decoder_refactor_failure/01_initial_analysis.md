# Initial Analysis: The "Red Line" Failure

## Technical Failure Mechanisms

### 1. Zero-Area Polygon Degeneracy
-   **Symptom**: Predicted boxes collapse to lines (height $\approx$ 0).
-   **Likely Cause**: Failure in `Threshold Map` binarization or scaling logic. If the threshold map is incorrectly predicted or processed, the differentiable binarization may produce extremely thin or collapsed regions.

### 2. Coordinate Clipping & Saturation
-   **Symptom**: `max(x)` and `max(y)` align exactly with the image boundary.
-   **Mechanism**: Predicted coordinates are strictly larger than the canvas size, forcing `polygon_utils` or standard clipping functions to clamp them to the maximum edge (e.g., $W-1$ or $H-1$). This results in "stacking" of all detections at the right/bottom edge.

### 3. Resolution Mismatch
-   **Symptom**: Coordinates appear "shifted" or scaled by a factor (e.g., 2x or 4x).
-   **Mechanism**: The detection head operates on a downsampled feature map (e.g., P2 or P3 from FPN). If `get_polygons_from_maps` does not receive the correct `src_scale` or `downsample_ratio`, it may map feature-space coordinates (e.g., 0-160) directly to image-space (0-600) incorrectly, or vice-versa.

## Systematic Impact of Image Pre-processing

The failure correlates with **Canvas-to-Content Ratio** and **Aspect Ratio Distortion**.

| Metric               | Image 1 ("The Near-Square Illusion") | Image 2 ("Optimal Vertical Fit") |
| :------------------- | :----------------------------------- | :------------------------------- |
| **Horizontal Usage** | ~99.8% (Theoretical) / ~60% (Visual) | ~78%                             |
| **Padding**          | Right-heavy / Asymmetric             | Balanced / Narrower              |
| **Status**           | **COLLAPSE** (Red line)              | **SUCCESS** (Aligned)            |

**Analysis**:
-   **Image 1**: Extreme aspect ratio distortion likely compromised the vertical resolution during resizing (squashing text vertically). This "crushed" features might be pushing regression targets outside the learnable range or causing loss instability.
-   **Suggestion**: Verify if `Letterbox` resizing is preserving aspect ratio correctly and if the `OCRModel` handles padding masks properly.
