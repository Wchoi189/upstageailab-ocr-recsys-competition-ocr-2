# Investigation & Mitigation Plan

## Identified Root Causes

### 1. Binarization Logic (`OCRModel` / `DBLoss`)
-   **Hypothesis**: `prob_maps` output from the detection head is entirely low-confidence (saturated near 0) or saturated high.
-   **Effect**: The contour finding algorithm in `get_polygons_from_maps` fails to find valid text regions, potentially defaulting to the entire image boundary or generating degenerate polygons.

### 2. Feature Extraction Metadata (`architecture.py`)
-   **Hypothesis**: The manual keys filtering in `OCRModel.forward` (added to prevent `torch.compile` tracing issues) might be stripping critical metadata like `src_scale`, `img_dims`, or `shape` from `kwargs`.
-   **Effect**: The Head or Post-processor defaults to an unscaled coordinate system (e.g., identity scale), causing massive scaling errors (4x shift) or coordinate collapse.

### 3. Data Denormalization (`wandb_utils.py`)
-   **Hypothesis**: `_crop_to_content` helper might be aggressively cropping predictions if the signal-to-noise ratio in prediction maps is low.
-   **Effect**: Visualization artifacts (like the "Red Line") might be an artifact of the visualization tool itself rather than the model, although the underlying low signal is the real issue.

## Recommended Mitigation Steps

### Phase 1: Diagnostic Hooks
-   [ ] **Instrument `OCRModel`**: Add logging to `generate` and `forward` to print min/max/mean of `prob_maps` and `thresh_maps`.
-   [ ] **Verify Inputs**: Check `kwargs` inside `forward` to ensure `shape` and `inverse_matrix` are present and correct.

### Phase 2: Post-Processing & Scaling
-   [ ] **Review `get_polygons_from_maps`**: Verify how it handles `src_scale`.
-   [ ] **Implement Letterbox Scaling**: Ensure coordinates are mapped back to original image space robustly, handling padding correctly.

### Phase 3: Upstream Stabilization
-   [ ] **Focus on Detection**: Disable Recognition (PARSeq) debugging until Detection maps are sane. "Garbage in, Garbage out" applies to the recognizer receiving collapsed crops.
