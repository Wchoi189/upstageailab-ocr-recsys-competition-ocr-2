# BUG-2025-001: Inference padding/scaling mismatch causes invalid boxes and 0 predictions

Last Updated: 2025-10-21

## Summary

During inference, clean images occasionally return 0 predictions and the UI displays extremely large images. Model inputs are correctly resized to 640 and padded to 640×640, but post-processing maps detections back to the original image size using an incorrect inverse transform that ignores padding and uses padded dimensions as the scaling basis. This produces malformed coordinates that are filtered out or drawn out of bounds.

## Impact

- Intermittent 0 predictions on clean, high-resolution images
- Incorrect polygon/box coordinates after decoding
- Confusing UI presentation (large originals with no overlays)

Severity: High (affects inference correctness)

## Environment

- Branch: 11_refactor/preprocessing
- App: Streamlit Inference UI and runners/predict.py

## Reproduction

1. Launch inference UI (ui-infer) and select a trained checkpoint.
2. Upload a clean high-res image (e.g., drp.en_ko.in_house.selectstar_001007.jpg).
3. Run inference.

Observed: Some images return 0 predictions despite high model hmean. UI shows very large originals without overlays.

Expected: Non-zero predictions with correctly mapped polygons onto the original image.

## Root Cause Analysis

Training/validation/predict transforms (configs/base.yaml) apply:

- LongestMaxSize(max_size: 640)
- PadIfNeeded(min_width: 640, min_height: 640)

Therefore, the model always receives 640×640 inputs (with preserved aspect ratio and padding). Post-processing must invert both the resize and the padding steps to map results back to the original resolution.

Current implementation:

- File: ui/utils/inference/postprocess.py
  - compute_inverse_matrix(processed_tensor, original_shape) builds a 3×3 scale matrix using model_width/model_height taken from the padded tensor (typically 640×640), without accounting for padding offsets or the pre-pad resized size.
  - fallback_postprocess likewise scales bounding boxes using original_width/model_width and original_height/model_height derived from the padded map dims.

- File: ocr/models/head/db_postprocess.py
  - __transform_coordinates applies the provided inverse_matrix to decoded boxes. Since inverse_matrix lacks translation to remove padding and uses the padded size as the scale basis, the resulting coordinates are shifted/scaled incorrectly and may be discarded downstream.

Conclusion: Padding and correct scale from the pre-pad resized size are not considered, producing invalid coordinates.

## Affected Files and Functions (Index)

- configs/base.yaml
  - transforms.predict_transform.transforms[PadIfNeeded]
  - transforms.val_transform.transforms[PadIfNeeded]
  - transforms.test_transform.transforms[PadIfNeeded]

- ui/utils/inference/postprocess.py
  - compute_inverse_matrix(processed_tensor, original_shape)
  - fallback_postprocess(predictions, original_shape, settings)

- ocr/models/head/db_postprocess.py
  - __transform_coordinates(coords, matrix)
  - boxes_from_bitmap(pred, _bitmap, inverse_matrix)
  - polygons_from_bitmap(pred, _bitmap, inverse_matrix)

## Proposed Fix

Recommendation: Option B — set padding to top-left and simplify inverse mapping.

1) Config change (explicitly set top-left padding):

Add `position: "top_left"` to all PadIfNeeded steps for val/test/predict transforms.

Example (predict_transform):

```yaml
transforms:
  predict_transform:
    _target_: ${dataset_path}.DBTransforms
    transforms:
      - _target_: albumentations.LongestMaxSize
        max_size: 640
        interpolation: 1
        p: 1.0
      - _target_: albumentations.PadIfNeeded
        min_width: 640
        min_height: 640
        border_mode: 0
        position: "top_left"  # NEW
        p: 1.0
      - _target_: albumentations.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    keypoint_params: null
```

2) Post-process change (scale only by pre-pad resize factor):

In ui/utils/inference/postprocess.py:

- compute_inverse_matrix:
  - Compute scale = 640 / max(original_h, original_w)
  - Compute W1, H1 = round(original * scale)
  - Build inverse as [[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]] (no translation needed with top-left pad)

- fallback_postprocess:
  - Use model map dims as (H1, W1) = pre-pad resize dims, not padded dims; multiply contours by (W0/W1, H0/H1)

Note: If we cannot reliably recover pre-pad dims from tensors, recompute from original size and the known LongestMaxSize rule.

## Acceptance Criteria

- Given a portrait or landscape high-res image, decoded polygons align with the original image when visualized.
- No 0-prediction cases on clean images attributable to inverse mapping.
- No runtime errors in the UI or predict runner.
- Unit/integration test validating mapping for a synthetic box survives round-trip (resize → pad → inverse).

## Tests (Suggested)

- Add a test constructing a synthetic rectangle on the pre-pad map, apply PadIfNeeded(top_left), then verify inverse_matrix maps it back (within 1px tolerance) to original coordinates for both portrait and landscape aspect ratios.

## Notes

- Option A (center padding + translation) is mathematically correct but more error-prone; Option B reduces complexity and future maintenance burden.
