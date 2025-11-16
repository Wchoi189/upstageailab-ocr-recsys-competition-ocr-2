---
title: "WandB Validation Images Show Black Background Due to PIL-OpenCV Color Format Mismatch"
author: "ai-agent"
timestamp: "2025-11-16 18:08 KST"
branch: "main"
type: "bug_report"
category: "troubleshooting"
status: "open"
version: "1.0"
tags: ['bug', 'wandb', 'validation', 'image-logging', 'opencv', 'pil', 'critical']
bug_id: "BUG-20251116-001"
severity: "Critical"
---

# Bug Report: WandB Validation Images Show Black Background Due to PIL-OpenCV Color Format Mismatch

## Bug ID
BUG-20251116-001

## Summary

WandB validation images display completely black backgrounds with green annotation overlays and almost no red prediction overlays. This indicates a critical bug in the train/validation pipeline causing incorrect image visualization and misleading performance metrics (very high precision 0.878, very low recall 0.426).

## Environment

- **Pipeline Version:** Training/Validation pipeline
- **Components:** WandbImageLoggingCallback, log_validation_images, OpenCV drawing
- **Configuration:** WandB enabled, validation image logging active
- **Python Version:** Not specified
- **Dependencies:** OpenCV, PIL, numpy, wandb

## Steps to Reproduce

1. Run training with WandB validation image logging enabled
2. Check WandB dashboard for validation images
3. Observe images with black backgrounds, green GT overlays, and minimal red prediction overlays
4. Review metrics showing high precision (0.878) and low recall (0.426)

## Expected Behavior

- Validation images should display the original image content with proper colors
- Green overlays should show ground truth bounding boxes
- Red overlays should show predicted bounding boxes
- Images should match the actual transformed images seen by the model

## Actual Behavior

- Validation images show completely black backgrounds
- Green annotation overlays are visible (indicating GT boxes are being drawn)
- Almost no red prediction overlays (indicating predictions are missing or not being drawn)
- Performance metrics show disproportional precision (0.878) vs recall (0.426)

## Error Messages

No explicit error messages, but visual inspection of WandB dashboard shows:
- Completely black image backgrounds
- Green annotation overlays visible
- Almost no red prediction overlays

## Screenshots/Logs

Training log shows:
```
test/hmean: 0.5526533126831055
test/precision: 0.8780823945999146
test/recall: 0.4262494444847107
```

## Impact

- **Severity**: Critical
- **Affected Users**: All users monitoring training via WandB
- **Workaround**: None - visualization is completely broken, making it impossible to debug model performance

## Investigation

### Root Cause Analysis

**Color Format Mismatch:** PIL Images are passed to `log_validation_images` in RGB format, but the function uses OpenCV for drawing which expects BGR format. The conversion path for PIL Images falls into a fallback branch that doesn't properly handle the color channel conversion.

**Code Path:**
```
WandbImageLoggingCallback.on_validation_epoch_end()
├── Collects PIL Images (RGB format)
├── Passes PIL Images to log_validation_images()
└── log_validation_images()
    ├── PIL Images fall into else branch (line 479-483)
    ├── arr = np.array(image)  # RGB format preserved
    ├── OpenCV draws on RGB array (expects BGR)
    ├── Color channels are misinterpreted
    └── _crop_to_content() may crop aggressively if image appears black
```

**Key Issues:**
1. **PIL Image Handling:** In `ocr/utils/wandb_utils.py:479-483`, PIL Images are converted to numpy arrays without proper RGB→BGR conversion for OpenCV
2. **OpenCV Color Format:** OpenCV's `cv2.polylines` expects BGR format, but receives RGB arrays
3. **Image Conversion:** The fallback branch doesn't check for PIL Image type or handle color space conversion
4. **Cropping Function:** `_crop_to_content()` may be cropping too aggressively if the image appears mostly black due to color channel issues

**Location:**
- `ocr/utils/wandb_utils.py` (lines 419-553) - Main image logging function
- `ocr/lightning_modules/callbacks/wandb_image_logging.py` (lines 125-127) - Passes PIL Images

**Trigger:** Any validation epoch with WandB image logging enabled

### Related Issues

- High precision (0.878) and low recall (0.426) metrics suggest model may be making very few predictions
- Black background issue may be masking actual model performance
- Need to verify if predictions are actually being generated or if they're being lost in the visualization pipeline

## Proposed Solution

### Fix Strategy

Add proper PIL Image detection and RGB→BGR conversion in `log_validation_images()` before OpenCV drawing operations. Ensure images are converted to BGR format before any OpenCV operations, then convert back to RGB for WandB logging.

### Implementation Plan

1. ✅ Add PIL Image import and type checking in `log_validation_images()`
2. ✅ Add RGB→BGR conversion for PIL Images before OpenCV drawing
3. ✅ Add RGB→BGR conversion for Tensor images (also typically RGB)
4. ✅ Convert BGR back to RGB after OpenCV drawing for WandB logging
5. ⏳ Test with actual validation runs to confirm fix

**Implemented Fix (BUG-20251116-001):**

**File: `ocr/utils/wandb_utils.py`**

1. Added PIL Image import:
```python
from PIL import Image as PILImage
```

2. Added PIL Image handling with RGB→BGR conversion:
```python
elif isinstance(image, PILImage.Image):
    # BUG-20251116-001: PIL Images are in RGB format
    # Convert to numpy array (RGB) then to BGR for OpenCV
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # PIL Image is RGB, convert to BGR for OpenCV drawing
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        needs_rgb_conversion = True  # Mark for conversion back to RGB for WandB
```

3. Added Tensor image RGB→BGR conversion:
```python
# BUG-20251116-001: Tensor images are typically RGB, convert to BGR for OpenCV
if arr.ndim == 3 and arr.shape[2] == 3:
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    needs_rgb_conversion = True  # Mark for conversion back to RGB for WandB
```

4. Added BGR→RGB conversion after OpenCV drawing:
```python
# BUG-20251116-001: Convert BGR back to RGB for WandB logging (WandB expects RGB)
if needs_rgb_conversion and img_to_draw.ndim == 3 and img_to_draw.shape[2] == 3:
    img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
```

5. **Fixed ImageNet denormalization in callback (Additional Fix):**
   - The `_tensor_to_pil` method was using incorrect denormalization logic
   - It assumed [-1, 1] range, but ImageNet normalization uses mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - Replaced with proper `ImageProcessor.tensor_to_pil_image` method that correctly denormalizes

**File: `ocr/lightning_modules/callbacks/wandb_image_logging.py`**

```python
# BUG-20251116-001: Use proper ImageNet denormalization
# Default ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
pil_image = ImageProcessor.tensor_to_pil_image(transformed_image, mean=mean, std=std)
```

6. **Fixed coordinate scaling and color issues (Critical Fix):**
   - GT polygons were in canonical/original coordinates but displayed image was 640x640 (transformed)
   - Added scaling logic to scale GT polygons from canonical size to 640x640 using LongestMaxSize scale factor
   - Fixed pred overlay color: changed from (255, 0, 0) to (0, 0, 255) in BGR format (red in BGR, not blue)
   - Fixed legend color to match

7. **Fixed rotation mismatch and improved scaling logic (Critical Fix):**
   - Rotation was not handled correctly when using transformed_image (orientation was always 1)
   - Now uses orientation_hint from entry/metadata when using transformed_image
   - Added check for polygons already in canonical frame before remapping (prevents double-rotation)
   - Improved canonical dimension computation for scaling (accounts for dimension swap on rotation)
   - Added sanity check for suspicious scale factors (> 4.0 or < 0.1) with fallback to raw_size

**File: `ocr/lightning_modules/callbacks/wandb_image_logging.py`**

```python
# BUG-20251116-001: Scale GT polygons to match transformed image size (640x640)
# Transforms: LongestMaxSize(640) preserves aspect ratio, then PadIfNeeded(640, 640)
# Scale factor = 640 / max(canonical_w, canonical_h)
if using_transformed_image and canonical_size_hint:
    canon_w, canon_h = canonical_size_hint
    if canon_w > 0 and canon_h > 0:
        max_side = max(canon_w, canon_h)
        scale = 640.0 / max_side
        # Scale all GT polygon coordinates
        for quad in gt_quads:
            scaled_quad = quad.copy()
            scaled_quad[:, 0] *= scale
            scaled_quad[:, 1] *= scale
```

**File: `ocr/utils/wandb_utils.py`**

```python
# BUG-20251116-001: In BGR format, red is (0, 0, 255), not (255, 0, 0)
cv2.polylines(img_to_draw, [box_array], isClosed=True, color=(0, 0, 255), thickness=2)  # Red in BGR
cv2.line(img_to_draw, (8, 26), (32, 26), (0, 0, 255), 3)  # Red in BGR for legend
```

**File: `ocr/lightning_modules/callbacks/wandb_image_logging.py` (Additional Fixes)**

```python
# BUG-20251116-001: Use correct orientation when using transformed_image
effective_orientation = orientation if not using_transformed_image else orientation_hint

# Check if polygons are already in canonical frame before remapping
if effective_orientation != 1 and check_width > 0 and check_height > 0:
    if not polygons_in_canonical_frame(gt_quads, check_width, check_height, effective_orientation):
        # Polygons are in raw frame, need remapping
        needs_remapping = True

# Compute canonical dimensions correctly for scaling (account for dimension swap on rotation)
if canonical_size_hint:
    scale_w, scale_h = canonical_size_hint
elif raw_size_hint and effective_orientation != 1:
    raw_w, raw_h = raw_size_hint
    if orientation_requires_rotation(effective_orientation):
        if effective_orientation in {5, 6, 7, 8}:
            scale_w, scale_h = raw_h, raw_w  # Dimensions swap for rotated images
        else:
            scale_w, scale_h = raw_w, raw_h

# Sanity check for suspicious scale factors
if scale > 4.0 or scale < 0.1:
    # Fallback to raw_size if canonical_size seems wrong
    # ... (fallback logic)
```

### Testing Plan

1. ⏳ Run training with WandB validation image logging enabled
2. ⏳ Verify WandB images display correctly with proper backgrounds
3. ⏳ Verify GT and prediction overlays display correctly
4. ⏳ Verify performance metrics align with visual inspection
5. ⏳ Run integration/E2E tests to ensure no regressions

## Status

- [x] Confirmed
- [x] Investigating
- [x] Fix in progress
- [x] Fixed
- [ ] Verified

## Assignee

AI Agent (initial investigation)

## Priority

Critical

---

*This bug report follows the project's standardized format for issue tracking.*
