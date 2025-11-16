# BUG-20251116-001: WandB Validation Image Overlay Debugging Handover

**Bug ID:** BUG-20251116-001
**Date:** 2025-11-16
**Status:** In Progress - Root Cause Identified
**Severity:** Critical

## Executive Summary

**Root Cause Identified:** Prediction polygons are NOT being scaled to match the 640x640 transformed image, while GT polygons ARE being scaled. Both GT and pred polygons are in original/canonical coordinates (pred via `inverse_matrix` transform), but only GT is scaled to 640x640 for display.

**Fix Required:** Apply the same scaling logic to `pred_quads` that is currently applied to `gt_quads` in `wandb_image_logging.py` (lines 165-218).

**Additional Issue:** Possible Python bytecode caching preventing code changes from taking effect.

## Current Issue

WandB validation images still show:
- **Scale mismatch**: GT overlays are 4x too large in some cases
- **Rotation mismatch**: GT overlays rotated 90° CW compared to image (e.g., `drp.en_ko.in_house.selectstar_000030.jpg`)
- **Alignment issues**: Overlays not aligning correctly on images
- **No visible changes**: Code changes may not be taking effect (possible caching issue)

## End-to-End Dataflow Trace

### 1. Validation Step (`ocr/lightning_modules/ocr_pl.py:validation_step`)

```python
# Line 212: Model predicts on batch
boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)

# Line 213: Format predictions
predictions = format_predictions(batch, boxes_batch)

# Line 216-222: Store for WandB logging
for idx, prediction_entry in enumerate(predictions):
    filename = batch["image_filename"][idx]
    if batch_idx < 2 and idx < 8:  # Only first 2 batches, 8 images each
        prediction_entry["transformed_image"] = batch["images"][idx].detach()
    self.validation_step_outputs[filename] = prediction_entry
```

**Key Points:**
- `batch["images"]` = 640x640 transformed tensors (CHW, normalized)
- `boxes_batch` = polygons from `get_polygons_from_maps()` - **CHECK COORDINATE FRAME**
- `transformed_image` = 640x640 tensor (CHW, normalized, on GPU, then `.detach()`)

### 2. Format Predictions (`ocr/lightning_modules/utils/prediction_utils.py:format_predictions`)

```python
# Line 23: Normalize boxes
normalized_boxes = [np.asarray(box, dtype=np.float32).reshape(-1, 2) for box in boxes]

# Line 40-48: Create prediction entry
predictions.append({
    "boxes": normalized_boxes,  # Pred polygons - CHECK COORDINATE FRAME
    "orientation": batch.get("orientation", [1])[idx],
    "raw_size": tuple(batch.get("raw_size", [(0, 0)])[idx]),
    "canonical_size": tuple(batch.get("canonical_size", [None])[idx]),
    "metadata": metadata_entry,
})
```

**Key Questions:**
- What coordinate frame are `normalized_boxes` in?
- Are they in 640x640 (transformed) or original/canonical coordinates?
- Does `get_polygons_from_maps` apply `inverse_matrix` to map back to original?

### 3. WandB Image Logging Callback (`ocr/lightning_modules/callbacks/wandb_image_logging.py`)

#### 3.1 Image Preparation (Lines 76-116)

```python
# Line 76: Get transformed_image if available
transformed_image = entry.get("transformed_image")

if transformed_image is not None:
    # Line 83-85: Convert tensor to PIL
    pil_image = ImageProcessor.tensor_to_pil_image(transformed_image, mean=mean, std=std)
    # Result: 640x640 PIL Image (RGB)

# Line 68-69: Get GT polygons from dataset
gt_boxes = val_dataset.anns[filename]  # Original annotations
gt_quads = self._normalise_polygons(gt_boxes)  # Convert to numpy arrays

# Line 70: Get pred polygons
pred_quads = self._normalise_polygons(pred_boxes)  # From entry["boxes"]
```

**Key Questions:**
- What coordinate frame are GT polygons in? (likely canonical/original)
- What coordinate frame are pred polygons in? (need to verify)

#### 3.2 GT Polygon Processing (Lines 128-218)

**Current Logic:**
1. Determine `effective_orientation` (orientation_hint when using transformed_image)
2. Check if polygons are in canonical frame
3. Remap if needed (rotation)
4. Scale to 640x640 if using transformed_image

**Potential Issues:**
- Orientation might not be correct
- Canonical frame detection might be wrong
- Scaling might use wrong dimensions
- Double-scaling might occur

### 4. WandB Image Drawing (`ocr/utils/wandb_utils.py:log_validation_images`)

```python
# Line 479-485: Convert PIL Image to BGR for OpenCV
elif isinstance(image, PILImage.Image):
    arr = np.array(image)  # RGB numpy array
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # Convert to BGR

# Line 521-532: Draw GT boxes (green)
cv2.polylines(img_to_draw, [box_array], isClosed=True, color=(0, 255, 0), thickness=2)

# Line 536-545: Draw pred boxes (red)
cv2.polylines(img_to_draw, [box_array], isClosed=True, color=(0, 0, 255), thickness=2)

# Line 572-574: Convert back to RGB for WandB
if needs_rgb_conversion:
    img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
```

## Critical Questions to Answer

### Q1: What coordinate frame are prediction boxes in? ✅ ANSWERED

**Location:** `ocr/models/head/db_postprocess.py:polygons_from_bitmap()`

```python
# Line 239-240: Transform coordinates using inverse_matrix
box = self.__transform_coordinates(box, inverse_matrix)
```

**ANSWER:** Prediction boxes ARE transformed back to **original/canonical coordinates** using `inverse_matrix`. The `inverse_matrix` maps from 640x640 (transformed) back to original/canonical space.

**IMPLICATION:** Both GT and pred boxes are in original/canonical coordinates, but the displayed image is 640x640. We need to scale **BOTH** GT and pred boxes to 640x640, but currently we're only scaling GT boxes!

### Q2: What coordinate frame are GT polygons in?

**Location to check:** Dataset annotations (`val_dataset.anns[filename]`)

**Action:**
- Check what coordinate frame annotations are stored in
- Check if they're in raw or canonical frame
- Verify `polygon_frame` metadata

### Q3: Are code changes being executed?

**Possible causes:**
- Python bytecode cache (`.pyc` files)
- Module not reloaded
- Code path not being hit (conditional logic)

**Action:**
- Add print statements to verify code execution
- Check if `using_transformed_image` is True
- Verify `effective_orientation` value
- Check scale factor values

### Q4: Is the scaling logic correct?

**Current logic:**
```python
scale = 640.0 / max(canonical_w, canonical_h)
```

**Potential issues:**
- `canonical_size` might be wrong
- Dimensions might be swapped incorrectly
- Scale might be applied multiple times

## Debugging Plan

### Step 1: Add Debug Logging

Add print statements to trace the dataflow:

```python
# In wandb_image_logging.py, after line 91
print(f"[DEBUG BUG-20251116-001] {filename}: using_transformed_image={using_transformed_image}")
print(f"  - image.size: {image.size}")
print(f"  - raw_size_hint: {raw_size_hint}")
print(f"  - canonical_size_hint: {canonical_size_hint}")
print(f"  - orientation_hint: {orientation_hint}")
print(f"  - effective_orientation: {effective_orientation}")
print(f"  - GT quads count: {len(gt_quads)}, sample coords: {gt_quads[0][:2] if gt_quads else 'N/A'}")
print(f"  - Pred quads count: {len(pred_quads)}, sample coords: {pred_quads[0][:2] if pred_quads else 'N/A'}")
```

### Step 2: Verify Coordinate Frames

Check what coordinate frame each polygon set is in:

```python
# After getting polygons
if gt_quads:
    gt_sample = gt_quads[0]
    print(f"[DEBUG] GT polygon sample: {gt_sample}")
    print(f"  - Min coords: ({gt_sample[:, 0].min():.1f}, {gt_sample[:, 1].min():.1f})")
    print(f"  - Max coords: ({gt_sample[:, 0].max():.1f}, {gt_sample[:, 1].max():.1f})")

if pred_quads:
    pred_sample = pred_quads[0]
    print(f"[DEBUG] Pred polygon sample: {pred_sample}")
    print(f"  - Min coords: ({pred_sample[:, 0].min():.1f}, {pred_sample[:, 1].min():.1f})")
    print(f"  - Max coords: ({pred_sample[:, 0].max():.1f}, {pred_sample[:, 1].max():.1f})")
```

### Step 3: Check Model Output Coordinate Frame

**File:** `ocr/models/head/db_postprocess.py`

**Check:**
- Does `polygons_from_bitmap` apply `inverse_matrix`?
- What coordinate frame are polygons in after `get_polygons_from_maps`?
- Are they in 640x640 or original/canonical coordinates?

### Step 4: Verify Transform Pipeline

**Check:**
- What transforms are applied? (LongestMaxSize(640), PadIfNeeded(640, 640))
- What is the actual scale factor?
- Are dimensions swapped correctly for rotated images?

### Step 5: Test with Specific Image

**Target:** `drp.en_ko.in_house.selectstar_000030.jpg`

**Check:**
- What is its raw_size?
- What is its canonical_size?
- What is its orientation?
- What are the GT polygon coordinates?
- What are the pred polygon coordinates?
- What scale factor is computed?
- What is the final image size?

## Files Modified (BUG-20251116-001)

1. `ocr/utils/wandb_utils.py`
   - Added PIL Image import
   - Added RGB→BGR conversion for PIL Images
   - Added BGR→RGB conversion after drawing
   - Fixed pred overlay color (red in BGR)

2. `ocr/lightning_modules/callbacks/wandb_image_logging.py`
   - Added ImageProcessor import
   - Fixed ImageNet denormalization
   - Added rotation handling for transformed images
   - Added canonical frame detection
   - Added scaling logic for GT polygons
   - Added sanity checks

## Potential Root Causes

### 1. Prediction Boxes Not Scaled ✅ IDENTIFIED

**Root Cause:** Prediction boxes are transformed back to original/canonical coordinates via `inverse_matrix` in `db_postprocess.py:polygons_from_bitmap()` (line 240). However, in `wandb_image_logging.py`, we only scale GT polygons to 640x640, but **we do NOT scale pred polygons**.

**Fix Required:** Apply the same scaling logic to pred polygons that we apply to GT polygons when using transformed images.

**Code Location:** `ocr/lightning_modules/callbacks/wandb_image_logging.py` - Need to add scaling for `pred_quads` similar to GT scaling (lines 165-218).

### 2. Double Scaling

**Hypothesis:** GT polygons might be scaled twice - once incorrectly, then again.

**Check:** Trace through the scaling logic to see if it's applied multiple times.

### 3. Wrong Dimensions Used for Scaling

**Hypothesis:** `canonical_size` might be incorrect or in wrong format (width, height vs height, width).

**Check:** Verify what `canonical_size` actually represents and its format.

### 4. Rotation Not Applied Correctly

**Hypothesis:** Rotation remapping might not be applied, or applied incorrectly.

**Check:** Verify `effective_orientation` value and whether remapping is actually executed.

### 5. Code Not Executing

**Hypothesis:** Code changes might not be running due to:
- Python bytecode cache
- Conditional logic preventing execution
- Module not reloaded

**Action:** Add print statements to verify execution.

## Next Steps

### Immediate Fix Required

1. **Scale prediction polygons** - Apply the same scaling logic to `pred_quads` that we apply to `gt_quads` when using transformed images (lines 165-218 in `wandb_image_logging.py`)

2. **Clear Python cache** before testing:
   ```bash
   find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null
   find . -name "*.pyc" -delete
   ```

3. **Add debug logging** to verify:
   - Both GT and pred polygons are being scaled
   - Scale factors are correct
   - Coordinate frames match after scaling

### Verification Steps

1. **Test with specific problematic image** (`selectstar_000030.jpg`)
2. **Verify coordinate frames** of both GT and pred polygons after scaling
3. **Check rotation handling** - ensure both GT and pred are rotated consistently
4. **Verify overlay alignment** in WandB images

## Key Files to Investigate

1. `ocr/models/head/db_postprocess.py` - Check `get_polygons_from_maps` coordinate frame
2. `ocr/lightning_modules/callbacks/wandb_image_logging.py` - Current fix location
3. `ocr/utils/wandb_utils.py` - Image drawing logic
4. `ocr/lightning_modules/utils/prediction_utils.py` - Prediction formatting
5. `ocr/lightning_modules/ocr_pl.py` - Validation step

## Testing Command

```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete

# Run training with debug output
UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
  data=canonical \
  data/performance_preset=minimal \
  batch_size=4 \
  trainer.max_epochs=1 \
  trainer.limit_val_batches=0.1
```

## Notes

- The issue is NOT limited to wandb_utils - it involves the entire dataflow from validation_step → callback → wandb_utils
- Need to understand coordinate frame transformations at each step
- Caching might prevent code changes from taking effect
- Rotation and scaling need to be handled together, not separately

