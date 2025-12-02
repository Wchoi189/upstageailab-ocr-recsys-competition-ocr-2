# BUG-20251116-001: Candidate Files for Low Performance & Bad Alignment

## 🔴 **HIGH PRIORITY SUSPECTS** (Most Likely to Cause Issues)

### 1. **`ocr/models/head/db_postprocess.py`** ⭐⭐⭐
**Why:** Core coordinate transformation logic
- **Line 240:** `box = self.__transform_coordinates(box, inverse_matrix)` - Transforms polygons back to original coordinates
- **Line 150-163:** `__transform_coordinates()` method - Critical for coordinate frame conversion
- **Issue:** If `inverse_matrix` is wrong or transformation is incorrect, predictions will be misaligned
- **Impact:** Both performance scores AND WandB alignment

**Key Questions:**
- Is `inverse_matrix` being passed correctly?
- Is the transformation matrix correct?
- Are coordinates in the right format (x, y) vs (y, x)?

### 2. **`ocr/lightning_modules/utils/prediction_utils.py`** ⭐⭐⭐
**Why:** Formats predictions before evaluation/logging
- **Line 23:** `normalized_boxes = [np.asarray(box, dtype=np.float32).reshape(-1, 2) for box in boxes]`
- **Line 40-48:** Creates prediction entry with metadata
- **Issue:** If boxes are reshaped incorrectly or metadata is wrong, evaluation and visualization fail
- **Impact:** Both performance scores AND WandB alignment

**Key Questions:**
- Are boxes being reshaped correctly?
- Is metadata (orientation, raw_size, canonical_size) correct?
- Are coordinate frames consistent?

### 3. **`ocr/evaluation/evaluator.py`** ⭐⭐⭐
**Why:** Computes performance metrics
- **Line 95:** `det_quads = [polygon.reshape(-1).tolist() for polygon in prediction.boxes if polygon.size > 0]`
- **Line 98:** `canonical_gt = self._remap_ground_truth(gt_words, raw_width, raw_height, prediction.orientation)`
- **Line 99:** `gt_quads = [poly.reshape(-1).tolist() for poly in canonical_gt if poly.size > 0]`
- **Issue:** If GT remapping is wrong or coordinate frames don't match, metrics will be low
- **Impact:** Performance scores (recall, precision, hmean)

**Key Questions:**
- Is GT remapping correct?
- Do GT and pred coordinate frames match?
- Are polygons being flattened correctly?

### 4. **`ocr/utils/wandb_utils.py`** ⭐⭐
**Why:** Draws overlays on images
- **Line 521-532:** Draws GT boxes (green)
- **Line 535-545:** Draws pred boxes (red)
- **Line 553:** `cropped = _crop_to_content(img_uint8)` - Removes padding
- **Line 589-609:** Pads images to same size
- **Issue:** If coordinate conversion (RGB/BGR) or padding is wrong, overlays misalign
- **Impact:** WandB image alignment

**Key Questions:**
- Are coordinates in the right format for OpenCV?
- Is padding applied correctly?
- Is `_crop_to_content` working correctly?

## 🟡 **MEDIUM PRIORITY SUSPECTS**

### 5. **`ocr/lightning_modules/ocr_pl.py`** ⭐⭐
**Why:** Validation step orchestrates everything
- **Line 212:** `boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)`
- **Line 213:** `predictions = format_predictions(batch, boxes_batch)`
- **Line 220-221:** Stores `transformed_image` for WandB (but we removed this path)
- **Issue:** If batch data is wrong or predictions aren't stored correctly, downstream fails
- **Impact:** Both performance scores AND WandB alignment

**Key Questions:**
- Is batch data correct?
- Are predictions being stored correctly?
- Is `transformed_image` still being stored (we removed usage)?

### 6. **`configs/transforms/base.yaml`** ⭐⭐
**Why:** Defines image transforms
- **Line 36-45:** Validation transforms (LongestMaxSize, PadIfNeeded, Normalize)
- **Line 40-44:** `PadIfNeeded` with `border_mode: 0` and commented `position: "top_left"`
- **Issue:** If padding position is wrong or transforms don't match, coordinates will be misaligned
- **Impact:** Both performance scores AND WandB alignment

**Key Questions:**
- Is padding position correct? (commented out `position: "top_left"`)
- Do transforms match between training and validation?
- Is `inverse_matrix` computed correctly for these transforms?

### 7. **`ocr/metrics/cleval_metric.py`** ⭐
**Why:** Computes CLEval metrics
- **Issue:** If metric computation is wrong, scores will be low
- **Impact:** Performance scores only

**Key Questions:**
- Is polygon matching correct?
- Are IoU thresholds appropriate?
- Is the metric implementation correct?

## 🟢 **LOW PRIORITY SUSPECTS** (Less Likely but Worth Checking)

### 8. **`ocr/datasets/db_collate_fn.py`** ⭐
**Why:** Batches data and computes transforms
- **Issue:** If `inverse_matrix` is computed incorrectly, all downstream will be wrong
- **Impact:** Both performance scores AND WandB alignment

### 9. **`ocr/datasets/base.py`** ⭐
**Why:** Loads images and annotations
- **Issue:** If annotations are in wrong coordinate frame, everything fails
- **Impact:** Both performance scores AND WandB alignment

### 10. **`ocr/utils/orientation.py`** ⭐
**Why:** Handles EXIF orientation and polygon remapping
- **Issue:** If remapping is wrong, GT and pred won't align
- **Impact:** Both performance scores AND WandB alignment

## 📋 **Investigation Plan**

### Phase 1: Coordinate Frame Verification
1. **Check `db_postprocess.py`** - Verify `inverse_matrix` transformation
2. **Check `prediction_utils.py`** - Verify box formatting and metadata
3. **Check `evaluator.py`** - Verify GT remapping matches pred coordinate frame

### Phase 2: Transform Pipeline Verification
4. **Check `transforms/base.yaml`** - Verify padding position and transform consistency
5. **Check `db_collate_fn.py`** - Verify `inverse_matrix` computation

### Phase 3: Visualization Verification
6. **Check `wandb_utils.py`** - Verify coordinate conversion and padding
7. **Check `wandb_image_logging.py`** - Verify image loading and polygon processing

## 🔍 **Quick Diagnostic Commands**

```bash
# Check if inverse_matrix is being computed correctly
grep -r "inverse_matrix" ocr/datasets/

# Check coordinate transformations
grep -r "__transform_coordinates\|transform_coordinates" ocr/models/

# Check polygon remapping
grep -r "remap_polygons\|_remap_ground_truth" ocr/

# Check transform configs
grep -r "PadIfNeeded\|position.*top_left" configs/
```

## 📝 **Notes**

- The old working version (Oct 18) didn't use `transformed_image` path - always loaded from disk
- Padding position in transforms is commented out - may be causing misalignment
- Coordinate frame mismatches are the most likely root cause
- Both GT and pred need to be in the same coordinate frame for evaluation and visualization
