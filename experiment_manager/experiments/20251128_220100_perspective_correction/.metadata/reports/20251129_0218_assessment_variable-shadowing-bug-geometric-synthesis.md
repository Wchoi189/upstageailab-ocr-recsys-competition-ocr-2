---
ads_version: '1.0'
type: report
title: Variable Shadowing Bug in Geometric Synthesis
date: 2025-11-29 02:18 (KST)
experiment_id: 20251128_220100_perspective_correction
phase: phase_3
priority: critical
severity: critical
status: complete
evidence_count: 1
created: '2025-11-29T02:18:00Z'
updated: '2025-12-27T16:16:43.463783'
tags:
- bug
- variable-shadowing
- geometric-synthesis
- regression
author: ai-agent
metrics: []
baseline: none
comparison: neutral
---
## Defect Analysis: Variable Shadowing Bug in Geometric Synthesis

### 1. Visual Artifacts (What does the output look like?)

* **Distortion Type:** Clipping/Truncation

* **Key Features:**
  - Fitted quadrilaterals stop short of reaching the full bounding box boundaries
  - Right and bottom edges are clipped by the amount of the object's margin/offset
  - Output appears limited to the left/top portion of the expected region
  - Geometric synthesis produces smaller-than-expected masks

* **Comparison:** Regression from expected behavior - geometric synthesis should produce full-size intersections but was producing truncated results

### 2. Input Characteristics (What is unique about the source?)

* **ROI Coverage:** Objects positioned away from image origin (e.g., x=100, y=50)

* **Contrast/Lighting:** Not applicable - this is a coordinate system bug

* **Geometry:** Objects with bounding boxes that don't start at (0,0) trigger the bug

### 3. Geometric/Data Analysis (The Math)

* **Mask Topology:**
  - Image dimensions: e.g., 1280x720 (full canvas)
  - Bounding box: e.g., x=100, y=50, w=1000, h=300 (object dimensions)
  - Bug: Canvas created with (h=300, w=1000) instead of (h=720, w=1280)
  - Result: Coordinates in image space (e.g., x=1100) exceed canvas width (1000), causing clipping

* **Corner Detection:** Corner coordinates are correct in image space, but canvas is too small

* **Transform Matrix:** Not applicable - this is a variable shadowing issue in Python

### 4. Root Cause Analysis

**The Bug:**
1. Line 996: `h, w = binary.shape[:2]` - Assigns **image dimensions** (e.g., 1280, 720)
2. Line 1046: `x, y, w, h = cv2.boundingRect(largest_component)` - **Overwrites** `h` and `w` with **bounding box dimensions** (e.g., 1000, 300)
3. Line 1151: `_geometric_synthesis(..., (h, w))` - Passes **bounding box size** instead of **image size** to synthesis function
4. Inside `_geometric_synthesis`: `np.zeros((h, w))` creates canvas with **object dimensions** (1000x300) instead of **image dimensions** (1280x720)
5. Coordinates in image space (e.g., x=1100) exceed canvas width (1000), causing right/bottom edges to be clipped

**Variable Shadowing:**
- Python allows variable shadowing where a variable name can be reused in the same scope
- The bounding box unpacking (`x, y, w, h = cv2.boundingRect(...)`) overwrote the image dimensions
- This caused the geometric synthesis function to receive incorrect canvas dimensions

### 5. Resolution

**Fix Applied:**
1. Renamed initial image dimensions to `img_h, img_w` to prevent shadowing:
   ```python
   img_h, img_w = binary.shape[:2]  # Image dimensions (preserved to avoid shadowing)
   total_pixels = float(img_h * img_w)
   ```

2. Updated `_geometric_synthesis` call to use image dimensions:
   ```python
   synthesized_corners = _geometric_synthesis(
       ordered,
       ordered_bbox,
       (img_h, img_w)  # Use image dimensions, not bounding box dimensions
   )
   ```

**Verification:**
- Tested on 25 worst performers: **25/25 success (100%)**
- All samples now produce correctly-sized geometric synthesis results
- No clipping or truncation observed

**Impact:**
- **Before Fix:** Geometric synthesis produced truncated results due to incorrect canvas size
- **After Fix:** Geometric synthesis correctly uses full image dimensions, producing complete intersections

---

## Related Resources

### Related Artifacts

* `artifacts/20251129_010932_dominant_extension/` - Initial geometric synthesis implementation (with bug)
* `artifacts/20251129_013700_dominant_extension/` - Post-fix verification (100% success)

### Related Assessments

* `assessments/run_02_coordinate_inversion_fix.md` - Previous bug fix (coordinate inversion)
* `assessments/run_03_geometric_synthesis.md` - Geometric synthesis implementation (to be created)

