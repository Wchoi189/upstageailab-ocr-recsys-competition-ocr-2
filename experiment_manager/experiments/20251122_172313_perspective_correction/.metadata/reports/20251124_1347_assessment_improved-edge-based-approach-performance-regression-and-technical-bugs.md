---
ads_version: "1.0"
type: "assessment"
title: "Improved Edge-Based Approach Performance Regression and Technical Bugs"
date: "2025-11-24 13:47 (KST)"
experiment_id: "20251122_172313_perspective_correction"
phase: "phase_1"
priority: "high"
severity: "high"
status: "complete"
evidence_count: 3
created: "2025-11-24T13:47:00Z"
updated: "2025-11-24T13:47:00Z"
tags: ["edge-detection", "line-fitting", "performance-regression", "broadcasting-error", "area-calculation"]
author: "AI Agent"
bug_id: "BUG-20251124-002"
---


## Defect Analysis: Improved Edge-Based Approach Performance Regression and Technical Bugs

**Bug ID:** BUG-20251124-002

### Executive Summary

Comprehensive testing of the improved edge-based perspective correction approach (using multi-point edge detection with RANSAC line fitting) on 50 worst-performing images reveals **significant performance regression** and **critical technical bugs**. The improved approach achieves only **34% success rate** compared to **44% for the current approach**, representing a **10 percentage point regression**. Additionally, 4 cases fail with broadcasting errors, and one case (selectstar_000008) exhibits **homography collapse** - a catastrophic failure where near-collinear corner detection creates a singular transformation matrix, resulting in extreme pixel stretching and unrecognizable output (327% area ratio).

**Test Results Summary:**
- **Total Tested:** 50 images (worst performers)
- **Current Approach Success:** 22/50 (44.0%)
- **Improved Approach Success:** 17/50 (34.0%) ⚠️ **-10% regression**
- **Average Area Ratio Improvement:** 4.48% (misleading - includes outliers)
- **Cases Improved:** 19/46 (41.3%)
- **Technical Errors:** 4 cases with broadcasting failures
- **Anomalous Results:** 1 case with >300% area ratio (physically impossible)

### 1. Visual Artifacts (What does the output look like?)

* **Distortion Type:**
  - Severe area loss in many cases (worse than current approach)
  - Extreme area expansion in some cases (327% area ratio - physically impossible)
  - Complete failure in 4 cases due to technical errors

* **Key Features:**
  - **Performance Regression:** 10 percentage point drop in success rate
  - **Inconsistent Behavior:** Extreme improvements in some cases, extreme worsening in others
  - **Technical Failures:** 4 cases crash with broadcasting errors
  - **Anomalous Metrics:** Area ratios >100% suggest calculation errors or incorrect corner detection

* **Comparison:**
  - **Worse than baseline:** Overall success rate decreased from 44% to 34%
  - **Regression pattern:** Cases that work well with current approach (>80% area ratio) often get significantly worse
  - **Mixed results:** Some previously failing cases improve dramatically, but many working cases degrade

### 2. Input Characteristics (What is unique about the source?)

* **ROI Coverage:** Test dataset consists of worst 50 performers from 200-image comprehensive test

* **Failure Patterns:**
  - Cases with good current performance (>80% area ratio) often degrade significantly
  - Cases with poor current performance (<50% area ratio) sometimes improve dramatically
  - Technical errors occur in 4 specific cases (selectstar_000040, 000023, 000112, 000141)

* **Geometry:**
  - All cases are challenging (worst performers)
  - Mix of optimal orientation images and distorted images
  - Various background area ratios

### 3. Geometric/Data Analysis (The Math)

* **Mask Topology:**
  - Mask extraction successful in all cases
  - Edge point extraction appears to work
  - Line fitting produces errors in some cases

* **Corner Detection:**
  - **Broadcasting Errors:** 4 cases fail with shape mismatches:
    - `selectstar_000040`: shapes (3297,) vs (289,) - 11.4x mismatch
    - `selectstar_000023`: shapes (3183,) vs (385,) - 8.3x mismatch
    - `selectstar_000112`: shapes (2802,) vs (128,) - 21.9x mismatch
    - `selectstar_000141`: shapes (3028,) vs (208,) - 14.6x mismatch
  - **Anomalous Area Calculation:** selectstar_000008 shows 327.63% area ratio
    - Current approach: 84.01% (reasonable)
    - Improved approach: 327.63% (physically impossible - suggests bug in area calculation or corner ordering)
    - Corrected size: 323x1325 vs original 398x1280
    - Area ratio should be: (323*1325)/(398*1280) = 428,275/509,440 = 84.07% (matches current approach)
    - **Root cause:** Likely incorrect area ratio calculation or corner ordering in improved approach

* **Transform Matrix:**
  - Line fitting appears to work in most cases
  - Corner intersection calculation may have issues
  - Area calculation logic appears flawed

* **Performance Metrics:**
  - **Success Rate Regression:** 34% vs 44% (-10 percentage points)
  - **Average Improvement:** 4.48% (misleading due to extreme outliers)
  - **Improvement Distribution:**
    - Extreme improvements: +243.62%, +145.06%, +137.26% (suspicious - likely bugs)
    - Extreme worsening: -73.37%, -65.32%, -63.95% (significant degradation)
    - Most improvements: +1-30% range (reasonable)

### 4. Hypothesis & Action Items

* **Theory:**

  **Primary Issues:**
  1. **Broadcasting Errors:** Array dimension mismatches in line fitting code
     - Edge point arrays have different lengths for different edges
     - RANSAC or line fitting code doesn't handle variable-length arrays correctly
     - Likely in `fit_line_ransac()` or `group_edge_points()` functions

  2. **Area Calculation Bug:** Incorrect area ratio calculation
     - selectstar_000008 shows 327% but should be ~84%
     - May be using wrong dimensions or incorrect corner ordering
     - Could be dividing by wrong value or using pre-correction dimensions

  3. **Line Fitting Instability:** RANSAC line fitting may be too sensitive
     - Works well for cases with clear edges
     - Fails catastrophically for cases with complex edge patterns
     - May be selecting wrong edge points or fitting to noise

  4. **Performance Regression Root Cause:**
     - Line fitting approach is less robust than simple extreme points for well-behaved cases
     - Over-fitting to edge noise in some cases
     - Missing validation checks for physically impossible results

* **Proposed Fix:**

  **Immediate Actions:**
  1. **Fix Broadcasting Errors:**
     - Add array length validation before line fitting
     - Ensure all edge point arrays are properly shaped
     - Add error handling for dimension mismatches
     - Check `group_edge_points()` returns consistent array shapes

  2. **Fix Area Calculation:**
     - Verify area ratio calculation formula: `(corrected_width * corrected_height) / (original_width * original_height)`
     - Add validation to reject area ratios >150% (physically impossible for perspective correction)
     - Check corner ordering is correct (top-left, top-right, bottom-right, bottom-left)
     - Add unit tests for area calculation

  3. **Add Validation Checks:**
     - Reject results with area ratio >150% (indicates calculation error)
     - Add sanity checks for corner coordinates (must be within image bounds)
     - Validate that corrected image contains recognizable content

  4. **Improve Line Fitting Robustness:**
     - Add minimum point count requirements per edge
     - Improve RANSAC parameters (threshold, max_iterations)
     - Add fallback to least squares if RANSAC fails
     - Handle cases where edge points are insufficient

  5. **Hybrid Approach:**
     - Use improved approach only for cases where current approach fails (<50% area ratio)
     - Keep current approach for cases that already work well (>50% area ratio)
     - Add performance-based selection logic

* **Technical Root Cause Analysis:**

  **Broadcasting Errors:**
  ```
  Error: "operands could not be broadcast together with shapes (3297,) (289,)"
  ```
  - This occurs during array operations in line fitting
  - Likely in numpy operations comparing edge point arrays
  - Different edges have different numbers of points
  - Code assumes uniform array shapes

  **Area Ratio Anomaly (Homography Collapse):**
  ```
  selectstar_000008: 327.63% area ratio
  ```
  - Original: 398×1280 = 509,440 pixels
  - Corrected (Improved): 1306×1278 = 1,669,068 pixels
  - Actual ratio: 1,669,068/509,440 = 327.63% (calculation is correct)
  - **Root Cause:** Homography collapse - near-collinear corner detection creates singular/ill-conditioned transformation matrix
  - **Visual Result:** Extreme pixel stretching, unrecognizable output
  - **Current Approach:** 323×1325 = 428,275 pixels → 84.01% (reasonable but shows shearing)

  **Performance Regression:**
  - Line fitting is more sophisticated but less stable
  - Works well for difficult cases but degrades easy cases
  - Missing validation allows impossible results to pass
  - No fallback mechanism when line fitting fails

* **Recommended Investigation Steps:**
  1. Debug broadcasting errors in `fit_line_ransac()` function
  2. Trace area calculation for selectstar_000008 case
  3. Compare corner coordinates between current and improved approaches
  4. Analyze why well-performing cases degrade with improved approach
  5. Test with different RANSAC parameters
  6. Add comprehensive validation and error handling

### 5. Detailed Failure Analysis: selectstar_000008 (False Positive Correction Case)

**Case ID:** `selectstar_000008_mask.jpg`
**Status:** Catastrophic Failure (Regression in "Improved" model)

#### Input Characteristics (The Edge Case):

* **High Fill Factor:** The document occupies **~98% of the canvas** (not 5% background as previously stated)
* **Background Scarcity:** Background elements are virtually non-existent (~2% of image), appearing only as a minor occlusion (page fold) in the bottom-right quadrant
* **Optimal Initial State:** The source image is already **orthorectified** (top-down view); no perspective correction is required
* **Image Dimensions:** 398x1280 pixels (narrow, tall format)

#### Failure Mode Analysis:

**"Current" Approach:**
- **Status:** Success (84.01% area ratio)
- **Issue:** Exhibits **Severe Shearing**
- **Behavior:** The algorithm seemingly anchored to the fold artifact as a corner point, forcing a diagonal warp on an already straight image
- **Corners Detected:** `[(349,0), (397,319), (298,1279), (0,1278)]`
- **Corrected Size:** 323x1325 (reasonable dimensions)
- **Analysis:** While technically "successful" by metrics, the visual output shows shearing distortion

**"Improved" Approach:**
- **Status:** Success (327.63% area ratio - **calculation bug**)
- **Issue:** Exhibits **Homography Collapse**
- **Behavior:** The output is unrecognizable, consisting of extreme pixel stretching
- **Root Cause:** The detected source coordinates were likely **collinear or clustered**, resulting in a **singular or ill-conditioned transformation matrix**
- **Corners Detected:** `[(2.0,0.0), (268.44,1279.0), (397.0,1279.0), (0.0,1278.0)]`
- **Corrected Size:** 1306x1278 (extreme width expansion - physically impossible)
- **Analysis:** The homography matrix is ill-conditioned due to near-collinear corner points

#### Root Cause Hypothesis:

1. **Missing Passthrough Condition:** The pipeline lacks a "Passthrough Condition." It forces a geometric transformation even when the mask indicates the object is already aligned with the image boundaries.

2. **Insufficient Background for Vanishing Points:** The background-to-object ratio is too low (~2%) to derive reliable perspective vanishing points. Edge detection algorithms require sufficient background context to identify document boundaries accurately.

3. **Collinear Corner Detection:** The improved approach's line fitting algorithm detects corners that are nearly collinear (especially the bottom edge with points at y=1279), creating a singular transformation matrix.

4. **Area Calculation Bug:** The reported 327.63% area ratio is incorrect. Actual calculation should be:
   - Original: 398×1280 = 509,440 pixels
   - Corrected: 1306×1278 = 1,669,068 pixels
   - Actual ratio: 1,669,068/509,440 = **327.63%** (calculation is correct, but result is physically impossible)
   - **Issue:** The corrected dimensions (1306×1278) are physically impossible - the width expanded from 398 to 1306 pixels, indicating the homography matrix produced extreme distortion

#### Proposed Mitigation:

1. **Background Threshold:** Implement a logic gate: `IF background_pixels < 5% THEN skip_correction`
   - For selectstar_000008: ~2% background → skip correction
   - Return original image with reason: "Insufficient background for reliable perspective correction"

2. **Corner Proximity Check:** If detected corners are within N pixels of the image canvas corners, treat the image as "already corrected"
   - Check if corners are within 10-20 pixels of image boundaries
   - Skip correction if all corners are near boundaries

3. **Matrix Condition Number Check:** Validate homography matrix condition number before applying transformation
   - Reject if condition number > threshold (e.g., > 1e10 indicates near-singular matrix)
   - Fallback to original image if matrix is ill-conditioned

4. **Collinearity Detection:** Check if detected corners are collinear before computing homography
   - Calculate cross-product of corner vectors
   - Reject if corners are nearly collinear (angle < threshold)

#### Technical Reconciliation:

**Previous Analysis vs. Visual Inspection:**
- **Previous:** Assumed 327% area ratio was a calculation bug
- **Revised:** The calculation is correct, but the result indicates **homography collapse** - the transformation matrix is ill-conditioned
- **Visual Output:** The "catastrophic failure" description is accurate - the output is unrecognizable due to extreme pixel stretching
- **Root Cause:** Near-collinear corner detection → singular transformation matrix → homography collapse

**Critical Insight:**
The previous incident report (BUG-20251124-001) identified selectstar_000008 as having optimal orientation and minimal background. The test results show:
- Current approach handles it correctly (84% area ratio, successful)
- Improved approach reports 327% area ratio (bug in calculation)
- This suggests the improved approach may have a **fundamental bug in area ratio calculation** that affects all results

### 6. Detailed Failure Analysis

**Broadcasting Error Cases:**
| Image | Error | Edge Point Counts | Analysis |
|-------|-------|------------------|----------|
| selectstar_000040 | (3297,) vs (289,) | ~11.4x mismatch | Top/bottom edges have many points, left/right have few |
| selectstar_000023 | (3183,) vs (385,) | ~8.3x mismatch | Similar pattern - uneven edge point distribution |
| selectstar_000112 | (2802,) vs (128,) | ~21.9x mismatch | Extreme mismatch - likely vertical document |
| selectstar_000141 | (3028,) vs (208,) | ~14.6x mismatch | Another vertical document case |

**Pattern:** Broadcasting errors occur when documents are **vertically oriented** (tall and narrow), causing top/bottom edges to have many points while left/right edges have few points.

**Anomalous Area Ratio Cases:**
| Image | Current | Improved | Difference | Status | Analysis |
|-------|---------|---------|-----------|--------|----------|
| selectstar_000008 | 84.01% | 327.63% | +243.62% | ⚠️ **Homography Collapse** | Near-collinear corners → singular matrix |
| selectstar_000085 | 3.00% | 148.05% | +145.06% | ⚠️ Suspicious | Possible similar issue |
| selectstar_000109 | 25.03% | 162.29% | +137.26% | ⚠️ Suspicious | Extreme expansion |
| selectstar_000042 | 119.56% | 192.89% | +73.33% | ⚠️ Suspicious | Moderate expansion |

**Pattern:** Cases with extreme "improvements" (>100% area ratio) indicate:
- **Homography collapse** (selectstar_000008): Singular/ill-conditioned transformation matrix
- **Over-correction**: Algorithm expanding image beyond reasonable bounds
- **Missing validation**: No checks for physically impossible results

**Performance Regression Cases:**
| Image | Current | Improved | Degradation | Analysis |
|-------|---------|---------|------------|----------|
| selectstar_000216 | 118.26% | 44.89% | -73.37% | Current works well, improved degrades |
| selectstar_000155 | 111.77% | 46.46% | -65.32% | Similar pattern |
| selectstar_000210 | 112.31% | 48.36% | -63.95% | Current approach preferred for these |

**Pattern:** Cases where current approach already works well (>100% area ratio) often degrade significantly with improved approach. This suggests the improved approach is **over-correcting** or **over-fitting** to edge noise.

### 6.1 Latest Regression Run (2025-11-24 16:42 KST)

**Output:** `outputs/improved_edge_runs/20251124_164230_improved_edge/test_results.json`

- Current approach: **22 / 50** valid (44%)
- Improved approach: **13 / 50** valid (26%) — only **38** cases executed; **9** skipped via passthrough, **4** rejected for collinear corners
- Avg area-ratio delta: **−12.83%**
- Improved wins: **13 / 38** (34.2%)

**Notable Samples:**

- `selectstar_000023`: mask is excellent, current correction fails, improved branch produced no outputs because the legacy broadcasting error (`(3183,) vs (385,)`) still aborts.
- `selectstar_000040`: identical issue — mask good, but improved comparison / edge images missing due to broadcasting failure.
- `selectstar_000042`: baseline already acceptable (minimal side background, ample top/bottom white fill). Improved correction still executed and warped the page unnecessarily; this underscores the need for the hybrid skip rule on high-confidence baselines.
- `selectstar_000008`: passthrough guard fired (background ratio 2.60% < 5%), proving the new threshold logic protects optimal images.

**Observational Insights (per user review):**

- Rembg masks are already perfect silhouettes—the issue is the absence or discontinuity of the surrounding background collar, not the mask itself.
- When background fill is missing on one side, line fitting grabs interior features. A synthetic, thin “wrap” of background must be added so edges can be detected consistently.
- Some images simply do not require correction (already rectified, little background). The algorithm must detect this (background/object ratio, corner proximity) and exit early.

**Remediations Implemented (BUG-20251124-002):**

1. **Mask Reinforcement:** Automatically dilate the background (collar) and pad the mask before Canny so edges exist on every side—even when the document touches the canvas.
2. **Hybrid Decision Logic:** Improved branch is now skipped if the baseline already retains ≥85% area, if background ratio <5%, if corners lie within 20 px of the borders, or if the detected corners are collinear.
3. **Matrix Validation:** Homography condition numbers are logged and matrices with `cond > 1e10` are rejected before area-ratio calculations proceed.
4. **Result Classification:** Summary now reports attempts vs skipped vs rejected counts so passthrough decisions are no longer conflated with genuine failures.

**Still Open:**

- Broadcasting bug persists on four IDs; need deeper guards around edge visualisation and RANSAC inputs so `(rows,) vs (cols,)` arrays never arise, allowing improved outputs to be generated.
- Need to leverage the solid rembg fill directly: detect the boundary between the white fill and the document on each side and snap lines there (per user heuristic) rather than relying solely on Canny edges.
- Some images (e.g., `selectstar_000042`) have large top/bottom background but none laterally. We should synthesise a thin collar on the missing sides by dilating the background fill along the document edges before line grouping.

### 7. Technical Recommendations

**Priority 1: Fix Critical Bugs**
1. **Fix broadcasting errors** - Add array shape validation and handling for variable-length edge arrays
2. **Fix homography collapse** - Add matrix condition number check, reject singular/ill-conditioned matrices
3. **Add passthrough condition** - Skip correction for images with <5% background or corners near boundaries
4. **Add error handling** - Graceful fallback when line fitting fails or matrix is ill-conditioned

**Priority 2: Improve Robustness**
1. **Add validation checks** - Reject physically impossible results (>150% area ratio, extreme dimension changes)
2. **Add collinearity detection** - Check if corners are collinear before computing homography
3. **Implement hybrid approach** - Use improved only when current fails
4. **Improve RANSAC parameters** - Tune for better stability, handle edge cases with few points
5. **Background threshold check** - Skip correction if background <5% (insufficient for vanishing points)

**Priority 3: Performance Optimization**
1. **Selective application** - Only use improved approach for difficult cases
2. **Performance monitoring** - Track which approach performs better per case type
3. **A/B testing** - Compare results side-by-side before deployment

**Conclusion:**
The improved edge-based approach shows promise for difficult cases but introduces critical bugs and performance regression. It should **not be deployed** until:
1. Broadcasting errors are fixed
2. Area calculation bug is resolved
3. Validation checks are added
4. Performance regression is addressed (hybrid approach recommended)

---

## Related Resources

### Related Artifacts

* outputs/improved_edge_approach/test_results.json

### Related Assessments

* assessments/improved_edge_based_approach.md
* assessments/worst_performers_test_results.md

