---
title: "Catastrophic Perspective Correction Failure on Optimal Orientation Images"
date: "2025-11-24 13:38 (KST)"
experiment_id: "20251122_172313_perspective_correction"
severity: "critical"
status: "investigating"
tags: ["perspective-correction", "edge-detection", "background-threshold", "optimal-orientation"]
author: "AI Agent"
bug_id: "BUG-20251124-001"
---


## Defect Analysis: Catastrophic Perspective Correction Failure on Optimal Orientation Images

**Bug ID:** BUG-20251124-001

### Executive Summary

The improved edge-based perspective correction approach exhibits catastrophic failure on images that already have optimal perspective orientation with minimal background area (<5%). The algorithm incorrectly applies perspective correction, resulting in severe distortion that completely removes document content from the output image. This failure mode is distinct from normal correction failures and indicates missing pre-validation checks for optimal orientation and background area thresholds.

**Affected Sample:** `drp.en_ko.in_house.selectstar_000008.jpg`

**Contrast Case (Working Correctly):** `drp.en_ko.in_house.selectstar_000006_mask.jpg` - demonstrates correct behavior when conditions are met

### 1. Visual Artifacts (What does the output look like?)

* **Distortion Type:** Severe distortion with complete loss of document content

* **Key Features:**
  - Resulting image shows no evidence of being related to the original image
  - Malformed mask output indicating incorrect edge detection
  - Edge lines placement is unclear and incorrect
  - Document content is completely lost in the transformation
  - Correction appears to target background regions instead of document

* **Comparison:**
  - **Improved Correction:** Catastrophic failure - no document visible in output
  - **Current Correction:** Worse than original but better than improved correction
  - **Current Correction Issue:** Highly influenced by folded document edge in bottom right corner, causing left-oriented shearing
  - **Excellent Results (selectstar_000006):** Works perfectly when conditions are met - demonstrates algorithm can function correctly

* **Background Area Behavior:**
  - Original image: <5% background area
  - Corrected image: **Increased** background area (opposite of expected)
  - This indicates correction is expanding background rather than document

### 2. Input Characteristics (What is unique about the source?)

* **ROI Coverage:** Document fills >95% of frame (minimal background)

* **Background Area:** Less than 5% of total image area (extreme case)

* **Geometry:**
  - Image already has optimal perspective orientation (no correction needed)
  - Minimal background elements except for folded edge in right corner
  - Document is already properly oriented and rectified

* **Edge Characteristics:**
  - Folded document edge located in bottom right corner
  - Triangular area in bottom right corner introduces perspective transformation artifacts
  - Current correction attempts to target background instead of document

### 3. Geometric/Data Analysis (The Math)

* **Mask Topology:**
  - Background area (black) to document object area (white) ratio is extreme (<5% background)
  - Correction results in **increased** background area (opposite of expected behavior)
  - Original image contained less background than the corrected result

* **Corner Detection:**
  - Edge detection fails to identify correct document boundaries
  - Edge lines placement is unclear and incorrect
  - Detected edges and resulting shape should have triggered failure criteria but did not

* **Transform Matrix:**
  - Perspective correction applied to already optimal orientation causes severe distortion
  - Transformation appears to target background regions instead of document
  - Result suggests incorrect corner ordering or edge detection failure

### 4. Hypothesis & Action Items

* **Theory:**
  The improved edge-based correction algorithm fails when:
  1. **Optimal Orientation:** Image already has correct perspective (no correction needed)
  2. **Minimal Background:** Background area <5% causes edge detection to fail
  3. **Edge Detection Confusion:** Algorithm cannot distinguish between document edges and background artifacts (folded edges)
  4. **Missing Pre-validation:** No check for optimal orientation before applying correction
  5. **Background Threshold Missing:** No validation to skip images with insufficient background area

* **Proposed Fix:**
  1. **Add Background Threshold (bg-threshold):**
     - Define experimental rule: skip images where background area < threshold (e.g., 5%)
     - Pre-validate before applying perspective correction
     - Return original image with reason logged if threshold not met

  2. **Add Optimal Orientation Detection:**
     - Check if image already has optimal perspective before correction
     - Measure skew angle and skip if <2° (already implemented in pre-validation)
     - Verify this check is working correctly for this case

  3. **Improve Edge Detection for Minimal Background Cases:**
     - Handle edge cases where background is <5%
     - Better handling of folded edges and artifacts
     - Consider using rembg mask approach for these cases instead

  4. **Enhanced Failure Criteria:**
     - Detect when detected edges and resulting shape indicate failure
     - Validate that correction doesn't increase background area
     - Check that output contains recognizable document content

  5. **Hybrid Approach:**
     - Use rembg mask-based approach for minimal background cases
     - Fallback to original image if both methods fail validation
     - Log failure reasons for analysis

* **Immediate Actions:**
  1. Implement background threshold validation (bg-threshold parameter)
  2. Verify optimal orientation detection is working correctly
  3. Add validation to prevent correction when background area would increase
  4. Test on selectstar_000008 and similar edge cases
  5. Compare with rembg mask-based approach for minimal background cases

### 5. Root Cause Analysis

**Primary Root Cause:** Missing pre-validation for optimal orientation and minimal background cases.

**Failure Chain:**
1. Image has optimal perspective orientation (no correction needed)
2. Background area is <5% (extreme case)
3. Edge detection algorithm attempts to find document boundaries
4. Algorithm confuses folded edge artifacts with document edges
5. Incorrect corner detection leads to malformed quadrilateral
6. Perspective transformation applied to already optimal image causes severe distortion
7. Result: Document content completely lost, background area increases

**Why Current Validation Failed:**
- Pre-validation checks for skew angle <2° should have caught this
- However, folded edge in bottom right corner may have influenced skew calculation
- Background threshold check does not exist
- No validation that correction should not increase background area

**Why Improved Correction Failed Worse Than Current:**
- Improved edge-based approach uses line fitting which may be more sensitive to edge artifacts
- Folded edge creates false edge lines that get fitted incorrectly
- RANSAC line fitting may have selected wrong edge points due to minimal background

---

## Related Resources

### Related Artifacts

* outputs/improved_edge_approach/drp.en_ko.in_house.selectstar_000006_mask.jpg
* outputs/improved_edge_approach/drp.en_ko.in_house.selectstar_000008_mask.jpg
* outputs/improved_edge_approach/drp.en_ko.in_house.selectstar_000008_comparison.jpg

### Related Assessments

* assessments/improved_edge_based_approach.md
* assessments/worst_performers_test_results.md

