---
title: "Interim Assessment: Run 02 - Coordinate Inversion Fix"
author: "ai-agent"
date: "2025-11-29 00:42 (KST)"
status: "draft"
kind: "run_log"
template_id: "run-log-negative-result"
---

# Interim Assessment: Run 02 - Coordinate Inversion Fix

**Date:** 2025-11-29
**Run ID:** 20251129_003040
**Hypothesis:** Fixing the coordinate inversion bug (Top/Bottom swapped) and refactoring binning to use horizontal/vertical classification will eliminate invalid_edge_angles failures and produce geometrically valid quadrilaterals.

## 1. Executive Summary

**Result:** 0/25 Success (Top-line metric unchanged, but critical geometric bug fixed).
**Verdict:** Hypothesis **validated**. All invalid_edge_angles failures eliminated. Algorithm now produces geometrically valid quadrilaterals for all 25 samples. Remaining failures are quality validation issues, not fundamental geometric problems.

## 2. Failure Mode Shift

*Did the errors change, even if the result didn't?*

| Metric | Previous Run (Run 01 - Angle Bucketing) | Current Run (Run 02 - Coordinate Fix) | Delta |
| :--- | :--- | :--- | :--- |
| **Success Rate** | 0% | 0% | – |
| **Primary Error** | `line_quality_fallback_edge_support_edge_support_ratio_linearity_rmse_corner_sharpness` | `line_quality_fallback_edge_support_edge_support_ratio_linearity_rmse_corner_sharpness` | Same |
| **Secondary Error** | **`validation_failed_invalid_edge_angles_using_bbox`** (5 samples) | **Eliminated** | **Fixed** |

**Observation:**
The critical coordinate inversion bug has been resolved. All 5 samples that previously failed with `invalid_edge_angles` (078, 101, 140, 145, 153) now pass geometric validation. The algorithm consistently produces valid quadrilaterals, but they still fail downstream quality heuristics (edge support, linearity, corner sharpness). This represents a significant improvement: we've moved from a geometric failure to a quality validation problem, which is more addressable.

## 3. Key Samples for Inspection

### A. Fixed Geometric Failure – Now Passing Validation

- **ID:** `drp.en_ko.in_house.selectstar_000140`
- **Behavior:** Previously failed `_validate_edge_angles` due to coordinate inversion (Top/Bottom swapped). Now passes geometric validation but fails line quality checks.
- **Implication:** The horizontal/vertical classification (`abs(dx) > abs(dy)`) correctly handles coordinate systems and aspect ratios. Bin assignments are now balanced (left: 6 points vs previous left: 0).

### B. Status Quo Failure – Still Failing Quality Checks

- **ID:** `drp.en_ko.in_house.selectstar_000006`
- **Behavior:** Still fails `line_quality_fallback` due to poor edge support/linearity metrics.
- **Implication:** Geometric validity is necessary but not sufficient. Edge support thresholds or regression weighting may need adjustment.

### C. Partial Improvement – Blend Cases

- **ID:** `drp.en_ko.in_house.selectstar_000024`, `000119`, `000145`, `000155`, `000159`, `000232`
- **Behavior:** Classified as `line_quality_partial_blend` (6 samples, up from 3 in Run 01).
- **Implication:** More samples are achieving partial quality thresholds, suggesting the geometric fix improves downstream metrics. May indicate progress toward full acceptance.

## 4. Next Steps

1. **Analyze Quality Metrics:** Inspect edge support/linearity/corner sharpness values for samples that pass geometric validation but fail quality checks. Determine if thresholds are too strict or if regression needs improvement.

2. **Weighted Regression:** Experiment with weighting `cv2.fitLine` inputs by segment length to reduce noise from short segments and improve line quality.

3. **Epsilon Tuning:** Evaluate if increasing `approxPolyDP` epsilon (to allow more segments) improves edge support metrics without introducing geometric instability.

4. **Threshold Relaxation:** Consider adaptive thresholds based on image quality or mask characteristics, rather than fixed values.

