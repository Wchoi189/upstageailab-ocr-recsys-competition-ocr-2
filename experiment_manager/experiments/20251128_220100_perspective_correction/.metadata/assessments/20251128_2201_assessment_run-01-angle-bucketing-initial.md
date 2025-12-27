---
ads_version: '1.0'
type: assessment
title: 'Interim Assessment: Run 01 - Angle Bucketing Initial'
author: ai-agent
date: 2025-11-28 22:34 (KST)
experiment_id: 20251128_220100_perspective_correction
phase: phase_1
priority: medium
status: complete
evidence_count: 2
created: '2025-11-28T22:34:00Z'
updated: '2025-12-27T16:16:43.396642'
tags:
- angle-bucketing
- run-log
- negative-result
kind: run_log
template_id: run-log-negative-result
---
# Interim Assessment: Run 01 - Angle Bucketing Initial

**Date:** 2025-11-28
**Run ID:** 20251128_223447
**Hypothesis:** Angle-based bucketing of hull segments will stabilize the four fitted edges by averaging noisy lines, improving corner intersections over the prior simple regression.

## 1. Executive Summary

**Result:** 0/25 Success (No change in top-line metric).
**Verdict:** Hypothesis **not yet validated**. The fitter now consistently derives four lines, but downstream validation still rejects most outputs.

## 2. Failure Mode Shift

| Metric | Previous Run (Simple Regression) | Current Run (Angle Bucketing) | Delta |
| :--- | :--- | :--- | :--- |
| **Success Rate** | 0% | 0% | – |
| **Primary Error** | `line_quality_fallback_edge_support_edge_support_ratio_linearity_rmse_corner_sharpness` | `line_quality_fallback_edge_support_edge_support_ratio_linearity_rmse_corner_sharpness` | Same |
| **Secondary Error** | `dominant_extension_failed_using_bbox` | **`validation_failed_invalid_edge_angles_using_bbox`** | **New** |

**Observation:** The fitter transitions from “could not find a quad” to “quad found but angles invalid.” Samples 78/101/140/145/153 now fail `_validate_edge_angles`, indicating the averaged lines intersect but violate near-orthogonality. This suggests bin composition/borrowing is working, yet geometric consistency needs tuning.

## 3. Key Samples for Inspection

### A. Status Quo Failure – No Material Change

- **ID:** `drp.en_ko.in_house.selectstar_000006`
- **Behavior:** Still trips the line-quality fallback due to poor edge support.
- **Implication:** Angle bucketing does not yet improve support on long, low-contrast edges; upstream contour filtering remains dominant.

### B. New Failure – Invalid Angles

- **ID:** `drp.en_ko.in_house.selectstar_000078`
- **Behavior:** Passes quad fitting but fails `_validate_edge_angles` and reverts to bbox.
- **Implication:** Bucketing likely mixed diagonal segments (e.g., Right borrowing from Top), yielding diamond-shaped intersections. Need to inspect per-bin membership.

### C. Blend Regression – Partial Success but Still Rejected

- **ID:** `drp.en_ko.in_house.selectstar_000024`
- **Behavior:** Classified as `line_quality_partial_blend`; shows modest improvement but still below thresholds.
- **Implication:** Demonstrates that averaging multiple segments can align with the mask, yet support metrics remain below guardrails—may require reweighting bins by segment length.

## 4. Next Steps

1. **Inspect Binning Diagnostics:** Dump per-bin membership for samples 078/140/145 to confirm whether bins borrowed the correct neighbors and whether point counts are sufficient for fitLine.
2. **Adjust Angular Windows:** Narrow quadrant thresholds (e.g., ±30°) or bias borrowing toward adjacent compass directions to avoid mixing diagonals.
3. **Weighted Regression:** Experiment with weighting cv2.fitLine inputs by segment length to reduce the influence of short noisy edges when bins borrow.
4. **Visualization Overlay:** Extend `visualize_mask_fit` to color-code points by bin so invalid-angle cases can be debugged visually.

