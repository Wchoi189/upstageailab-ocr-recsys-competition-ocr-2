---
ads_version: '1.0'
type: report
title: Edge Line Selection and Area Metric Defects
date: 2025-11-24 19:25 (KST)
experiment_id: 20251122_172313_perspective_correction
phase: phase_1
priority: high
severity: high
status: complete
evidence_count: 2
created: '2025-11-24T19:25:00Z'
updated: '2025-12-27T16:16:42.618830'
tags:
- perspective
- metrics
- edge-detection
- bug
- analysis
author: AI Agent
metrics: []
baseline: none
comparison: neutral
---
## Defect Analysis: Edge Line Selection and Area Metric Defects

### 1. Visual Artifacts (What does the output look like?)

* **Distortion Types Observed:**
  - Metric flags `✗ worsened` even though perspective visibly improves (e.g., `selectstar_000006`, `000097`, `000045`) because the canvas shrinks or expands.
  - Edge overlays (`*_improved_edges.jpg`) show >4 lines, with segments wrapping around corners or tracking interior folds (`selectstar_000011`, `000085`).
  - Some lines lock onto background rectangles above the actual page (`selectstar_000146`), leaving the document region untouched.

* **Key Features:**
  - Improved outputs often contain straighter text and orthogonal margins, yet the current metric only counts pixel area, not document content.
  - When the mask lacks background on a side, the fitter snaps to interior features, producing 90° turns and multiple colored lines.
  - IDs that experienced broadcasting errors (`000023`, `000040`, `000112`, `000141`) produced no improved images even though masks were valid.

* **Comparison:**
  - Baseline retains a large canvas (high area ratio) while leaving skewed content.
  - Improved branch can crop tighter (lower canvas area) yet deliver superior usability—current scoring does not reflect that.

### 2. Input Characteristics (What is unique about the source?)

* **ROI Coverage:** Several samples (e.g., `selectstar_000042`) are already orthorectified, occupying ~100% of the canvas with little side background.
* **Background Distribution:** Many masks show strong top/bottom fill but missing left/right fill; others have gaps on one or more sides.
* **Geometry:** Tall, narrow documents produce highly unbalanced edge-point counts, aggravating the `(rows,) vs (cols,)` array issue during visualization.

### 3. Geometric/Data Analysis (The Math)

* **Area-Ratio Definition:** Currently `(corrected_width × corrected_height) / (original_width × original_height)`—pure raster area, not document mask area. As a result, cropping blank space or padding the canvas dramatically swings the metric without reflecting document quality.
* **Mask Topology:** Masks themselves are accurate silhouettes, but lack a continuous background collar. When a side touches the image border, the fitter anchors lines to interior folds or detached background blobs.
* **Corner / Line Detection:** We keep every fitted line per edge group. Without enforcing “exactly four lines,” the quadrilateral can wrap around concave folds or ignore the actual page altogether.
* **Failure Logs:** Broadcasting errors still show `operands could not be broadcast together with shapes (3297,) (289,)`, confirming that some edge-group arrays remain mismatched despite earlier guards.

### 4. Hypothesis & Action Items

* **Theory:**
  1. Area ratio is unreliable because it measures canvas pixels, not document preservation.
  2. Edge-selection logic is under-constrained—multiple lines per side and gaps in background produce arbitrary quadrilaterals.
  3. Missing background collars allow RANSAC to grab interior features, causing wraps and failures.

* **Proposed Fix:**
  1. Introduce document-aware metrics: warp the mask alongside the image, compute document-area retention, bounding-box preservation, and geometric distortion scores (edge lengths, skew).
  2. Reinforce masks by adding a synthetic thin background halo so each side presents a detectable boundary, even when rembg removed adjacent background.
  3. Limit line selection to one per side: choose the line closest to the expected margin with correct orientation, merge near-collinear duplicates, and trim lines at intersection points.
  4. Anchor lines using the outermost band of rembg background (top/bottom/left/right) instead of arbitrary Canny edges whenever possible.
  5. Eliminate remaining broadcasting errors by standardizing `(N, 2)` point arrays for visualization and logging shapes when mismatches occur.

---

## Related Resources

### Related Artifacts

* outputs/improved_edge_runs/20251124_164230_improved_edge/test_results.json

### Related Assessments

* incident_reports/20251124_1347-improved-edge-based-approach-performance-regression-and-technical-bugs.md
* assessments/improved_edge_based_approach.md

