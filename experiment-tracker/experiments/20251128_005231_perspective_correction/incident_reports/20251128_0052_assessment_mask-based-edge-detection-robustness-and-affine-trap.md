---
ads_version: "1.0"
type: "assessment"
title: "Mask-Based Edge Detection: Robustness & Geometric Limitations"
date: "2025-11-28 01:10 (KST)"
experiment_id: "20251128_005231_perspective_correction"
phase: "phase_2"
priority: "medium"
severity: "medium"
status: "complete"
evidence_count: 3
created: "2025-11-28T01:10:00Z"
updated: "2025-11-28T01:10:00Z"
tags: ["mask-only", "edge-support", "approx-poly-dp", "affine-rigidity"]
author: "AI Agent"
---

## Executive Summary

The mask-only rectangle fitter now demonstrates strong resilience against catastrophic failures thanks to largest connected component (LCC) filtering, deterministic mask geometry, and the current line-quality heuristics. However, the system still exhibits **affine rigidity**: it extrapolates perfectly parallel edges even when the document clearly requires a projective (trapezoidal) fit, leading to **virtual corners** with zero edge support. This report captures both the robustness wins and the remaining geometric deficiencies to anchor future tuning work.

## 1. Success Analysis — Robustness & Generalization

### Case: `selectstar_000155` — Irregular Polygon Handling
- **Observation:** Previously-fatal disjoint artifacts (finger occlusion) no longer derail the fit.
- **Mechanism:** LCC filtering isolates the true document silhouette, and the heuristics compress the wavy outline into a stable bounding quadrilateral.
- **Outcome:** The fitter maintains full mask retention without homography instability, proving high robustness to outliers and non-Manhattan edges.

### Case: `selectstar_000247` — Trapezoidal Approximation
- **Observation:** The algorithm handled long, slightly non-parallel edges on a tall document.
- **Mechanism:** Orientation is inferred from the mask’s overall hull rather than pixel gradients, so mild perspective skew does not confuse the fitter.
- **Outcome:** General orientation remains correct despite weak parallelism, establishing a reliable baseline for more advanced projective handling.

## 2. Defect Analysis — The “Affine Trap”

### Defect A: Uncorrected Yaw / Rotation (`selectstar_000119`)
- **Issue:** Yaw-induced foreshortening compresses the right edge, but the fitter still enforces a rectangle.
- **Failure Mode:** The algorithm bounds outermost points instead of honoring the four local maxima of the convex hull, so the resulting homography cannot undo the projective distortion.
- **Ideal Behavior:** Detect hull vertices directly (quad) and feed them into the perspective corrector, enabling a true trapezoid-to-rectangle transform.

### Defect B: “Virtual Corner” Hallucination (`selectstar_000216`)
- **Issue:** Perspective foreshortening makes the top edge physically shorter than the bottom edge.
- **Failure Mode:** The fitter projects the side edges upward into empty background to create a virtual rectangular corner, yielding near-zero edge support along the real top edge.
- **Root Cause:** Rectangularity takes precedence over edge evidence, so ghost lines form when the mask demands a converging-line solution.

## 3. Root Cause & Recommended Actions

1. **Pivot Geometry Source:** Retire `cv2.minAreaRect` and treat **`cv2.approxPolyDP` over the mask hull** as the authoritative quadrilateral (already implemented as the new baseline).
2. **Iterative Epsilon Strategy:** Adjust `epsilon` until the hull simplifies to exactly four vertices; this preserves perspective foreshortening (narrow tops, wide bottoms) and eliminates virtual corners.
3. **Edge-Support Enforcement:** Maintain strict per-edge support thresholds—reject or snap any edge whose coverage or point ratio <0.80 to avoid ghost lines.
4. **Future Work:** Investigate corner snapping toward dense contour clusters and adaptive thresholding so legitimately trapezoidal documents can pass validation without falling back to the mask bbox.

With these mitigations recorded, future experiments can iterate on snapping and threshold tuning while referencing this incident report whenever regressions threaten the newly-stable baseline.



