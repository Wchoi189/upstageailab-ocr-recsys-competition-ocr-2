---
ads_version: "1.0"
title: "001 Dominant Edge Extension Failure"
date: "2025-12-06 18:09 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---





## Summary
The "Dominant Edge Extension" algorithm, designed to handle folded or torn document corners by extending infinite lines from dominant segments, failed to achieve any success (0/25) on the worst-case dataset. The failure stems from geometric instability where small angular errors in short segments are magnified when extended, and topological inconsistency where `approxPolyDP` rarely produces exactly 4 dominant segments.

## Environment
- **Experiment ID**: `20251128_005231_perspective_correction`
- **Script**: `mask_only_edge_detector.py`
- **Dataset**: 25 worst-performing images from training set (masks).

## Steps to Reproduce
1.  Run `fit_mask_rectangle` with `use_dominant_extension=True`.
2.  Input a mask with a folded corner (e.g., `selectstar_000138`) or a curved edge.
3.  Observe either:
    *   **Fallback to Bbox**: Filtering logic finds != 4 segments.
    *   **Validation Failure**: Reconstructed quad fails 0.50 support threshold due to drift.

## Expected vs Actual Behavior
- **Expected**: The algorithm filters out the short "fold" segment, extends the remaining 4 sides, and reconstructs the original corner.
- **Actual**:
    *   **Topology Trap**: `approxPolyDP` produces 3, 5, or 6 segments, leading to immediate fallback.
    *   **Lever Arm Effect**: Short or noisy segments have slight angular deviations. Extending them by ~1000px causes the intersection point to drift significantly from the true corner, resulting in a shape that fails overlap validation.

## Root Cause Analysis
1.  **Topology Trap**: Relying on a fixed segment count (4) from `approxPolyDP` is brittle. Organic masks have curves and tears that result in variable segment counts.
2.  **Lever Arm Effect**: Linear regression on short, noisy segments is unstable. A 1-degree error on a 50px segment becomes a ~20px error when extended to 1000px.

## Resolution Plan
Pivot to **Angle-Based Bucketing**:
1.  Accept any number of segments from `approxPolyDP`.
2.  Classify *all* segments into 4 bins (Top/Bottom/Left/Right) based on angle.
3.  Perform regression on the *entire cloud of points* for each bin to average out noise and stabilize the angle.
4.  Intersect the 4 consensus lines.

## Prevention
- Avoid algorithms that rely on exact topological counts (e.g., "must find 4 lines").
- Prefer statistical aggregation (RANSAC, Regression on bins) over single-segment extrapolation.

