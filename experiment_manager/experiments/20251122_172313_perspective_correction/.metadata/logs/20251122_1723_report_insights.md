---
ads_version: '1.0'
type: assessment
experiment_id: 20251122_172313_perspective_correction
status: complete
created: '2025-12-17T17:59:48Z'
updated: '2025-12-27T16:16:42.548611'
tags:
- perspective-correction
phase: phase_0
priority: medium
evidence_count: 0
title: 20251122 1723 Report Insights
---
### Insight [2025-11-22T19:17:25.091683] (general)
It works

### Insight [2025-11-22T19:27:23.837393] (execution)
Started execution of main

### Insight [2025-11-23T13:30:00.000000] (algorithm)
**Rembg mask provides perfect object segmentation** - The rembg model achieves 100% reliable object detection for this dataset. The alpha channel (mask) contains all the information needed to identify document boundaries. This is more reliable than cv2's brittle edge detection.

### Insight [2025-11-23T13:45:00.000000] (algorithm)
**Using rembg mask for corner detection** - Instead of relying on cv2's edge detection (which fails on 27% of cases), we can extract the mask from rembg output and use it directly to find document boundaries. This approach achieved 44% success rate on worst performers, recovering 22 cases that previously failed.

### Insight [2025-11-23T13:50:00.000000] (analysis)
**Worst performers analysis** - Identified 50 worst performing images based on area loss metrics. Key patterns: 72% of failures are in 40-50% area loss range, suggesting corner detection is finding regions close to valid but slightly too small. Many failures show aspect ratio mismatches, indicating cv2 detects wrong regions (text blocks instead of full document).

### Insight [2025-11-23T14:00:00.000000] (algorithm)
**Multi-point edge detection vs 4 extreme points** - Current approach uses only 4 extreme points (topmost, rightmost, bottommost, leftmost), which may be noisy or outliers. Better approach: extract all edge points from mask, group by edge, fit lines using RANSAC, then find corner intersections. This uses actual edge shape rather than just extremes.

### Insight [2025-11-23T14:15:00.000000] (implementation)
**Edge-based line fitting challenges** - Initial implementation of multi-point edge detection and line fitting is performing worse than 4-point approach. Likely issues: edge grouping logic may be incorrect, RANSAC parameters need tuning, or line intersection logic has bugs. Need to refine algorithm parameters and test on more cases.

### Insight [2025-11-23T14:30:00.000000] (methodology)
**Visual inspection reveals corner placement issues** - Numerically, area ratios may be correct, but visual inspection of worst performers shows poor corner placement. The corners detected don't align well with actual document boundaries. This suggests the problem is in corner detection accuracy, not just validation thresholds.

### Insight [2025-11-23T14:35:00.000000] (process)
**Rembg mask edges are the ground truth** - Since rembg provides perfect object segmentation, the mask edges represent the true document boundaries. We should leverage this by: 1) extracting edge points from mask, 2) grouping points by edge direction, 3) fitting lines to encompass the object, 4) using line intersections for corners. This is fundamentally different from cv2's approach of detecting edges in the processed image.
