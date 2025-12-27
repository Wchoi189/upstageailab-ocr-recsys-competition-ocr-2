---
ads_version: '1.0'
type: assessment
title: Mask-Only Edge Detection with Line Quality Heuristics
date: '2025-11-25'
experiment_id: 20251122_172313_perspective_correction
phase: phase_2
priority: high
status: complete
evidence_count: 8
created: '2025-11-25T00:00:00Z'
updated: '2025-12-27T16:16:42.205900'
tags:
- mask-only
- edge-detection
- heuristics
- perspective-correction
- validation
author: AI Agent
---
# Mask-Only Edge Detection with Line Quality Heuristics

## Overview

Complete rewrite of edge detection to work exclusively on rembg mask binary data (0/255), with critical line-quality validation heuristics to ensure fitted rectangles meet geometric constraints of real documents.

## Implementation

### Core Module: `mask_only_edge_detector.py`

**Principle:** All edge detection operates on binary mask data only. No image pixels are ever considered.

**Key Components:**

1. **Mask Processing:**
   - Binary normalization (0/255)
   - Morphological closing to fill gaps
   - Largest contour extraction

2. **Rectangle Fitting:**
   - Convex hull computation
   - Minimum area rectangle via `cv2.minAreaRect()`
   - Clockwise corner ordering

3. **Line Quality Heuristics (CRITICAL):**
   - **Edge Angle Validation:** Ensures all corners form approximately 90° angles (tolerance: 15°)
   - **Edge Length Validation:** Validates aspect ratio is within reasonable document bounds (0.1 to 10.0)
   - **Contour Alignment:** Cross-validates fitted rectangle against mask bounding box alignment (15% tolerance)
   - **Contour Structure:** Validates contour has sufficient structure (not too fragmented)

4. **Fallback Strategy:**
   - If validation fails → returns mask bounding box corners
   - If no contour found → returns mask bounding box
   - All decisions based purely on mask geometry

### Integration

Updated `test_improved_edge_approach.py` to use mask-only detector. Removed all image-based edge detection code.

## Line Quality Heuristics Details

### 1. Edge Angle Validation (`_validate_edge_angles`)

Validates that all four corners form approximately right angles, ensuring the fitted shape is rectangular.

- Computes edge vectors from ordered corners
- Calculates angle at each corner using dot product
- Rejects if any angle deviates >15° from 90°

### 2. Edge Length Validation (`_validate_edge_lengths`)

Validates edge lengths form reasonable document proportions.

- Computes average lengths of opposite edge pairs
- Checks aspect ratio (width/height or height/width)
- Rejects if aspect ratio outside 0.1 to 10.0 range

### 3. Contour Alignment (`_validate_contour_alignment`)

Cross-validates fitted rectangle against mask bounding box.

- Compares fitted rectangle bbox to mask foreground bbox
- Validates width/height ratios match within 15%
- Validates centroid alignment within 15% tolerance

### 4. Contour Structure (`_validate_contour_segments`)

Validates contour has sufficient structure (not too fragmented).

- Computes arc length of contour
- Calculates average segment length
- Rejects if average segment < 5 pixels

## Test Results

### Initial Test (10 worst performers)

- **Total tested:** 10
- **Improved approach attempts:** 10/10 (100%)
- **Improved approach success:** 1/10 (10% by area ratio)
- **Cases improved:** 2/10 (20% by area ratio)
- **Average document area retention:** ~99.7%
- **Average bbox retention:** ~99.5%
- **Skew deviation:** 0.00° (perfect rectangles)

### Key Observations

1. **High Document Retention:** Mask-only approach preserves ~99.7% of document pixels
2. **Perfect Rectangular Output:** All successful fits produce 0° skew (perfect rectangles)
3. **Canvas Shrinking:** Area ratio drops because rectangles tightly fit document (expected behavior)
4. **Validation Working:** Heuristics successfully filter invalid fits, falling back to mask bbox when needed

## Validation Failure Reasons

When validation fails, the system falls back to mask bounding box with specific reason codes:

- `contour_too_fragmented_using_bbox`: Contour structure insufficient
- `validation_failed_invalid_edge_angles_using_bbox`: Angles not approximately 90°
- `validation_failed_invalid_edge_proportions_using_bbox`: Aspect ratio out of bounds
- `validation_failed_poor_bbox_alignment_using_bbox`: Rectangle doesn't align with mask bbox

## Strengths

1. **Deterministic Results:** Mask-only approach produces consistent, reproducible corner fits
2. **High Document Retention:** Average document area retention ~99.7%
3. **No Broadcasting Errors:** Eliminated all shape mismatch issues
4. **Clean Architecture:** Single responsibility - mask geometry only
5. **Robust Validation:** Multiple heuristics ensure fitted rectangles meet geometric constraints

## Current Limitations

1. **Canvas Shrinking:** Mask-derived rectangles tightly fit document, causing area ratio to drop (expected, but metrics flag as "worsened")
2. **Strict Validation:** Some valid cases may be rejected and fall back to bbox
3. **No Edge Blending:** Falls back to mask bbox when validation fails, doesn't blend fitted rectangle with contour structure

## Next Steps

1. **Tune Validation Thresholds:** Adjust angle tolerance, aspect ratio bounds, and alignment tolerance based on broader testing
2. **Edge Blending:** Blend fitted rectangle with contour convex hull to recover padding without using image pixels
3. **Broader Testing:** Run full worst-performer suite (25+ samples) to validate heuristics across diverse cases
4. **Performance Analysis:** Compare document-aware metrics (mask retention, bbox retention, skew) between validated fits and bbox fallbacks

## Related Files

- `experiment-tracker/experiments/20251122_172313_perspective_correction/scripts/mask_only_edge_detector.py`
- `experiment-tracker/experiments/20251122_172313_perspective_correction/scripts/test_improved_edge_approach.py`
- `outputs/improved_edge_approach/mask_only_heuristics_test/test_results.json`
- `outputs/improved_edge_approach/mask_only_heuristics_test/*_improved_mask_fit.jpg` (debug visualizations)

## Related Assessments

- `incident_reports/20251124_1925-edge-line-selection-and-area-metric-defects.md`
- `assessments/improved_edge_based_approach.md`

