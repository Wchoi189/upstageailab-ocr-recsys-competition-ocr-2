---
title: "Out-of-Bounds Polygon Coordinates in Training Dataset"
author: "ai-agent"
date: "2025-11-10"
timestamp: "2025-11-10 00:11 KST"
type: "bug_report"
category: "troubleshooting"
status: "open"
version: "1.0"
tags: ['bug', 'data-quality', 'polygon-validation', 'training-error']
bug_id: "BUG-20251110-001"
severity: "High"
---

# Bug Report: Out-of-Bounds Polygon Coordinates in Training Dataset

## Bug ID
BUG-20251110-001

## Summary
Training dataset contains 867 images (26.5%) with out-of-bounds Y coordinates exceeding image height, causing training errors.

## Environment
- **OS**: Not specified
- **Python Version**: Not specified
- **Dependencies**: Not specified
- **Browser**: Not specified

## Steps to Reproduce
1. Run data cleaning script on train/val datasets
2. Observe out-of-bounds Y coordinate errors
3. Training fails with shape mismatch errors

## Expected Behavior
All polygon coordinates should be within image bounds [0, width] x [0, height]

## Actual Behavior
867 train images and 96 validation images have Y coordinates exceeding image height of 960px

## Error Messages
```
out_of_bounds_y: Y coordinates out of bounds [0, 960]
```

## Screenshots/Logs
If applicable, include screenshots or relevant log entries.

## Impact
- **Severity**: High
- **Affected Users**: Who is affected
- **Workaround**: Any temporary workarounds

## Investigation

### Root Cause Analysis
- **Cause**: Polygon annotations contain Y coordinates exceeding image height, likely due to coordinate system mismatch or image resizing after annotation
- **Location**: data/datasets/jsons/train.json and val.json
- **Trigger**: Training pipeline validates polygons against image dimensions

### Related Issues
Related issue 1
Related issue 2

## Proposed Solution

### Fix Strategy
Remove problematic samples using data cleaning script, then add coordinate bounds validation to prevent future issues

### Implementation Plan
1. Investigate coordinate system
2. Remove problematic samples with --remove-bad
3. Add validation in dataset loader

### Testing Plan
1. Re-run data cleaning to verify fix
2. Test training with cleaned dataset
3. Verify no shape mismatch errors

## Status
- [ ] Confirmed
- [ ] Investigating
- [ ] Fix in progress
- [ ] Fixed
- [ ] Verified

## Assignee
Who is working on this bug.

## Priority
High/Medium/Low

---

*This bug report follows the project's standardized format for issue tracking.*

## Summary
Training dataset contains significant number of images with out-of-bounds polygon coordinates, specifically Y coordinates exceeding image height. This causes training errors and data pipeline failures.

**Affected Datasets:**
- Train: 867 images (26.5%) with out-of-bounds Y coordinates
- Validation: 96 images (23.8%) with out-of-bounds Y coordinates
- Test: 0 images (clean)

## Environment
- **Pipeline Version:** Data cleaning phase
- **Components:** Dataset validation, Polygon coordinate validation
- **Configuration:** Standard training configuration
- **Dataset Paths:**
  - Train: `data/datasets/images/train` with `data/datasets/jsons/train.json`
  - Validation: `data/datasets/images/val` with `data/datasets/jsons/val.json`
  - Test: `data/datasets/images/test` with `data/datasets/jsons/test.json`

## Steps to Reproduce
1. Run data cleaning script on training dataset:
   ```bash
   uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json
   ```
2. Observe annotation errors for out-of-bounds Y coordinates
3. Check validation dataset:
   ```bash
   uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/val --annotation-file data/datasets/jsons/val.json
   ```
4. Observe similar out-of-bounds Y coordinate errors

## Expected Behavior
All polygon coordinates should be within image bounds:
- X coordinates: [0, image_width]
- Y coordinates: [0, image_height]

Polygons should not exceed image dimensions.

## Actual Behavior
**Train Dataset:**
- 867 images (26.5%) have polygons with Y coordinates exceeding image height of 960px
- Example: `drp.en_ko.in_house.selectstar_000003.jpg` has Y coordinates out of bounds [0, 960]

**Validation Dataset:**
- 96 images (23.8%) have polygons with Y coordinates exceeding image height of 960px
- Example: `drp.en_ko.in_house.selectstar_000007.jpg` has Y coordinates out of bounds [0, 960]

**Error Pattern:**
```
ANNOTATION_ERRORS:
  - drp.en_ko.in_house.selectstar_000003.jpg
    • out_of_bounds_y: Y coordinates out of bounds [0, 960]
    • out_of_bounds_y: Y coordinates out of bounds [0, 960]
```

## Error Messages
```
⚠️  Issues by Category:
  annotation_errors: 867 (train), 96 (validation)

🔍 Sample Issues:
  ANNOTATION_ERRORS:
    - drp.en_ko.in_house.selectstar_000003.jpg
      • out_of_bounds_y: Y coordinates out of bounds [0, 960]
```

## Impact
- **Severity:** High
- **Affected Users:** Training pipeline, model training
- **Workaround:** Filter out problematic samples using data cleaning script with `--remove-bad` flag

**Training Impact:**
- Out-of-bounds coordinates cause shape mismatch errors during training
- Loss computation fails when polygons exceed image bounds
- Data pipeline crashes or produces invalid batches

## Investigation

### Root Cause Analysis
- **Cause:** Polygon annotations contain Y coordinates that exceed image height (960px). This suggests:
  1. Annotations may have been created for different image dimensions
  2. Images may have been resized after annotation
  3. Coordinate system mismatch (e.g., 1-indexed vs 0-indexed)
  4. Annotation errors during data collection

- **Location:**
  - Dataset: `data/datasets/jsons/train.json` and `data/datasets/jsons/val.json`
  - Validation: `scripts/data/clean_dataset.py` (validate_polygon method)
  - Training: `ocr/datasets/base.py` (polygon processing)

- **Trigger:**
  - Training pipeline loads polygons and validates against image dimensions
  - Out-of-bounds coordinates detected during polygon validation
  - Training fails when polygons exceed image bounds

### Related Issues
- Similar issues may exist with X coordinates (not detected in initial scan)
- Potential coordinate system inconsistencies across dataset
- May be related to EXIF orientation handling if images were rotated

## Proposed Solution

### Fix Strategy
1. **Immediate Fix:** Remove or fix out-of-bounds polygons
   - Option A: Remove problematic samples using data cleaning script
   - Option B: Clamp coordinates to image bounds (may lose annotation accuracy)
   - Option C: Investigate and fix root cause (coordinate system mismatch)

2. **Long-term Fix:**
   - Add validation during annotation creation
   - Implement coordinate bounds checking in dataset loader
   - Add pre-processing step to validate and fix coordinates

### Implementation Plan
1. **Phase 1: Investigation**
   - Analyze sample problematic images to understand coordinate system
   - Check if images were resized after annotation
   - Verify coordinate system (0-indexed vs 1-indexed)

2. **Phase 2: Immediate Fix**
   - Use data cleaning script to remove problematic samples:
     ```bash
     uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json --remove-bad --backup
     uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/val --annotation-file data/datasets/jsons/val.json --remove-bad --backup
     ```

3. **Phase 3: Prevention**
   - Add coordinate bounds validation in dataset loader
   - Add validation checks in annotation creation tools
   - Update data cleaning script to auto-fix coordinates (clamp to bounds)

### Testing Plan
1. **Validation:**
   - Re-run data cleaning script after fix
   - Verify no out-of-bounds coordinates remain
   - Confirm training pipeline runs without errors

2. **Training Test:**
   - Run training with cleaned dataset
   - Verify no shape mismatch errors
   - Confirm loss computation succeeds

3. **Regression Test:**
   - Ensure valid annotations are not affected
   - Verify image-polygon alignment is correct
   - Check that model training produces expected results
