---
ads_version: "1.0"
type: "assessment"
experiment_id: "20251122_172313_perspective_correction"
status: "complete"
created: "2025-12-17T17:59:47Z"
updated: "2025-12-17T17:59:47Z"
tags: ['perspective-correction', 'failure-analysis', 'analysis']
phase: "phase_0"
priority: "medium"
evidence_count: 0
---

# Perspective Correction Failures - Analysis & Solutions

**Date**: 2025-11-22
**Status**: ⚠️ Critical Issues Identified
**Author**: AI Agent

## Problem Summary

The perspective correction pipeline is experiencing critical failures:

1. **False Positives**: Applying correction to images that don't need it
2. **Severe Cropping**: Detecting small regions (text blocks, artifacts) instead of full document
3. **Data Loss**: Output images are dramatically smaller than input (e.g., 398x1280 → 351x249)

### Example Failure

- **Input**: `drp.en_ko.in_house.selectstar_000008_01_rembg.jpg` (398x1280)
- **Output**: `drp.en_ko.in_house.selectstar_000008_02_perspective.jpg` (351x249)
- **Issue**: Original image didn't need correction, but algorithm detected corners from a small region and cropped most of the image

## Root Causes

### 1. Brittle Corner Detection

The current `DocumentDetector` uses edge-based contour detection which is prone to:
- Detecting small text blocks or artifacts instead of document boundaries
- Failing on images with complex backgrounds (even after rembg)
- No validation of detected corner area vs image area

### 2. No Pre-Correction Validation

The pipeline doesn't check:
- Whether correction is actually needed (skew angle)
- Whether detected corners are reasonable (area ratio, aspect ratio)
- Whether corners represent the full document vs a small region

### 3. No Post-Correction Validation

The pipeline doesn't validate:
- Output size vs input size (area ratio)
- Dimension shrinkage (width/height ratios)
- Content preservation (non-zero pixel ratio)

## Statistical Measurements That Catch Failures

### Pre-Correction Validation

1. **Corner Area Ratio**
   ```python
   corner_area = cv2.contourArea(corners)
   area_ratio = corner_area / image_area
   # Should be > 0.3 (30% of image)
   ```

2. **Aspect Ratio Match**
   ```python
   detected_aspect = width / height
   image_aspect = image_width / image_height
   # Should be within 50-200% of image aspect
   ```

3. **Skew Angle**
   ```python
   skew_angle = calculate_max_deviation_from_rectangular(corners)
   # If < 2°, correction probably not needed
   ```

### Post-Correction Validation

1. **Area Retention Ratio**
   ```python
   area_ratio = output_area / input_area
   # Should be > 0.5 (50% area retained)
   ```

2. **Dimension Ratios**
   ```python
   width_ratio = output_width / input_width
   height_ratio = output_height / input_height
   # Both should be > 0.5
   ```

3. **Content Preservation**
   ```python
   non_zero_ratio = non_zero_pixels / total_pixels
   # Should be > 0.1 (10% non-zero)
   ```

## Current Implementation Status

### Are We Using the Suggested Repository?

**No.** The code references `https://github.com/sraddhanjali/Automated-Perspective-Correction-for-Scanned-Documents-and-Cards` but:

1. **Actual Implementation**: Uses `ocr.data.datasets.preprocessing.detector.DocumentDetector` and `ocr.data.datasets.preprocessing.perspective.PerspectiveCorrector`
2. **Fallback Implementation**: `SimplePerspectiveCorrector` in `test_pipeline_rembg_perspective.py` is based on the reference but not actively used
3. **Issue**: The existing implementation is also unpredictable (as mentioned by user)

### Current Implementation Issues

1. **DocumentDetector**:
   - Uses Canny edge detection + contour approximation
   - No validation of detected corner area
   - Can detect small regions instead of full document
   - `min_area_ratio=0.1` is too low (allows 10% of image)

2. **PerspectiveCorrector**:
   - Applies transformation without validation
   - No checks for output size
   - No fallback if correction fails

## More Robust Open-Source Solutions

### 1. **ScanTailor** (C++/Qt)
- **Pros**: Battle-tested, handles complex documents, manual correction option
- **Cons**: Not Python, requires separate process
- **Use Case**: For critical document processing pipelines

### 2. **OpenCV Document Scanner** (Python)
- **Pros**: Well-documented, multiple detection methods
- **Cons**: Still requires careful parameter tuning
- **Implementation**: Enhanced version with validation (see below)

### 3. **Deep Learning Approaches**

#### a. **DocTR (Document Text Recognition)**
- **Pros**: Uses deep learning for document detection, more robust
- **Cons**: Requires GPU, heavier dependencies
- **Status**: Already available in codebase (`use_doctr_geometry`)

#### b. **U-Net-based Document Segmentation**
- **Pros**: Learns document boundaries, handles complex backgrounds
- **Cons**: Requires training data, model size

### 4. **Hybrid Approaches**

Combine multiple methods:
1. Try deep learning first (DocTR)
2. Fallback to edge-based with strict validation
3. Skip correction if validation fails

## Recommended Solution

### Immediate Fix: Add Validation

Created `scripts/test_perspective_robust.py` with:

1. **Pre-correction validation**:
   - Corner area ratio check (>30%)
   - Aspect ratio validation
   - Skew angle check (skip if <2°)

2. **Post-correction validation**:
   - Area retention check (>50%)
   - Dimension ratio checks
   - Content preservation check

3. **Graceful failure**:
   - Skip correction if not needed
   - Return original if validation fails
   - Log detailed failure reasons

### Long-term Solution: Use DocTR

The codebase already has DocTR integration (`use_doctr_geometry=True`). This should be:
1. **Enabled by default** for better robustness
2. **Used as primary method** with edge-based as fallback
3. **Validated** with the same statistical checks

### Alternative: Manual Correction Option

For critical documents:
1. Detect if correction is needed
2. If validation fails, flag for manual review
3. Provide UI for manual corner selection

## Implementation Recommendations

### 1. Update `test_perspective_on_rembg.py`

Add validation checks before and after correction:

```python
# Pre-validation
if not validate_corners(corners, image.shape):
    logger.info("Correction not needed or corners invalid")
    return original_image

# Apply correction
corrected = corrector.correct(image, corners)

# Post-validation
if not validate_correction_result(image, corrected):
    logger.warning("Correction result invalid, using original")
    return original_image
```

### 2. Enable DocTR by Default

```python
corrector = PerspectiveCorrector(
    use_doctr_geometry=True,  # Use deep learning
    ...
)
```

### 3. Increase `min_area_ratio`

```python
detector = DocumentDetector(
    min_area_ratio=0.3,  # Require 30% of image (was 0.1)
    ...
)
```

## Testing

Run the robust test script:

```bash
# Test on rembg outputs
python scripts/test_perspective_robust.py \
    --input-dir outputs/perspective_test \
    --output-dir outputs/perspective_robust_test \
    --num-samples 10
```

This will:
- Skip correction if not needed
- Validate before and after
- Log detailed failure reasons
- Preserve original if validation fails

## Conclusion

The current perspective correction is **too brittle** and needs:

1. ✅ **Validation** (implemented in `test_perspective_robust.py`)
2. ⚠️ **DocTR integration** (available but not default)
3. ⚠️ **Better corner detection** (needs tuning or replacement)
4. ⚠️ **Graceful failure** (return original instead of bad correction)

**Recommendation**: Use the robust validation script and enable DocTR for production use.

