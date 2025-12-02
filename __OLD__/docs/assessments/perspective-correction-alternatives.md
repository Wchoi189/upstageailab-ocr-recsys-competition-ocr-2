# Perspective Correction Tools - Alternatives Assessment

**Date**: 2025-11-22
**Status**: Comprehensive Comparison

## Executive Summary

After evaluating ScanTailor (blocked by dependencies/GUI requirements), this document assesses alternative tools for perspective correction with CLI/Python support suitable for batch processing.

## Current Implementation Status

### What We Have

1. **DocTR (Document Text Recognition)**
   - ✅ Already integrated in codebase
   - ✅ Deep learning-based document detection
   - ✅ More robust than edge-based methods
   - ⚠️ Requires GPU for optimal performance
   - ⚠️ Heavier dependencies

2. **OpenCV-based Document Detection**
   - ✅ Currently implemented (`DocumentDetector`)
   - ✅ Lightweight, fast
   - ❌ 40% failure rate (detects small regions instead of full document)
   - ❌ Requires careful parameter tuning

3. **Perspective Correction Pipeline**
   - ✅ `PerspectiveCorrector` class exists
   - ✅ Supports both DocTR and edge-based methods
   - ❌ Validation missing (causes bad outputs)
   - ❌ No graceful fallback

## Alternative Tools Comparison

### 1. ImageMagick (CLI)

**Description**: Command-line image manipulation suite with perspective distortion correction.

**Pros**:
- ✅ Native CLI support (perfect for batch processing)
- ✅ Well-documented and widely used
- ✅ No GUI dependencies
- ✅ Fast processing
- ✅ Supports batch operations

**Cons**:
- ❌ Requires manual corner point specification
- ❌ No automatic document detection
- ❌ Need to integrate with detection step

**Usage Example**:
```bash
# Requires 4 corner points (x1,y1 x2,y2 x3,y3 x4,y4)
convert input.jpg -distort perspective \
  "0,0 0,0  100,0 100,0  100,100 100,100  0,100 0,100" \
  output.jpg
```

**Integration Effort**: Medium
- Need to combine with corner detection (OpenCV/DocTR)
- Generate ImageMagick commands programmatically

**Verdict**: ⭐⭐⭐⭐ Good for batch processing if combined with detection

---

### 2. OpenCV (Python) - Enhanced Version

**Description**: Improve existing OpenCV-based implementation with better validation.

**Pros**:
- ✅ Already in codebase
- ✅ Python-native (easy integration)
- ✅ Fast processing
- ✅ Good documentation
- ✅ Can combine multiple detection methods

**Cons**:
- ❌ Current implementation has 40% failure rate
- ❌ Requires parameter tuning
- ❌ Edge detection can be unreliable

**Improvements Needed**:
1. Better corner validation (area ratio >30%)
2. Multiple detection methods (Canny, adaptive, morphological)
3. Post-correction validation
4. Graceful fallback to original

**Integration Effort**: Low (enhance existing code)

**Verdict**: ⭐⭐⭐⭐ Best immediate option (improve what we have)

---

### 3. DocTR (Already Available)

**Description**: Deep learning document detection and geometry analysis.

**Pros**:
- ✅ Already integrated
- ✅ More robust than edge-based
- ✅ Handles complex backgrounds
- ✅ Better corner detection accuracy

**Cons**:
- ⚠️ Slower than OpenCV (GPU recommended)
- ⚠️ Heavier dependencies
- ⚠️ May still fail on difficult images

**Current Status**: Available but not default

**Recommendation**: Enable by default, use as primary method

**Verdict**: ⭐⭐⭐⭐⭐ Best long-term solution

---

### 4. imutils (Python Library)

**Description**: Collection of OpenCV convenience functions including document scanner.

**Pros**:
- ✅ Python library (easy integration)
- ✅ Built on OpenCV (familiar)
- ✅ Includes `four_point_transform()` function
- ✅ Well-documented

**Cons**:
- ❌ Still requires corner detection (same challenges)
- ❌ Not significantly different from OpenCV approach

**Usage Example**:
```python
from imutils.perspective import four_point_transform
corrected = four_point_transform(image, corners)
```

**Integration Effort**: Low (wrapper around OpenCV)

**Verdict**: ⭐⭐⭐ Convenient but doesn't solve detection issues

---

### 5. scikit-image (Python)

**Description**: Scientific image processing library with geometric transformations.

**Pros**:
- ✅ Python-native
- ✅ Good for research/experimentation
- ✅ Multiple transformation methods

**Cons**:
- ❌ No built-in document detection
- ❌ More complex API
- ❌ Slower than OpenCV for simple operations

**Verdict**: ⭐⭐ Overkill for this use case

---

### 6. Hugin (CLI)

**Description**: Panorama creation tool with perspective correction capabilities.

**Pros**:
- ✅ CLI support
- ✅ Advanced correction algorithms
- ✅ Handles complex distortions

**Cons**:
- ❌ Primarily for panoramas (overkill)
- ❌ Complex setup
- ❌ Requires control point specification

**Verdict**: ⭐⭐ Not suitable for document correction

---

### 7. GIMP (via Scripting)

**Description**: GNU Image Manipulation Program with Python scripting (Script-Fu).

**Pros**:
- ✅ Powerful image processing
- ✅ Scriptable
- ✅ Perspective correction tools

**Cons**:
- ❌ Requires GIMP installation
- ❌ Complex scripting API
- ❌ Slower startup time
- ❌ Overkill for batch processing

**Verdict**: ⭐ Not practical for automation

---

### 8. Perspec (CLI Tool)

**Description**: Scriptable desktop application for perspective correction.

**Pros**:
- ✅ CLI support
- ✅ Designed for perspective correction
- ✅ Cross-platform (macOS, Linux)

**Cons**:
- ❌ Requires manual corner marking
- ❌ Less mature than other tools
- ❌ Limited documentation

**Verdict**: ⭐⭐ Limited automation support

---

## Recommended Solutions

### Short-term (Immediate)

**Option A: Enhance Existing OpenCV Implementation** ⭐⭐⭐⭐⭐

**Actions**:
1. Add robust validation (already in `test_perspective_robust.py`)
2. Increase `min_area_ratio` from 0.1 to 0.3
3. Add pre-correction validation (corner area, aspect ratio, skew)
4. Add post-correction validation (area retention, dimensions)
5. Implement graceful fallback (return original if validation fails)

**Effort**: Low (1-2 hours)
**Impact**: High (reduces failure rate significantly)

**Code Location**: `ocr/datasets/preprocessing/detector.py`, `perspective.py`

---

### Medium-term (1-2 weeks)

**Option B: Enable DocTR as Primary Method** ⭐⭐⭐⭐⭐

**Actions**:
1. Set `use_doctr_geometry=True` as default
2. Use OpenCV as fallback when DocTR fails
3. Add validation for both methods
4. Performance optimization (GPU support, caching)

**Effort**: Medium (already integrated, needs configuration)
**Impact**: High (better accuracy, handles complex cases)

**Code Location**: `ocr/datasets/preprocessing/perspective.py`

---

### Long-term (Alternative)

**Option C: ImageMagick + Detection Pipeline** ⭐⭐⭐⭐

**Actions**:
1. Use DocTR/OpenCV for corner detection
2. Generate ImageMagick commands programmatically
3. Execute via subprocess
4. Handle output conversion

**Effort**: Medium (new integration)
**Impact**: Medium (CLI-based, good for batch processing)

**Pros**: True CLI, no Python dependencies for processing
**Cons**: Additional subprocess overhead, need detection step anyway

---

## Implementation Plan

### Phase 1: Immediate Improvements (Recommended)

```python
# Enhanced DocumentDetector with validation
detector = DocumentDetector(
    min_area_ratio=0.3,  # Increased from 0.1
    use_adaptive=True,
    use_fallback=True,
    validate_corners=True,  # New: pre-validation
)

# Enhanced PerspectiveCorrector with validation
corrector = PerspectiveCorrector(
    use_doctr_geometry=True,  # Enable by default
    validate_output=True,  # New: post-validation
    fallback_to_original=True,  # New: graceful failure
)
```

**Files to Update**:
- `ocr/datasets/preprocessing/detector.py`
- `ocr/datasets/preprocessing/perspective.py`
- Use validation logic from `scripts/test_perspective_robust.py`

---

### Phase 2: DocTR Integration

```python
# Make DocTR default
corrector = PerspectiveCorrector(
    use_doctr_geometry=True,  # Default
    fallback_to_opencv=True,  # Fallback if DocTR fails
)
```

---

### Phase 3: ImageMagick Integration (Optional)

```python
def correct_with_imagemagick(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Use ImageMagick for perspective correction."""
    # Convert corners to ImageMagick format
    # Generate command
    # Execute and read result
    pass
```

---

## Comparison Matrix

| Tool | CLI Support | Python API | Auto Detection | Validation | Integration Effort | Recommendation |
|------|------------|------------|----------------|------------|-------------------|----------------|
| **OpenCV (Enhanced)** | ❌ | ✅ | ✅ | ⚠️ (add) | Low | ⭐⭐⭐⭐⭐ Best immediate |
| **DocTR** | ❌ | ✅ | ✅ | ⚠️ (add) | Low | ⭐⭐⭐⭐⭐ Best long-term |
| **ImageMagick** | ✅ | ⚠️ (subprocess) | ❌ | ❌ | Medium | ⭐⭐⭐⭐ Good for batch |
| **imutils** | ❌ | ✅ | ❌ | ❌ | Low | ⭐⭐⭐ Convenient wrapper |
| **ScanTailor** | ❌ | ❌ | ✅ | ✅ | High (blocked) | ❌ Not feasible |
| **scikit-image** | ❌ | ✅ | ❌ | ❌ | Medium | ⭐⭐ Overkill |
| **Hugin** | ✅ | ❌ | ❌ | ❌ | High | ⭐⭐ Wrong tool |
| **GIMP** | ⚠️ | ⚠️ | ❌ | ❌ | High | ⭐ Not practical |

---

## Conclusion

**Best Approach**: Enhance existing OpenCV implementation + enable DocTR

1. **Immediate**: Improve validation in current OpenCV-based code
2. **Short-term**: Enable DocTR as primary method with OpenCV fallback
3. **Long-term**: Consider ImageMagick for pure CLI batch processing if needed

**Why Not ScanTailor**:
- Original: Qt4 dependency (not available)
- Advanced: GUI-only, project file format unknown
- Integration complexity too high

**Why Current Approach is Better**:
- Already integrated
- Python-native (easier debugging)
- Can combine multiple methods
- Full control over validation
- No external dependencies beyond what we have

## Next Steps

1. ✅ **Implement validation** (use `test_perspective_robust.py` as reference)
2. ✅ **Enable DocTR by default** in production pipeline
3. ⚠️ **Test enhanced implementation** on sample dataset
4. ⚠️ **Monitor failure rates** and adjust parameters

## References

- Current Implementation: `ocr/datasets/preprocessing/detector.py`, `perspective.py`
- Robust Test Script: `scripts/test_perspective_robust.py`
- DocTR Integration: `scripts/test_perspective_doctr_rembg.py`
- Failure Analysis: `docs/assessments/perspective_correction_failures_analysis.md`

