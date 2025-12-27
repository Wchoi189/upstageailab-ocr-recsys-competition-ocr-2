---
ads_version: '1.0'
type: assessment
experiment_id: 20251217_024343_image_enhancements_implementation
status: complete
created: '2025-12-17T17:59:48Z'
updated: '2025-12-17T17:59:48Z'
tags:
- image-enhancements
phase: phase_0
priority: medium
evidence_count: 0
---
# Revised Priority Plan: Background Normalization & Text Deskewing

**Date**: 2025-12-17
**Based On**: Real-world observations showing 100% cropping success but critical background/slant issues

---

## 🎯 Critical Issues Identified (Priority Order)

### 1. **Internal Background Color Variation** 🔴 HIGH IMPACT
**Problem**: Document backgrounds show wide range of tinted white shades (cream, gray, yellowish) causing inference degradation

**Impact**:
- Model trained on clean white backgrounds sees unfamiliar color distributions
- Tinted backgrounds reduce text-background contrast
- Color inconsistency confuses feature extraction

**Office Lens Steps**: Steps 7-8, 11-12
- Step 7: White-balance correction
- Step 8: Illumination correction
- Step 11: Adaptive thresholding
- Step 12: Background normalization

### 2. **Text Slant/Skew** 🔴 HIGH IMPACT
**Problem**: No correction for slanted/rotated text, causing inference degradation

**Impact**:
- Text line detection fails on angled text
- Character recognition optimized for upright text
- Sequence models expect horizontal text flow

**Office Lens Steps**: Step 4
- Step 4: Deskewing/rotation correction

### 3. **Shadow Artifacts** 🟡 MEDIUM IMPACT (Deferred)
**Current State**: Less critical given background issues are dominant

**Office Lens Steps**: Step 13
- Step 13: Shadow removal (revisit after background normalization)

---

## 🚀 Incremental Implementation Plan (4 Approaches)

### **Approach 1: Background White-Balance Normalization** ⭐ HIGHEST ROI
**Target**: Eliminate tinted white variation, normalize to consistent white background

**Implementation Strategy**:
```python
def normalize_background_white(image_bgr: np.ndarray) -> np.ndarray:
    """
    Normalize tinted document backgrounds to consistent white.

    Strategy:
    1. Detect background color (assume document edges/corners are background)
    2. Compute white-balance correction transform
    3. Apply global correction to shift background toward neutral white
    4. Preserve text contrast during correction
    """
    # Step 1: Estimate background color from image periphery
    # Step 2: Compute correction factors (per channel)
    # Step 3: Apply correction with saturation protection
    # Step 4: Validate text regions maintain contrast
```

**Office Lens Mapping**: Steps 7-8
- Step 7: White-balance correction (auto white point)
- Step 8: Illumination correction (flatten lighting gradients)

**Testing Approach**:
1. Collect 10 images with tinted backgrounds (cream, gray, yellow)
2. Apply background normalization
3. Compare OCR accuracy before/after
4. Measure background color variance (should decrease significantly)

**Expected Gain**: +5-10% accuracy on tinted background images

**Implementation Phases**:
- **Phase 1a**: Simple gray-world white balance (1-2 hours)
- **Phase 1b**: Edge-based background estimation (2-3 hours)
- **Phase 1c**: Illumination flattening with morphological operations (2-3 hours)

---

### **Approach 2: Text Rotation/Deskew Correction** ⭐ HIGHEST ROI
**Target**: Detect and correct text slant/rotation before OCR

**Implementation Strategy**:
```python
def deskew_text(image_gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect text orientation and rotate to horizontal alignment.

    Strategy:
    1. Detect text orientation via projection profile analysis
    2. Or use Hough line transform on edge-detected text
    3. Compute rotation angle
    4. Rotate image with high-quality interpolation
    5. Crop to valid region (avoid black borders)
    """
    # Step 1: Compute projection profiles (horizontal/vertical)
    # Step 2: Detect dominant text angle (-45° to +45°)
    # Step 3: Rotate with Lanczos4 interpolation
    # Step 4: Update coordinate transform matrix
```

**Office Lens Mapping**: Step 4
- Step 4: Deskewing/rotation correction (text alignment)

**Testing Approach**:
1. Collect 10 images with slanted text (5°, 10°, 15°, 20° angles)
2. Apply deskewing
3. Measure rotation angle accuracy (should be <1° error)
4. Compare OCR accuracy before/after
5. Validate coordinate transforms still work

**Expected Gain**: +10-20% accuracy on slanted text images

**Implementation Phases**:
- **Phase 2a**: Projection profile-based angle detection (2-3 hours)
- **Phase 2b**: Hough transform-based angle detection (3-4 hours)
- **Phase 2c**: Hybrid approach with confidence scoring (2-3 hours)

---

### **Approach 3: Adaptive Background Normalization** ⭐⭐ HIGH ROI (After Approach 1)
**Target**: Handle non-uniform backgrounds (gradients, patterns, textures)

**Implementation Strategy**:
```python
def normalize_background_adaptive(image_gray: np.ndarray) -> np.ndarray:
    """
    Normalize non-uniform backgrounds using adaptive techniques.

    Strategy:
    1. Estimate local background intensity (morphological operations)
    2. Divide image by estimated background (illumination correction)
    3. Rescale to full dynamic range
    4. Apply adaptive histogram equalization (CLAHE)
    """
    # Step 1: Morphological opening with large kernel (estimate background)
    # Step 2: Divide original by background (flatten illumination)
    # Step 3: Normalize intensity range
    # Step 4: CLAHE for local contrast
```

**Office Lens Mapping**: Steps 8, 10, 11
- Step 8: Illumination correction (local adaptation)
- Step 10: Local contrast enhancement (CLAHE)
- Step 11: Adaptive thresholding (if binarization needed)

**Testing Approach**:
1. Test on images with gradient backgrounds
2. Test on patterned/textured backgrounds
3. Measure background uniformity (std dev should decrease)
4. Compare OCR accuracy before/after

**Expected Gain**: +3-7% accuracy on non-uniform backgrounds

**Implementation Phases**:
- **Phase 3a**: Morphological background estimation (2-3 hours)
- **Phase 3b**: Division normalization (1-2 hours)
- **Phase 3c**: CLAHE integration (1 hour)

---

### **Approach 4: Combined Pipeline Integration** ⭐⭐⭐ SYNERGY GAINS
**Target**: Combine background normalization + deskewing for maximum impact

**Implementation Strategy**:
```python
# New pipeline order (based on your observations):
Input Image (BGR)
      ↓
[NEW] Background White-Balance Normalization  ← Approach 1
      ↓
[EXISTING] Perspective Correction (100% success - keep as-is)
      ↓
[NEW] Text Deskew/Rotation Correction         ← Approach 2
      ↓
[NEW] Adaptive Background Normalization       ← Approach 3 (optional)
      ↓
[EXISTING] Resize & Pad (LongestMaxSize)
      ↓
[EXISTING] Normalization (ToTensor + Normalize)
      ↓
Model Input
```

**Office Lens Mapping**: Steps 1-4, 7-8, 10-11
- Steps 1-3: Document detection + perspective (✅ already done)
- Step 4: Deskewing (new)
- Steps 7-8: White-balance + illumination (new)
- Step 10: Local contrast (optional)

**Testing Approach**:
1. Run full pipeline on 25 worst performers
2. Ablation study:
   - Baseline (current)
   - +Background normalization only
   - +Deskewing only
   - +Both combined
3. Measure cumulative accuracy gains

**Expected Gain**: +15-30% accuracy on worst performers (combined effect)

---

## 📋 Concrete Implementation Roadmap

### **Week 1: Background Normalization (Approach 1)**

#### Day 1-2: Simple White-Balance
- [ ] Create `scripts/background_normalization.py`
- [ ] Implement gray-world white balance
- [ ] Test on 5 cream/yellow-tinted images
- [ ] Measure color shift (target: background → neutral white)

#### Day 3-4: Edge-Based Background Estimation
- [ ] Implement corner/edge sampling for background detection
- [ ] Add per-channel correction factors
- [ ] Test on 10 tinted background images
- [ ] Compare OCR accuracy baseline vs. normalized

#### Day 5: Illumination Correction
- [ ] Implement morphological background estimation
- [ ] Add division normalization (flatten gradients)
- [ ] Test on non-uniform lighting images
- [ ] Document optimal parameters (kernel sizes)

**Deliverables**:
- `background_normalization.py` with 3 methods
- Test results JSON comparing accuracy
- Parameter tuning documentation
- Visual before/after examples

---

### **Week 2: Text Deskewing (Approach 2)**

#### Day 1-2: Projection Profile Method
- [ ] Create `scripts/text_deskewing.py`
- [ ] Implement projection profile angle detection
- [ ] Test on synthetic rotated images (known angles)
- [ ] Measure angle detection accuracy

#### Day 3-4: Hough Transform Method
- [ ] Implement Hough line-based angle detection
- [ ] Add robust angle voting mechanism
- [ ] Test on real slanted receipts
- [ ] Compare with projection profile method

#### Day 5: Rotation & Coordinate Updates
- [ ] Implement high-quality rotation (Lanczos4)
- [ ] Update perspective transform matrix tracking
- [ ] Test coordinate alignment with polygon overlays
- [ ] Validate on 10 slanted images

**Deliverables**:
- `text_deskewing.py` with 2 detection methods
- Angle detection accuracy report
- OCR accuracy improvement metrics
- Coordinate transform validation

---

### **Week 3: Integration & Validation (Approach 4)**

#### Day 1-2: Pipeline Integration
- [ ] Integrate background normalization into `PreprocessingPipeline`
- [ ] Integrate deskewing into pipeline
- [ ] Add configuration flags (`enable_background_norm`, `enable_deskew`)
- [ ] Update `PreprocessingResult` metadata

#### Day 3: Ablation Study
- [ ] Run baseline (current pipeline) on 25 worst performers
- [ ] Run +background_norm only
- [ ] Run +deskew only
- [ ] Run +both combined
- [ ] Analyze accuracy gains per combination

#### Day 4-5: Final Validation & Documentation
- [ ] Full pipeline test on all worst performers
- [ ] Visual inspection of results
- [ ] Performance profiling (time per image)
- [ ] Update experiment documentation
- [ ] Create integration guide for main pipeline

**Deliverables**:
- Integrated preprocessing pipeline
- Ablation study results (4 configurations)
- Performance benchmarks
- Integration documentation

---

## 🎯 Success Metrics (Aligned with Your Observations)

### Primary Metrics
1. **Background Color Consistency**
   - Measure: Std dev of background pixel values
   - Target: <10 (currently variable)

2. **Text Alignment Accuracy**
   - Measure: Detected angle error vs. ground truth
   - Target: <1° error

3. **OCR Character Accuracy**
   - Baseline: Current CER on worst performers
   - Target: -15-30% relative CER (lower is better)

### Secondary Metrics
4. **Processing Time**: <200ms per image (maintain real-time)
5. **Success Rate**: 100% (no pipeline failures)
6. **Coordinate Accuracy**: Polygon overlays remain aligned

---

## 🔧 Implementation Details

### Background Normalization Methods

#### Method 1: Gray-World White Balance (Simplest)
```python
def white_balance_gray_world(image_bgr):
    """Assume average color should be neutral gray."""
    avg_b, avg_g, avg_r = cv2.mean(image_bgr)[:3]
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Compute per-channel scaling factors
    scale_b = avg_gray / avg_b if avg_b > 0 else 1.0
    scale_g = avg_gray / avg_g if avg_g > 0 else 1.0
    scale_r = avg_gray / avg_r if avg_r > 0 else 1.0

    # Apply scaling with saturation clipping
    result = image_bgr.astype(np.float32)
    result[:,:,0] *= scale_b
    result[:,:,1] *= scale_g
    result[:,:,2] *= scale_r
    return np.clip(result, 0, 255).astype(np.uint8)
```

#### Method 2: Edge-Based Background Estimation (More Robust)
```python
def white_balance_edge_sampling(image_bgr, border_ratio=0.05):
    """Sample background color from image edges (assume edges are background)."""
    h, w = image_bgr.shape[:2]
    border_w = int(w * border_ratio)
    border_h = int(h * border_ratio)

    # Sample top, bottom, left, right borders
    top = image_bgr[:border_h, :]
    bottom = image_bgr[-border_h:, :]
    left = image_bgr[:, :border_w]
    right = image_bgr[:, -border_w:]

    # Compute median background color (robust to outliers)
    background_samples = np.vstack([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ])
    bg_color = np.median(background_samples, axis=0)

    # Target: neutral white [255, 255, 255]
    scale_factors = 255.0 / bg_color

    result = image_bgr.astype(np.float32)
    result[:,:,0] *= scale_factors[0]
    result[:,:,1] *= scale_factors[1]
    result[:,:,2] *= scale_factors[2]
    return np.clip(result, 0, 255).astype(np.uint8)
```

#### Method 3: Illumination Correction (Handles Gradients)
```python
def correct_illumination(image_gray, kernel_size=51):
    """Divide by estimated background to flatten illumination gradients."""
    # Estimate background using large morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)

    # Avoid division by zero
    background = np.maximum(background, 1)

    # Divide and rescale
    corrected = (image_gray.astype(np.float32) / background.astype(np.float32)) * 128
    return np.clip(corrected, 0, 255).astype(np.uint8)
```

### Text Deskewing Methods

#### Method 1: Projection Profile (Fast)
```python
def detect_angle_projection(image_gray, angle_range=(-45, 45), angle_step=0.5):
    """Detect text angle using projection profile analysis."""
    best_angle = 0
    max_variance = 0

    for angle in np.arange(angle_range[0], angle_range[1], angle_step):
        # Rotate image
        (h, w) = image_gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image_gray, M, (w, h))

        # Compute horizontal projection (sum across rows)
        projection = np.sum(rotated, axis=1)

        # Higher variance = better alignment (text lines separated)
        variance = np.var(projection)
        if variance > max_variance:
            max_variance = variance
            best_angle = angle

    return best_angle
```

#### Method 2: Hough Lines (More Robust)
```python
def detect_angle_hough(image_gray, edge_threshold=50):
    """Detect text angle using Hough line transform on edges."""
    # Edge detection (text edges)
    edges = cv2.Canny(image_gray, edge_threshold, edge_threshold * 2)

    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return 0.0

    # Extract angles, convert to degrees
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta) - 90  # Convert to rotation angle
        # Normalize to [-45, 45]
        while angle_deg > 45:
            angle_deg -= 90
        while angle_deg < -45:
            angle_deg += 90
        angles.append(angle_deg)

    # Return median angle (robust to outliers)
    return np.median(angles)
```

#### Rotation with Coordinate Update
```python
def rotate_and_update_matrix(image, angle, existing_matrix=None):
    """Rotate image and update coordinate transformation matrix."""
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # Rotation matrix
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding box to avoid cropping
    cos = np.abs(M_rot[0, 0])
    sin = np.abs(M_rot[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust translation
    M_rot[0, 2] += (new_w / 2) - center[0]
    M_rot[1, 2] += (new_h / 2) - center[1]

    # Rotate image
    rotated = cv2.warpAffine(image, M_rot, (new_w, new_h),
                             flags=cv2.INTER_LANCZOS4,
                             borderValue=(255, 255, 255))

    # Update transformation matrix (chain with existing)
    if existing_matrix is not None:
        # Convert to 3x3 homogeneous
        M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
        existing_3x3 = np.vstack([existing_matrix, [0, 0, 1]])
        combined = M_rot_3x3 @ existing_3x3
        updated_matrix = combined[:2, :]
    else:
        updated_matrix = M_rot

    return rotated, updated_matrix, angle
```

---

## 🧪 Testing Strategy (Aligned with Your Issues)

### Test Set Composition
1. **Tinted Backgrounds** (10 images)
   - Cream/beige documents
   - Gray-tinted paper
   - Yellowish aging
   - Blue/pink tints

2. **Slanted Text** (10 images)
   - 5° rotation (mild)
   - 10° rotation (moderate)
   - 15-20° rotation (severe)
   - Mixed slant directions

3. **Combined Issues** (5 images)
   - Tinted + slanted
   - Gradient + slanted
   - Worst performers from previous experiments

### Validation Protocol
```bash
# Test background normalization only
python scripts/test_background_norm.py \
  --method gray_world \
  --input_dir artifacts/tinted_backgrounds/ \
  --output_dir artifacts/normalized/

# Test deskewing only
python scripts/test_deskewing.py \
  --method projection \
  --input_dir artifacts/slanted_text/ \
  --output_dir artifacts/deskewed/

# Test combined pipeline
python scripts/test_combined_pipeline.py \
  --enable_background_norm \
  --enable_deskew \
  --input_dir artifacts/worst_performers/ \
  --output_dir artifacts/enhanced/

# Compare OCR accuracy
python scripts/compare_ocr_accuracy.py \
  --baseline artifacts/baseline_ocr/ \
  --enhanced artifacts/enhanced_ocr/ \
  --metrics cer wer confidence
```

---

## 📊 Expected Results (Based on Your Observations)

### Addressing Background Color Variation
**Before**:
- Background colors: [cream(180,175,165), gray(200,200,200), yellow(240,235,180), ...]
- Background std dev: 45+ (high variation)
- OCR accuracy on tinted: ~70%

**After Background Normalization**:
- Background colors: [white(250,250,250), white(248,248,248), white(252,252,252), ...]
- Background std dev: <10 (low variation)
- OCR accuracy on tinted: ~85-90% (+15-20 points)

### Addressing Text Slant
**Before**:
- Text angles: [0°, 5°, -8°, 12°, -15°, ...] (undetected)
- OCR accuracy on slanted: ~65%
- Line detection failures: frequent

**After Deskewing**:
- Text angles: all within ±1° of horizontal
- OCR accuracy on slanted: ~85-90% (+20-25 points)
- Line detection failures: rare

### Combined Effect
**Worst Performers Before**:
- Average CER: 35-40% (bad)
- Issues: tinted backgrounds + slanted text

**Worst Performers After**:
- Average CER: 15-20% (acceptable)
- Relative improvement: 40-60% reduction in error rate

---

## 🎓 Lessons from Your Observations

### Key Insight 1: Cropping Works, Enhancement Doesn't
✅ **Your finding**: Perspective correction (cropping) is 100% successful
❌ **Your finding**: Internal document quality issues remain

**Implication**: Don't reinvent geometric correction. Focus on pixel-level quality enhancement.

### Key Insight 2: Background > Shadow
✅ **Your finding**: Background color variation is critical
⚠️ **Previous assumption**: Shadow removal was highest priority

**Implication**: Revise priority order. Background normalization before shadow removal.

### Key Insight 3: Text Geometry > Text Quality
✅ **Your finding**: Text slant causes inference degradation
⚠️ **Previous assumption**: Noise/blur reduction was next priority

**Implication**: Deskewing is co-equal with background normalization. Both are foundational.

---

## 🚦 Go/No-Go Decision Points

### After Week 1 (Background Normalization)
**Measure**:
- Background color std dev reduction: >50%?
- OCR accuracy gain on tinted images: >10 points?
- Processing time: <50ms added?

**Decision**:
- ✅ All YES → Proceed to Week 2
- ⚠️ Mixed → Tune parameters, iterate
- ❌ No improvement → Investigate root cause (maybe not a background issue?)

### After Week 2 (Deskewing)
**Measure**:
- Angle detection accuracy: <2° error?
- OCR accuracy gain on slanted images: >15 points?
- Coordinate alignment: Still working?

**Decision**:
- ✅ All YES → Proceed to Week 3
- ⚠️ Mixed → Refine angle detection
- ❌ Breaks coordinates → Rework matrix tracking

### After Week 3 (Integration)
**Measure**:
- Combined accuracy gain: >20 points on worst performers?
- Processing time: <200ms total?
- Pipeline robustness: No new failures?

**Decision**:
- ✅ All YES → Deploy to main pipeline
- ⚠️ Mixed → Cherry-pick successful components
- ❌ Regression → Rollback, reassess

---

## 🔗 Office Lens Roadmap Alignment

Your prioritized steps from the 21-step pipeline:

**✅ Already Done (100% Success)**:
- Step 1: Document boundary detection
- Step 2: Page contour extraction
- Step 3: Perspective correction
- Step 5: Content cropping

**🚀 Implementing Now (Weeks 1-3)**:
- Step 4: **Deskewing** ← Week 2
- Step 7: **White-balance correction** ← Week 1, Phase 1a-1b
- Step 8: **Illumination correction** ← Week 1, Phase 1c
- Step 10: Local contrast (CLAHE) ← Week 3 (if time)

**📋 Deferred (Future)**:
- Steps 11-14: Adaptive thresholding, background norm, shadow removal, noise suppression
- Steps 15-17: Sharpening, smoothing, morphological optimization
- Steps 18-21: Layout cleanup, multi-mode rendering

**Priority Rationale**: Address your observed critical issues first (background variation + text slant) before moving to nice-to-haves.

---

## 🎯 Summary: 4 Incremental Approaches

| Approach | Target | Office Lens Steps | Expected Gain | Effort | Priority |
|----------|--------|-------------------|---------------|--------|----------|
| **1. Background White-Balance** | Tinted whites → neutral white | 7-8 | +5-10% | 6-8 hrs | ⭐⭐⭐ |
| **2. Text Deskewing** | Slanted text → horizontal | 4 | +10-20% | 7-9 hrs | ⭐⭐⭐ |
| **3. Adaptive Background Norm** | Non-uniform → flat | 8, 10, 11 | +3-7% | 5-6 hrs | ⭐⭐ |
| **4. Combined Pipeline** | Synergy of 1+2+3 | 4, 7-8, 10-11 | +15-30% | 2-3 hrs | ⭐⭐⭐ |

**Recommended Sequence**: 1 → 2 → 4 → 3 (get high-impact items first, then optimize)

---

## 📝 Next Steps for You

1. **Review this plan** - Does it align with your observations?
2. **Collect test images** - 10 tinted backgrounds, 10 slanted text images
3. **Run baseline OCR** - Capture current accuracy on test set
4. **Start Week 1, Day 1** - Implement gray-world white balance
5. **Report results** - Share findings to calibrate next steps

This plan directly addresses your two critical issues (background variation + text slant) with concrete, testable implementations. Ready to start Week 1?
