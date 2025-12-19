---
ads_version: "1.0"
type: "assessment"
experiment_id: "20251217_024343_image_enhancements_implementation"
status: "complete"
created: "2025-12-17T17:59:48Z"
updated: "2025-12-17T17:59:48Z"
tags: ['image-enhancements']
phase: "phase_0"
priority: "medium"
evidence_count: 0
---

# Quick Reference: Enhancement Opportunities (REVISED)

**Based On**: Real-world observations showing tinted backgrounds + text slant as critical issues

---

## 🎯 Top 3 Priorities (REVISED - Highest ROI)

### 1. Background White-Balance Normalization ⭐⭐⭐ CRITICAL
**Why**: Observed in production - tinted backgrounds (cream, gray, yellow) cause severe inference degradation
**Method**: Gray-world or edge-based white balance → illumination correction
**Implementation**: 15-20 lines in OpenCV
**Expected Gain**: +5-10% accuracy on tinted background images (addresses observed critical issue)

```python
# Pseudocode - Gray-World White Balance
def white_balance_gray_world(image_bgr):
    avg_b, avg_g, avg_r = cv2.mean(image_bgr)[:3]
    avg_gray = (avg_b + avg_g + avg_r) / 3

    scale_b = avg_gray / avg_b if avg_b > 0 else 1.0
    scale_g = avg_gray / avg_g if avg_g > 0 else 1.0
    scale_r = avg_gray / avg_r if avg_r > 0 else 1.0

    result = image_bgr.astype(np.float32)
    result[:,:,0] *= scale_b
    result[:,:,1] *= scale_g
    result[:,:,2] *= scale_r
    return np.clip(result, 0, 255).astype(np.uint8)

# Pseudocode - Illumination Correction
def correct_illumination(image_gray, kernel_size=51):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)
    background = np.maximum(background, 1)
    corrected = (image_gray.astype(np.float32) / background.astype(np.float32)) * 128
    return np.clip(corrected, 0, 255).astype(np.uint8)
```

### 2. Text Deskewing/Rotation Correction ⭐⭐⭐ CRITICAL
**Why**: Observed in production - slanted text causes inference degradation; no deskewing after perspective correction
**Method**: Projection profile or Hough transform angle detection → rotation
**Implementation**: 30-40 lines in OpenCV
**Expected Gain**: +10-20% accuracy on slanted text images (addresses observed critical issue)

```python
# Pseudocode - Projection Profile Angle Detection
def detect_angle_projection(image_gray, angle_range=(-45, 45), angle_step=0.5):
    best_angle = 0
    max_variance = 0

    for angle in np.arange(angle_range[0], angle_range[1], angle_step):
        (h, w) = image_gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image_gray, M, (w, h))

        projection = np.sum(rotated, axis=1)
        variance = np.var(projection)

        if variance > max_variance:
            max_variance = variance
            best_angle = angle

    return best_angle

# Pseudocode - Rotation with Coordinate Update
def rotate_with_matrix(image, angle, existing_matrix=None):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Adjust translation to avoid cropping
    cos = np.abs(M_rot[0, 0])
    sin = np.abs(M_rot[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M_rot[0, 2] += (new_w / 2) - center[0]
    M_rot[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M_rot, (new_w, new_h),
                             flags=cv2.INTER_LANCZOS4)

    # Update coordinate transform matrix
    if existing_matrix is not None:
        M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
        existing_3x3 = np.vstack([existing_matrix, [0, 0, 1]])
        combined = M_rot_3x3 @ existing_3x3
        updated_matrix = combined[:2, :]
    else:
        updated_matrix = M_rot

    return rotated, updated_matrix
```

### 3. Adaptive Background Normalization ⭐⭐ HIGH (After #1 & #2)
**Why**: Handles non-uniform backgrounds (gradients, patterns) - complements white-balance
**Method**: Morphological background estimation + division normalization + CLAHE
**Implementation**: 20 lines in OpenCV
**Expected Gain**: +3-7% accuracy on non-uniform backgrounds (synergy with #1)

```python
# Pseudocode - Combined Adaptive Normalization
def normalize_background_adaptive(image_gray, kernel_size=51):
    # Estimate background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)
    background = np.maximum(background, 1)

    # Divide by background
    normalized = (image_gray.astype(np.float32) / background.astype(np.float32)) * 128
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Apply CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    return enhanced
```

---

## 📋 Revised Implementation Checklist

### Phase 1: Background White-Balance (Week 1) 🔴 CRITICAL
- [ ] **Day 1-2**: Implement gray-world white balance
  - [ ] Create `scripts/background_normalization.py`
  - [ ] Implement `white_balance_gray_world()`
  - [ ] Test on 5 cream/yellow-tinted images
  - [ ] Measure background color shift (target: → [250,250,250])

- [ ] **Day 3**: Implement edge-based white balance
  - [ ] Implement `white_balance_edge_sampling()`
  - [ ] Compare with gray-world method
  - [ ] Test on 10 tinted background images
  - [ ] Measure color consistency (std dev <10)

- [ ] **Day 4**: Add illumination correction
  - [ ] Implement `correct_illumination()`
  - [ ] Test on non-uniform lighting images
  - [ ] Tune kernel size parameter (31, 41, 51, 61)

- [ ] **Day 5**: Validation
  - [ ] Compare OCR accuracy: baseline vs. normalized
  - [ ] Document optimal parameters
  - [ ] Create visual examples (before/after)
  - [ ] **Success Criteria**: +5-10% accuracy on tinted images

### Phase 2: Text Deskewing (Week 2) 🔴 CRITICAL
- [ ] **Day 1-2**: Projection profile method
  - [ ] Create `scripts/text_deskewing.py`
  - [ ] Implement `detect_angle_projection()`
  - [ ] Test on synthetic rotated images (known angles)
  - [ ] Measure angle detection accuracy (target: <2° error)

- [ ] **Day 3**: Hough transform method
  - [ ] Implement `detect_angle_hough()`
  - [ ] Add robust angle voting
  - [ ] Test on real slanted receipts
  - [ ] Compare methods (projection vs. Hough)

- [ ] **Day 4**: Rotation implementation
  - [ ] Implement `rotate_with_matrix()`
  - [ ] Test Lanczos4 interpolation quality
  - [ ] Handle bounding box expansion (avoid cropping)

- [ ] **Day 5**: Coordinate tracking validation
  - [ ] Update perspective transform matrix chain
  - [ ] Test coordinate alignment with polygon overlays
  - [ ] Compare OCR accuracy: baseline vs. deskewed
  - [ ] **Success Criteria**: +10-20% accuracy on slanted images

### Phase 3: Integration (Week 3)
- [ ] **Day 1-2**: Pipeline integration
  - [ ] Add to `PreprocessingPipeline.process()`
  - [ ] New pipeline order:
    ```
    Input → Background Norm → Perspective → Deskew → Resize → Normalize
    ```
  - [ ] Add config flags: `enable_background_norm`, `enable_deskew`
  - [ ] Update `PreprocessingResult` metadata

- [ ] **Day 3**: Ablation study
  - [ ] Baseline (current pipeline)
  - [ ] +Background normalization only
  - [ ] +Deskewing only
  - [ ] +Both combined
  - [ ] Measure accuracy gains per combination

- [ ] **Day 4-5**: Final validation
  - [ ] Full test on 25 worst performers
  - [ ] Visual inspection of results
  - [ ] Performance profiling (<200ms target)
  - [ ] **Success Criteria**: +15-30% combined improvement

### Phase 4: Secondary Enhancements (Deferred)
- [ ] ~~Shadow removal~~ (may be addressed by illumination correction)
- [ ] ~~CLAHE~~ (integrated in adaptive background norm)
- [ ] ~~Bilateral denoise~~ (lower priority than critical issues)

---

## 🧪 Testing Commands

```bash
# Setup experiment context
cd experiment-tracker/experiments/20251217_024343_image_enhancements_implementation

# Test shadow removal only
python scripts/run_enhancement_test.py \
  --enhancement shadow_removal \
  --method tophat \
  --worst_case_dir /path/to/worst_performers \
  --n 5

# Test CLAHE only
python scripts/run_enhancement_test.py \
  --enhancement clahe \
  --clip_limit 2.0 \
  --tile_size 8 \
  --n 5

# Test denoising only
python scripts/run_enhancement_test.py \
  --enhancement denoise \
  --method bilateral \
  --n 5

# Test combination
python scripts/run_enhancement_test.py \
  --enhancement all \
  --n 25

# Compare OCR accuracy
python scripts/compare_ocr_accuracy.py \
  --baseline artifacts/baseline/ \
  --enhanced artifacts/enhanced_all/ \
  --metrics cer wer edit_distance
```

---

## 📊 Evaluation Metrics

### Image Quality Metrics
- **Brightness**: Mean pixel value (target: 100-200 for 8-bit)
- **Contrast**: Std dev of pixels (target: >50)
- **Blur**: Laplacian variance (target: >100)
- **PSNR**: Peak Signal-to-Noise Ratio vs. reference (if available)
- **SSIM**: Structural Similarity Index vs. reference (if available)

### OCR Metrics
- **CER**: Character Error Rate (lower is better)
- **WER**: Word Error Rate (lower is better)
- **Edit Distance**: Levenshtein distance from ground truth
- **Confidence**: Average model confidence per character

### Processing Metrics
- **Time**: Per-image processing time (target: <200ms)
- **Success Rate**: Percentage of images processed without error
- **Enhancement Triggered**: How often each enhancement is applied (if adaptive)

---

## 🎓 Best Practices from Perspective Correction

1. **Start Simple**: Top-hat before deep learning shadow removal
2. **Test Incrementally**: One enhancement at a time
3. **Measure Everything**: Accuracy before/after each change
4. **Keep Baselines**: Save unenhanced versions for comparison
5. **Visual Inspection**: Look at output images, not just metrics
6. **Edge Cases**: Test on worst performers first
7. **Avoid Over-Processing**: Good images should stay good
8. **Document Parameters**: Record all settings used
9. **Fallback Strategy**: Enhancement fails → use original
10. **Coordinate Tracking**: Maintain transform matrices

---

## 🔧 Parameter Tuning Guidelines

### Shadow Removal (Top-Hat)
- **Kernel Size**: Larger for bigger shadows (11, 15, 19, 23)
- **Kernel Shape**: Ellipse works better than rectangle
- **Trade-off**: Too large removes document structure

### CLAHE
- **Clip Limit**: Higher = more contrast (1.0-4.0)
  - 1.0: Subtle, natural
  - 2.0: Balanced (recommended start)
  - 4.0: Aggressive, may over-enhance
- **Tile Size**: Smaller = more local (4, 8, 16)
  - 4×4: Very local, may cause artifacts
  - 8×8: Balanced (recommended start)
  - 16×16: Global, smoother

### Bilateral Filter
- **d**: Diameter (5, 7, 9)
  - Larger = slower but smoother
- **Sigma Color**: Higher = more colors blended (50-150)
  - 75: Balanced (recommended start)
- **Sigma Space**: Higher = larger spatial influence (50-150)
  - 75: Balanced (recommended start)

---

## 🚨 Common Pitfalls

1. **Over-Enhancement**: Applying all enhancements to all images
   - **Fix**: Use quality metrics to apply selectively

2. **Wrong Order**: Enhancing before perspective correction
   - **Fix**: Shadow/brightness → Perspective → Contrast/denoise

3. **Breaking Coordinates**: Enhancement changes image size
   - **Fix**: Apply before resize, or update metadata

4. **Slow Processing**: Heavy filters on high-res images
   - **Fix**: Downscale → enhance → perspective → upscale

5. **Loss of Color**: Converting to grayscale too early
   - **Fix**: Enhance in color, convert later if needed

6. **Artifact Introduction**: Aggressive parameters
   - **Fix**: Start conservative, tune up gradually

---

## 📈 Expected Results

### Before Enhancements (Current State)
- Perspective correction: ✅ 100% success
- Shadow-affected images: ⚠️ Low OCR accuracy
- Low-contrast images: ⚠️ Missing characters
- Noisy images: ⚠️ Spurious detections

### After Shadow Removal
- Shadow-affected images: ✅ Improved readability
- Expected CER improvement: 15-30% relative
- Side effects: Minimal (non-shadowed images unchanged)

### After CLAHE
- Low-contrast images: ✅ Better character separation
- Expected CER improvement: 10-20% relative
- Side effects: May over-enhance good images (need quality gate)

### After Denoising
- Noisy images: ✅ Cleaner, fewer artifacts
- Expected CER improvement: 5-15% relative
- Side effects: Slight softening (acceptable with bilateral)

### Combined Pipeline
- Overall CER improvement: 20-40% on worst performers
- Processing time: <150ms per image (with optimization)
- Success rate: Maintain 100% (with fallbacks)

---

## 📝 Documentation Template

For each enhancement, document:

```markdown
## [Enhancement Name]

**Purpose**: What problem does it solve?

**Method**: Which algorithm/technique?

**Parameters**:
- param1: [value] - [description]
- param2: [value] - [description]

**Test Results**:
- Baseline CER: X.XX%
- Enhanced CER: Y.YY%
- Improvement: Z.ZZ% (relative)
- Processing time: XXms

**Visual Examples**:
- [Before image]
- [After image]
- [OCR comparison]

**Edge Cases**:
- When it helps: [description]
- When it hurts: [description]
- Recommended threshold: [value]

**Integration**:
```python
# Code snippet showing usage
```

**Related Issues**: [Links to images/experiments/bugs]
```

---

## 🔄 Iteration Strategy

1. **Implement**: Code the enhancement
2. **Unit Test**: Synthetic/controlled images
3. **Validate**: Worst performers (5 images)
4. **Measure**: OCR metrics before/after
5. **Tune**: Adjust parameters
6. **Document**: Record findings
7. **Integrate**: Add to pipeline
8. **Regression Test**: Ensure no breakage
9. **Scale Test**: Full dataset (25 images)
10. **Review**: Decide keep/discard/modify

---

## 💡 Quick Wins

**Easiest to Implement** (< 1 hour):
1. CLAHE (2 lines of code, huge impact)
2. Bilateral denoise (1 line, safe for text)

**Highest Impact** (best accuracy gain):
1. Shadow removal (addresses frequent issue)
2. CLAHE (handles contrast/brightness together)

**Best ROI** (impact / effort):
1. CLAHE (trivial to implement, major impact)
2. Shadow removal (moderate complexity, major impact on affected images)

**Start Here**: CLAHE → Shadow removal → Bilateral denoise
