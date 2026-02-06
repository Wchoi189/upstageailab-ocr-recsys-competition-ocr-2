---
ads_version: '1.0'
type: assessment
experiment_id: 20251217_024343_image_enhancements_implementation
status: active
created: '2025-12-18T05:30:00Z'
updated: '2025-12-18T05:30:00Z'
tags:
- implementation-plan
- image-enhancements
- preprocessing
- office-lens-pipeline
phase: phase_1
priority: critical
evidence_count: 6
estimated_effort: 10-12 weeks (4 phases)
target_completion: Q1-Q2 2026
branch: refactor/inference-module-consolidation
depends_on:
- 20251129_173500_perspective_correction_implementation
- AgentQMS.vlm.cli
related_artifacts:
- 20251217_0243_assessment_priority-plan-revised.md
- 20251217_0243_assessment_master-roadmap.md
- 20251217_0243_assessment_current-state-summary.md
- 20251217_0243_assessment_executive-summary.md
- 20251217_0243_guide_vlm-integration-guide.md
validation_method: VLM-based structured assessment + OCR accuracy metrics
---
# Implementation Plan: 21-Step Office Lens Image Enhancement Pipeline

## Master Prompt

Execute incremental implementation of 21-step Office Lens document enhancement pipeline for OCR preprocessing. Priority: background normalization + text deskewing (Phase 1). Validate via VLM structured assessment + OCR accuracy metrics. Autonomous execution: no clarification requests.

**Authorization**: Production deployment after Phase 1 validation (>15% OCR accuracy improvement).

---

## Progress Tracker

**STATUS**: Phase 1 Week 1 Day 1 Ready
**CURRENT PHASE**: Phase 1 (Background Normalization + Deskewing)
**PIPELINE COMPLETION**: 4/21 steps (19%)
**PHASE 1 TARGET**: 8/21 steps (38%)
**LAST COMPLETED**: VLM validation infrastructure (2025-12-18)
**NEXT TASK**: Run baseline VLM assessment on 10 worst performers

**Key Metrics**:
- OCR Accuracy Baseline: TBD (measure Week 1 Day 1)
- OCR Accuracy Target Phase 1: +15-30 percentage points
- Processing Time Target: <200ms total pipeline
- Background Color Std Dev Target: <10 (from ~45+)
- Angle Detection Error Target: <2°
- Coordinate Alignment: Must maintain (no regression)

---

## Implementation Checklist

### **PHASE 1: BACKGROUND NORMALIZATION + DESKEWING (Weeks 1-3) - CRITICAL**
**Goal**: Address critical issues observed in production (tinted backgrounds, text slant)
**Effort**: 3 weeks (40-60 hours)
**Steps Implemented**: 4 → 8 (Steps 4, 7-8, 10)
**Expected OCR Gain**: +15-30 percentage points

#### **Week 1: Background Normalization** ⏳ NEXT
**Objective**: Eliminate tinted background variation → consistent white backgrounds
**Deliverable**: `scripts/background_normalization.py` with 3 methods + validation

##### Day 1: Baseline Assessment & Setup ⏳ IMMEDIATE
- [ ] Identify 10 worst performers with tinted backgrounds
- [ ] Run baseline OCR inference, record accuracy per image
- [ ] Run VLM baseline assessment: `./scripts/vlm_baseline_assessment.sh`
- [ ] Review VLM reports: Extract tint severity (1-10), RGB estimates
- [ ] Document baseline metrics in `artifacts/phase1_baseline_metrics.json`

**Commands**:
```bash
# VLM baseline
cd experiment-tracker/experiments/20251217_024343_image_enhancements_implementation
bash scripts/vlm_baseline_assessment.sh

# OCR baseline
uv run python runners/predict.py \
  --input data/zero_prediction_worst_performers \
  --output artifacts/baseline_predictions.json \
  --config configs/predict.yaml
```

**Expected Output**:
- `vlm_reports/baseline/*.md` (10 quality assessment reports)
- `artifacts/phase1_baseline_metrics.json` (OCR accuracy, tint severity)

##### Day 2-3: Implement Gray-World + Edge-Based White Balance
- [ ] Create `scripts/background_normalization.py`
- [ ] Implement Method 1: Gray-world white balance
  - Compute global RGB mean: `r_avg, g_avg, b_avg`
  - Calculate scale factors: `r_scale = gray_avg/r_avg`, etc.
  - Apply: `img[:,:,0] *= b_scale`, `img[:,:,1] *= g_scale`, `img[:,:,2] *= r_scale`
- [ ] Implement Method 2: Edge-based background estimation
  - Detect background pixels: Canny edges → dilate → invert mask
  - Compute background mean RGB (ignore text/edges)
  - Scale channels to target white (255, 255, 255)
- [ ] Add CLI: `--method [gray-world|edge-based]`, `--output-dir`
- [ ] Test on 5 tinted images: Visual inspection + VLM validation

**Code Structure**:
```python
# scripts/background_normalization.py
class BackgroundNormalizer:
    def normalize_gray_world(self, img: np.ndarray) -> np.ndarray:
        """Gray-world white balance assumption."""
        r_avg, g_avg, b_avg = img.mean(axis=(0,1))
        gray_avg = (r_avg + g_avg + b_avg) / 3
        scale_factors = gray_avg / np.array([b_avg, g_avg, r_avg])
        return np.clip(img * scale_factors, 0, 255).astype(np.uint8)

    def normalize_edge_based(self, img: np.ndarray) -> np.ndarray:
        """Background sampling via edge detection."""
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
        bg_mask = cv2.dilate(edges, np.ones((5,5)), iterations=2) == 0
        bg_mean = img[bg_mask].mean(axis=0)
        target = np.array([255, 255, 255])
        scale_factors = target / bg_mean
        return np.clip(img * scale_factors, 0, 255).astype(np.uint8)
```

**Validation**:
```bash
# Test on tinted images
python scripts/background_normalization.py \
  --input data/zero_prediction_worst_performers \
  --method gray-world \
  --output artifacts/bg_norm_gray_world

# VLM validation
bash scripts/vlm_validate_enhancement.sh phase1_bg_norm_gray_world
```

##### Day 4: Implement Illumination Correction
- [ ] Implement Method 3: Morphological background estimation
  - Convert to grayscale
  - Large morphological opening: `cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_size=(50,50))`
  - Estimate background illumination field
  - Subtract from original: `img_corrected = img - (bg_estimate - 128)`
- [ ] Add `--method illumination-correction` to CLI
- [ ] Test on 5 images with uneven lighting
- [ ] Compare all 3 methods: Accuracy gain, processing time, visual quality

**Code Extension**:
```python
def normalize_illumination(self, img: np.ndarray) -> np.ndarray:
    """Morphological background estimation + subtraction."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    bg_estimate = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    corrected_gray = cv2.subtract(gray, cv2.subtract(bg_estimate, 128))
    return cv2.cvtColor(corrected_gray, cv2.COLOR_GRAY2BGR)
```

##### Day 5: Week 1 Validation & Integration Preparation
- [ ] Run all 3 methods on 10 test images
- [ ] Create before/after comparisons: `scripts/create_before_after_comparison.py`
- [ ] Run VLM validation: `bash scripts/vlm_validate_enhancement.sh phase1_bg_norm`
- [ ] Aggregate VLM reports: `python scripts/aggregate_vlm_validations.py`
- [ ] OCR accuracy comparison: Baseline vs. Gray-World vs. Edge-Based vs. Illumination
- [ ] Select best method for integration (target: >+5% OCR gain)
- [ ] Document results: `summaries/week1_background_normalization_results.md`

**Success Criteria**:
- Background color std dev: <10 (from ~45+)
- OCR accuracy gain: >5 percentage points
- Processing time: <50ms per image
- VLM overall improvement score: >+3 points
- No coordinate regression

---

#### **Week 2: Text Deskewing** 📋 PLANNED
**Objective**: Detect and correct text rotation → horizontal alignment
**Deliverable**: `scripts/text_deskewing.py` with 2 methods + coordinate tracking

##### Day 1-2: Implement Projection Profile Angle Detection
- [ ] Create `scripts/text_deskewing.py`
- [ ] Implement Method 1: Projection profile
  - Binarize: `cv2.threshold()` or Otsu
  - Compute horizontal projection: Sum pixels row-wise for angles -15° to +15° (0.5° steps)
  - Find angle with maximum variance (peak sharpness)
  - Rotate to detected angle
- [ ] Add coordinate transformation matrix tracking
- [ ] Test on synthetic rotations (known ground truth angles)

**Code Structure**:
```python
class TextDeskewer:
    def detect_angle_projection(self, img: np.ndarray) -> float:
        """Projection profile variance maximization."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        best_angle, max_variance = 0, 0
        for angle in np.arange(-15, 15, 0.5):
            rotated = self._rotate(binary, angle)
            projection = rotated.sum(axis=1)
            variance = projection.var()
            if variance > max_variance:
                max_variance = variance
                best_angle = angle
        return best_angle

    def deskew(self, img: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate and return transformation matrix."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderValue=(255,255,255))
        return rotated, M
```

##### Day 3-4: Implement Hough Transform Angle Detection
- [ ] Implement Method 2: Hough transform on text edges
  - Canny edge detection
  - Hough line transform: `cv2.HoughLinesP()`
  - Cluster line angles (filter outliers)
  - Compute median angle as text orientation
- [ ] Compare projection vs. Hough on real slanted receipts
- [ ] Measure angle detection error vs. manual annotation

**Code Extension**:
```python
def detect_angle_hough(self, img: np.ndarray) -> float:
    """Hough transform on text edges."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -15 < angle < 15:  # Filter near-horizontal
            angles.append(angle)

    return np.median(angles) if angles else 0
```

##### Day 5: Week 2 Validation
- [ ] Run both methods on 10 slanted images
- [ ] Create before/after comparisons
- [ ] VLM validation: `bash scripts/vlm_validate_enhancement.sh phase1_deskewing`
- [ ] OCR accuracy comparison
- [ ] Coordinate alignment test: Verify polygons still map correctly
- [ ] Select best method (target: >+10% OCR gain)
- [ ] Document results: `summaries/week2_deskewing_results.md`

**Success Criteria**:
- Angle detection error: <2°
- OCR accuracy gain: >10 percentage points (over baseline)
- Coordinate transformation matrix: Valid and tracked
- Processing time: <100ms per image

---

#### **Week 3: Integration & Validation** 📋 PLANNED
**Objective**: Combined pipeline with background normalization + deskewing
**Deliverable**: Integrated PreprocessingPipeline + ablation study + production validation

##### Day 1-2: Pipeline Integration
- [ ] Modify `ocr/inference/preprocessing_pipeline.py`
- [ ] Add Stage 1.5: Background normalization (before perspective correction)
- [ ] Add Stage 2.5: Text deskewing (after perspective correction)
- [ ] Add config flags: `enable_background_norm`, `enable_deskewing`
- [ ] Update `PreprocessingResult` dataclass: Add `bg_norm_method`, `deskew_angle`
- [ ] Maintain coordinate matrix composition: `M_total = M_perspective @ M_deskew @ M_bg`

**Code Changes**:
```python
# ocr/inference/preprocessing_pipeline.py
class PreprocessingPipeline:
    def process(self,
                image: np.ndarray,
                enable_perspective: bool = False,
                enable_background_norm: bool = False,  # NEW
                bg_norm_method: str = "edge-based",    # NEW
                enable_deskewing: bool = False,        # NEW
                deskew_method: str = "hough",          # NEW
                enable_grayscale: bool = False) -> PreprocessingResult:

        # Stage 0: Background normalization (NEW)
        if enable_background_norm:
            image = self.bg_normalizer.normalize(image, method=bg_norm_method)

        # Stage 1: Perspective correction
        if enable_perspective:
            image, perspective_matrix = self.perspective_corrector.correct(image)

        # Stage 1.5: Text deskewing (NEW)
        if enable_deskewing:
            angle = self.deskewer.detect_angle(image, method=deskew_method)
            image, deskew_matrix = self.deskewer.deskew(image, angle)

        # Stage 2-5: Original pipeline (grayscale, resize, pad, normalize)
        # ...
```

##### Day 3: Ablation Study
- [ ] Define 4 configurations:
  1. Baseline (no enhancements)
  2. Background normalization only
  3. Deskewing only
  4. Both enhancements
- [ ] Run all 4 on 25 worst performers
- [ ] Measure OCR accuracy, processing time, coordinate accuracy
- [ ] Statistical analysis: Paired t-test for significance
- [ ] Document ablation results: `summaries/phase1_ablation_study.md`

**Ablation Metrics**:
| Configuration | OCR Accuracy | Δ from Baseline | Processing Time | Coordinate Error |
|---------------|--------------|-----------------|-----------------|------------------|
| Baseline | X% | - | Yms | 0px |
| +BG Norm | X+5% | +5% | Y+50ms | 0px |
| +Deskew | X+10% | +10% | Y+100ms | <2px |
| +Both | X+18% | +18% | Y+150ms | <2px |

##### Day 4-5: Full Validation & Documentation
- [ ] VLM batch validation on all 25 images: `bash scripts/vlm_validate_enhancement.sh phase1_combined`
- [ ] Aggregate VLM summary: `python scripts/aggregate_vlm_validations.py`
- [ ] Performance benchmarking: Measure latency distribution (p50, p95, p99)
- [ ] Integration testing: Run full inference pipeline end-to-end
- [ ] Create Phase 1 completion report: `artifacts/phase1_completion_report.md`
- [ ] Update MASTER_ROADMAP.md: Mark Steps 4, 7-8, 10 as complete
- [ ] Go/No-Go decision: Proceed to Phase 2 if >15% OCR gain achieved

**Success Criteria (Phase 1 Complete)**:
- Combined OCR accuracy gain: >15 percentage points
- Processing time: <200ms total pipeline
- VLM overall improvement: >+5 points
- No coordinate regression (<2px error)
- No new pipeline failures (100% processing success rate)
- Production-ready: Code reviewed, tested, documented

---

### **PHASE 2: BACKGROUND WHITENING + TEXT ISOLATION (Weeks 4-6) - HIGH PRIORITY**
**Goal**: Implement Steps 11-14 for advanced background/text processing
**Effort**: 3 weeks (40-60 hours)
**Steps Implemented**: 8 → 12 (Steps 11-14)
**Expected OCR Gain**: +5-10 percentage points (cumulative: +20-40%)

#### **Week 4: Adaptive Thresholding + Background Normalization** 📋 FUTURE
**Deliverable**: `scripts/adaptive_processing.py` with Otsu/Sauvola/Adaptive methods

##### Implementation Tasks
- [ ] Step 11: Implement adaptive thresholding (Otsu, Sauvola, Adaptive Gaussian)
- [ ] Step 12: Implement background normalization (extend from Week 1)
- [ ] Test on 10 low-contrast images
- [ ] VLM validation + OCR accuracy comparison
- [ ] Document results

**Methods**:
```python
# Otsu's method
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive Gaussian
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, blockSize=11, C=2)

# Sauvola's method (local thresholding)
from skimage.filters import threshold_sauvola
thresh = threshold_sauvola(gray, window_size=25)
binary = (gray > thresh).astype(np.uint8) * 255
```

#### **Week 5: Shadow Removal + Noise Suppression** 📋 FUTURE
**Deliverable**: `scripts/shadow_removal.py` + `scripts/noise_suppression.py`

##### Implementation Tasks
- [ ] Step 13: Implement shadow removal (illumination normalization, morphological reconstruction)
- [ ] Step 14: Implement noise suppression (bilateral filter, morphological opening)
- [ ] Test on 10 shadowed/noisy images
- [ ] VLM validation + OCR accuracy comparison
- [ ] Document results

**Methods**:
```python
# Shadow removal via morphological reconstruction
bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_size=(50,50))
shadow_free = cv2.divide(gray, bg, scale=255)

# Noise suppression via bilateral filter
denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
```

#### **Week 6: Phase 2 Integration & Validation** 📋 FUTURE
- [ ] Integrate Steps 11-14 into PreprocessingPipeline
- [ ] Ablation study (8 configurations: Phase 1 baseline + 4 new steps)
- [ ] Full validation on 25 worst performers
- [ ] Performance benchmarking
- [ ] Phase 2 completion report
- [ ] Update MASTER_ROADMAP.md: Mark Steps 11-14 complete (12/21 = 57%)

---

### **PHASE 3: EDGE ENHANCEMENT + DETAIL (Weeks 7-8) - MEDIUM PRIORITY**
**Goal**: Implement Steps 15-17 for sharpening and edge enhancement
**Effort**: 2 weeks (25-35 hours)
**Steps Implemented**: 12 → 15 (Steps 15-17)
**Expected OCR Gain**: +3-5 percentage points (cumulative: +23-45%)

#### **Week 7: Sharpening + Unsharp Masking** 📋 FUTURE
**Deliverable**: `scripts/sharpening.py` with multiple techniques

##### Implementation Tasks
- [ ] Step 15: Implement sharpening (Laplacian, Unsharp masking)
- [ ] Test on blurry images
- [ ] VLM validation + OCR accuracy comparison

**Methods**:
```python
# Unsharp masking
gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

# Laplacian sharpening
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpened = cv2.subtract(img, laplacian)
```

#### **Week 8: Smoothing + Morphological Operations** 📋 FUTURE
**Deliverable**: `scripts/morphological_ops.py`

##### Implementation Tasks
- [ ] Step 16: Implement selective smoothing (preserve edges, smooth backgrounds)
- [ ] Step 17: Implement morphological operations (closing, opening for text cleanup)
- [ ] Phase 3 integration + validation
- [ ] Update MASTER_ROADMAP.md: Mark Steps 15-17 complete (15/21 = 71%)

---

### **PHASE 4: CLEANUP + MULTI-MODE (Weeks 9-10) - LOW PRIORITY**
**Goal**: Implement Steps 18-21 for final cleanup and multi-mode support
**Effort**: 2 weeks (25-35 hours)
**Steps Implemented**: 15 → 21 (Steps 18-21)
**Pipeline Completion**: 100%

#### **Week 9: Cleanup + Artifact Removal** 📋 FUTURE
**Deliverable**: `scripts/cleanup.py` + `scripts/artifact_removal.py`

##### Implementation Tasks
- [ ] Step 18: Speckle removal (morphological opening, connected components)
- [ ] Step 19: Small artifact removal (size-based filtering)
- [ ] Test on noisy scans

#### **Week 10: Border Removal + Multi-Mode Support** 📋 FUTURE
**Deliverable**: Complete pipeline with mode configurations

##### Implementation Tasks
- [ ] Step 20: Border/margin cleanup (detect and crop)
- [ ] Step 21: Multi-mode pipeline (document type-specific processing)
- [ ] Create pipeline presets: `receipt`, `invoice`, `id_card`, `contract`
- [ ] Final validation on all document types
- [ ] Phase 4 completion + MASTER_ROADMAP.md update (21/21 = 100%)

---

## Validation Framework

### VLM-Based Structured Assessment

#### Workflow Integration
1. **Baseline Assessment**: Run before any implementation
2. **Incremental Validation**: After each week's implementation
3. **Debugging Analysis**: When preprocessing fails
4. **Aggregated Reporting**: End of each phase

#### VLM Prompt Modes
- **image_quality**: Baseline quality scoring (1-10 scales, RGB estimates, angle detection)
- **enhancement_validation**: Before/after comparison with Δ metrics, success criteria (✅/⚠️/❌)
- **preprocessing_diagnosis**: Root cause analysis for failures with remediation strategies

#### VLM Report Structure
```
vlm_reports/
├── baseline/                    # Week 1 Day 1
│   ├── image_001_quality.md
│   └── ...
├── phase1_validation/           # Week 3 Day 4
│   ├── image_001_validation.md
│   └── ...
├── debugging/                   # As needed
│   ├── image_003_diagnosis.md
│   └── ...
└── summaries/
    ├── phase1_vlm_summary.md
    └── ...
```

#### Aggregation Scripts
- `scripts/aggregate_vlm_validations.py`: Parse reports → metrics table
- `scripts/vlm_baseline_assessment.sh`: Batch baseline assessment
- `scripts/vlm_validate_enhancement.sh`: Batch validation after enhancement

### OCR Accuracy Metrics

#### Baseline Measurement (Week 1 Day 1)
```bash
# Run baseline OCR
uv run python runners/predict.py \
  --input data/zero_prediction_worst_performers \
  --output artifacts/baseline_predictions.json \
  --config configs/predict.yaml

# Compute accuracy (if ground truth available)
python scripts/compute_accuracy.py \
  --predictions artifacts/baseline_predictions.json \
  --ground-truth data/ground_truth.json \
  --output artifacts/baseline_accuracy.json
```

#### Incremental Comparison
```bash
# After each enhancement
python scripts/compute_accuracy.py \
  --predictions artifacts/phase1_bg_norm_predictions.json \
  --ground-truth data/ground_truth.json \
  --baseline artifacts/baseline_accuracy.json \
  --output artifacts/phase1_bg_norm_accuracy.json \
  --show-delta
```

#### Success Criteria
| Metric | Baseline | Phase 1 Target | Phase 2 Target | Phase 3 Target | Phase 4 Target |
|--------|----------|----------------|----------------|----------------|----------------|
| OCR Accuracy | X% | X+15% | X+25% | X+30% | X+35% |
| Processing Time | Yms | Y+200ms | Y+300ms | Y+350ms | Y+400ms |
| Zero Predictions | Z images | Z-10 | Z-15 | Z-18 | Z-20 |

### Coordinate Alignment Validation

#### Test Method
```python
# scripts/validate_coordinates.py
def validate_coordinate_alignment(original_polygons, transformed_image, transform_matrix):
    """Ensure OCR polygons still map correctly after preprocessing."""
    # Apply transformation to ground truth polygons
    transformed_polygons = apply_transform(original_polygons, transform_matrix)

    # Visual overlay test
    overlay = draw_polygons(transformed_image, transformed_polygons)

    # Compute alignment error (IoU with manual annotation)
    error = compute_iou_error(transformed_polygons, manual_annotation)

    return error < 2.0  # Pass if <2px average error
```

---

## Technical Architecture

### Pipeline Extension Points

```python
# ocr/inference/preprocessing_pipeline.py
class PreprocessingPipeline:
    def process(self, image, **kwargs):
        # Stage 0: Background normalization (NEW - Phase 1)
        # Stage 1: Perspective correction (EXISTING)
        # Stage 1.5: Text deskewing (NEW - Phase 1)
        # Stage 2: Grayscale conversion (EXISTING - optional)
        # Stage 2.5: Adaptive thresholding (NEW - Phase 2)
        # Stage 3: Shadow removal (NEW - Phase 2)
        # Stage 3.5: Noise suppression (NEW - Phase 2)
        # Stage 4: Sharpening (NEW - Phase 3)
        # Stage 4.5: Morphological operations (NEW - Phase 3)
        # Stage 5: Resize with aspect preservation (EXISTING)
        # Stage 6: Padding to square (EXISTING)
        # Stage 6.5: Cleanup artifacts (NEW - Phase 4)
        # Stage 7: Normalization (EXISTING)
        return PreprocessingResult(...)
```

### Configuration Schema

```yaml
# configs/preprocessing.yaml
preprocessing:
  enable_perspective: true
  enable_background_norm: true    # NEW
  bg_norm_method: "edge-based"    # NEW: [gray-world|edge-based|illumination]
  enable_deskewing: true          # NEW
  deskew_method: "hough"          # NEW: [projection|hough]
  enable_grayscale: false
  enable_adaptive_threshold: false # NEW - Phase 2
  adaptive_method: "sauvola"       # NEW - Phase 2
  enable_shadow_removal: false     # NEW - Phase 2
  enable_noise_suppression: false  # NEW - Phase 2
  enable_sharpening: false         # NEW - Phase 3
  enable_morphological: false      # NEW - Phase 3
  enable_cleanup: false            # NEW - Phase 4

  # Mode presets (Phase 4)
  mode: "auto"  # [auto|receipt|invoice|id_card|contract]
```

### Coordinate Transform Composition

```python
# Maintain transformation matrix chain
M_total = M_perspective @ M_deskew @ M_background @ M_resize @ M_pad

# For inverse transforms (OCR polygons → original image)
M_inverse = np.linalg.inv(M_total)
original_polygon = apply_transform(ocr_polygon, M_inverse)
```

---

## Dependencies

### Existing Infrastructure
- ✅ Perspective correction: `ocr/utils/perspective_correction.py` (461 lines, 100% success)
- ✅ Preprocessing pipeline: `ocr/inference/preprocessing_pipeline.py` (276 lines, modular)
- ✅ VLM CLI: `AgentQMS.vlm.cli.analyze_defects`
- ✅ VLM prompts: `AgentQMS/vlm/prompts/markdown/{image_quality_analysis,enhancement_validation,preprocessing_diagnosis}.md`
- ✅ Test data: `data/zero_prediction_worst_performers/` (25 images)

### New Artifacts to Create
- [ ] `scripts/background_normalization.py` (Week 1)
- [ ] `scripts/text_deskewing.py` (Week 2)
- [ ] `scripts/create_before_after_comparison.py` (Week 1 Day 5)
- [ ] `scripts/aggregate_vlm_validations.py` (Week 1 Day 5)
- [ ] `scripts/vlm_baseline_assessment.sh` (Week 1 Day 1)
- [ ] `scripts/vlm_validate_enhancement.sh` (Week 1 Day 5)
- [ ] `scripts/compute_accuracy.py` (Week 1 Day 1)
- [ ] `scripts/validate_coordinates.py` (Week 3 Day 4)
- [ ] `scripts/adaptive_processing.py` (Week 4)
- [ ] `scripts/shadow_removal.py` (Week 5)
- [ ] `scripts/noise_suppression.py` (Week 5)
- [ ] `scripts/sharpening.py` (Week 7)
- [ ] `scripts/morphological_ops.py` (Week 8)
- [ ] `scripts/cleanup.py` (Week 9)
- [ ] `scripts/artifact_removal.py` (Week 9)

### External Libraries
```python
# requirements.txt additions
opencv-python>=4.8.0      # Already present
numpy>=1.24.0             # Already present
scikit-image>=0.21.0      # NEW: For Sauvola thresholding
scipy>=1.10.0             # NEW: For signal processing
```

---

## Risk Mitigation

### Risk 1: Preprocessing Breaks Coordinate Alignment
**Mitigation**: Maintain transformation matrix composition, validate after each step, <2px error threshold

### Risk 2: Processing Time Exceeds Budget (>400ms)
**Mitigation**: Profile each stage, optimize hot paths, consider GPU acceleration for morphological ops

### Risk 3: OCR Accuracy Regression on Some Images
**Mitigation**: Conditional application based on VLM quality score, fallback to baseline if preprocessing worsens quality

### Risk 4: Method Selection Ambiguity (e.g., Gray-World vs. Edge-Based)
**Mitigation**: Ablation study in Week 1 Day 5, select best method based on OCR gain + VLM scores

### Risk 5: VLM API Rate Limits
**Mitigation**: Batch processing with exponential backoff, cache VLM reports, use Solar Pro 2 backend for speed

---

## Success Metrics (Final)

### Phase 1 (Week 3)
- [x] OCR Accuracy: +15-30% (from baseline)
- [x] Processing Time: <200ms
- [x] VLM Improvement: >+5 points
- [x] Coordinate Error: <2px
- [x] Zero Predictions: Reduce by 10 images

### Phase 2 (Week 6)
- [ ] OCR Accuracy: +20-40% (cumulative)
- [ ] Processing Time: <300ms
- [ ] Pipeline Completion: 57% (12/21 steps)

### Phase 3 (Week 8)
- [ ] OCR Accuracy: +23-45% (cumulative)
- [ ] Processing Time: <350ms
- [ ] Pipeline Completion: 71% (15/21 steps)

### Phase 4 (Week 10)
- [ ] OCR Accuracy: +25-50% (cumulative)
- [ ] Processing Time: <400ms
- [ ] Pipeline Completion: 100% (21/21 steps)
- [ ] Multi-mode support: 4 document type presets

---

## Reporting Requirements

### Weekly Reports
- `summaries/week{N}_{enhancement_name}_results.md`
  - VLM scores (before/after)
  - OCR accuracy delta
  - Processing time
  - Visual examples (3-5 best improvements)
  - Failure analysis (if any)

### Phase Completion Reports
- `artifacts/phase{N}_completion_report.md`
  - Ablation study results
  - Aggregated VLM summary
  - OCR accuracy comparison table
  - Performance benchmarks
  - Go/No-Go recommendation for next phase

### Master Roadmap Updates
- Update `20251217_0243_assessment_master-roadmap.md` after each phase
- Mark completed steps, update timeline, adjust targets

---

## References

### Source Documentation
- `20251217_0243_assessment_priority-plan-revised.md`: 3-week detailed plan, code pseudocode
- `20251217_0243_assessment_master-roadmap.md`: 21-step pipeline tracking, phase breakdown
- `20251217_0243_assessment_current-state-summary.md`: Capabilities matrix, extension points
- `20251217_0243_assessment_executive-summary.md`: Strategic overview, success metrics
- `20251217_0243_guide_vlm-integration-guide.md`: VLM workflows, helper scripts

### VLM Prompts
- `AgentQMS/vlm/prompts/markdown/image_quality_analysis.md`: Baseline quality assessment
- `AgentQMS/vlm/prompts/markdown/enhancement_validation.md`: Before/after comparison
- `AgentQMS/vlm/prompts/markdown/preprocessing_diagnosis.md`: Failure root cause analysis

### Code References
- `ocr/utils/perspective_correction.py`: Perspective correction implementation
- `ocr/inference/preprocessing_pipeline.py`: Main pipeline architecture
- `ocr/inference/preprocess.py`: Transform helpers
- `experiment-tracker/experiments/20251129_173500_perspective_correction_implementation/scripts/`: Test scripts

---

## Autonomous Execution Protocol

1. **No clarification requests**: Execute tasks sequentially as defined
2. **Report after each week**: Update progress tracker, document results
3. **Block on failures**: If success criteria not met, analyze with VLM diagnosis, adjust approach
4. **Validate incrementally**: Run VLM + OCR validation after each enhancement
5. **Update master roadmap**: Mark steps complete, adjust timeline based on actual effort
6. **Go/No-Go gates**: Phase 1 completion requires >15% OCR gain for Phase 2 authorization

**START EXECUTION**: Week 1 Day 1 - Baseline Assessment & Setup
