---
ads_version: "1.0"
type: plan
title: Border Removal Preprocessing Experiment Implementation Plan
status: draft
created: 2025-12-18T19:05:00+09:00
updated: 2025-12-18T19:05:00+09:00
experiment_id: 20251217_024343_image_enhancements_implementation
phase: phase_0
tags: [implementation-plan, border-removal, new-experiment]
related_artifacts:
 - 20251218_1900_assessment_border-removal-scoping.md
target_experiment: 20251218_1900_border_removal_preprocessing
---

# Border Removal Preprocessing - Implementation Plan

## Experiment Metadata

- **Experiment ID**: `20251218_1900_border_removal_preprocessing`
- **Type**: Preprocessing improvement
- **Parent Experiment**: `20251217_024343_image_enhancements_implementation`
- **Timeline**: 5-8 days (parallel with integration)
- **Priority**: Medium (runs parallel with high-priority integration)

## Problem Statement

Black borders around scanned documents cause severe skew misdetection in preprocessing pipeline. Image 000732 shows -83° skew due to edge detection picking up border edges instead of text lines.

**Impact**:
- Deskewing fails on border-affected images
- OCR accuracy degraded (zero predictions on 000732)
- Pipeline robustness compromised

## Objectives

### Primary
1. Detect document boundaries accurately (>95% success rate)
2. Crop to content area, removing border artifacts
3. Resolve -83° misdetection to <15° baseline skew

### Secondary
1. Handle various border types (black, white, colored, partial)
2. Preserve border-free images (0% false crop rate)
3. Process efficiently (<50ms per image)

## Success Criteria

- **Accuracy**: Correct boundary detection on 95% of border-affected images
- **False positive**: <5% false crops on border-free images
- **Skew improvement**: 000732 skew from -83° to <15°
- **Performance**: <50ms processing time per image
- **Generalization**: Works on multiple receipt/invoice types

## Phase 1: Research & Baseline (Days 1-2)

### 1.1 Literature Review (4 hours)
**Goal**: Survey border detection methods

**Methods to research**:
1. **Classical CV**:
 - Canny edge + contour detection + largest quadrilateral
 - Morphological operations (closing, dilation) + contours
 - Hough transform for line detection + intersection

2. **Adaptive Thresholding**:
 - OTSU + morphology + boundary tracing
 - Adaptive Gaussian threshold + contours

3. **ML/Deep Learning** (optional, if time permits):
 - Pre-trained segmentation models (U-Net, DeepLabV3)
 - Document boundary detection models (DocUNet, etc.)

**Deliverables**:
- Research notes document (markdown)
- Method comparison table (pros/cons, complexity, speed)
- Selected 2-3 methods for implementation

### 1.2 Test Data Collection (2 hours)
**Goal**: Build comprehensive border test set

**Tasks**:
1. Identify border-affected images in test set:
 - Search for high skew angles (>15°)
 - Visual inspection of zero-prediction images
 - Check 91+ test images across 6 categories

2. Categorize border types:
 - Black borders (like 000732)
 - White borders
 - Colored borders
 - Partial borders (one or two sides)
 - No borders (control group)

3. Create synthetic border dataset:
 - Take 10 clean images
 - Add artificial borders (various widths, colors)
 - Use as validation set

**Deliverables**:
- Border test set manifest (JSON)
- Categorized image lists
- Synthetic dataset (10 images × 3 border types = 30 images)

### 1.3 Baseline Metrics (2 hours)
**Goal**: Establish baseline performance

**Metrics to measure**:
- Skew angles on border-affected images (current)
- OCR accuracy on border-affected images (with epoch-18 checkpoint)
- Comparison with border-free images

**Deliverables**:
- Baseline metrics JSON
- Assessment document (EDS-compliant)

## Phase 2: Implementation (Days 3-4)

### 2.1 Method 1: Canny + Contour Detection (4 hours)

**Algorithm**:
```python
1. Convert to grayscale
2. Apply Gaussian blur (reduce noise)
3. Canny edge detection
4. Find contours
5. Filter contours (area threshold, convexity)
6. Fit largest quadrilateral (Douglas-Peucker approximation)
7. Perspective transform to crop
```

**Parameters to tune**:
- Canny thresholds (50, 150 default)
- Gaussian kernel size (5×5 default)
- Contour area threshold (% of image area)
- Approximation epsilon (0.02 default)

**Implementation**:
- Script: `border_removal_canny.py`
- Tests: Validate on 000732 first, then full test set

### 2.2 Method 2: Morphological Operations (4 hours)

**Algorithm**:
```python
1. Convert to grayscale
2. OTSU threshold
3. Morphological closing (fill holes)
4. Dilation (connect components)
5. Find contours
6. Select largest rectangular contour
7. Crop to bounding rectangle
```

**Parameters to tune**:
- Morphological kernel size (3×3, 5×5, 7×7)
- Closing iterations (1-3)
- Dilation iterations (1-2)

**Implementation**:
- Script: `border_removal_morph.py`
- Tests: Compare with Method 1 results

### 2.3 Method 3: Hough Line + Intersection (4 hours, optional)

**Algorithm**:
```python
1. Canny edge detection
2. Hough line transform (detect border lines)
3. Cluster lines by orientation (horizontal/vertical)
4. Find intersections (corner points)
5. Crop to corner-defined rectangle
```

**Implementation**:
- Script: `border_removal_hough.py`
- Tests: Validate on synthetic dataset

### 2.4 Unified Interface (2 hours)

**Goal**: Create consistent API for all methods

**Interface**:
```python
class BorderRemover:
 def __init__(self, method='canny'):
 self.method = method

 def detect_border(self, img) -> Tuple[np.ndarray, dict]:
 # Returns: corner points, metrics
 pass

 def remove_border(self, img) -> Tuple[np.ndarray, dict]:
 # Returns: cropped image, metrics
 pass
```

**Features**:
- Method selection via config
- Metrics logging (processing time, confidence)
- Fallback if detection fails (return original)

## Phase 3: Validation (Days 5-6)

### 3.1 Method Comparison (4 hours)

**Test suite**:
1. Run all methods on border test set
2. Measure:
 - Detection accuracy (% correct boundaries)
 - False positive rate (crops on border-free images)
 - Processing time per image
 - Skew improvement (before/after deskewing)

**Deliverables**:
- Comparison table (methods × metrics)
- Visualization (before/after crops)
- Assessment document (method selection)

### 3.2 Skew Validation (4 hours)

**Goal**: Verify border removal resolves skew misdetection

**Test**:
1. Apply border removal to 000732
2. Run Hough deskewing on result
3. Measure skew angle (expect <15° instead of -83°)
4. Repeat for all border-affected images

**Deliverables**:
- Skew improvement metrics
- Comparison: baseline → border removed → deskewed

### 3.3 False Positive Analysis (2 hours)

**Goal**: Ensure no unwanted crops on border-free images

**Test**:
1. Run selected method on 10 border-free images
2. Measure crop area vs original area (expect >99% preserved)
3. Visual inspection for false crops

**Deliverables**:
- False positive rate
- Failed cases analysis (if any)

### 3.4 VLM Validation (Optional, 2 hours)

**Goal**: Visual quality confirmation

**Test**:
1. Run VLM analysis on 000732 before/after border removal
2. Check for artifacts, content loss
3. Confirm boundary detection quality

**Deliverables**:
- VLM validation report (if time permits)

## Phase 4: Integration Planning (Day 7)

### 4.1 Pipeline Placement Decision (2 hours)

**Question**: Where does border removal fit?

**Option A - First (before all enhancements)**:
```
Input → Border Removal → Background Norm → Deskewing → OCR
```
**Pros**: Clean borders before edge detection
**Cons**: May crop needed info if detection fails

**Option B - After background norm, before deskewing**:
```
Input → Background Norm → Border Removal → Deskewing → OCR
```
**Pros**: Background norm may help border detection
**Cons**: Background tint may confuse border detection

**Option C - Conditional (only on high skew)**:
```
Input → Deskewing → [if skew >20°] → Border Removal → Deskewing again → OCR
```
**Pros**: Only runs when needed
**Cons**: Two deskewing passes (slower)

**Deliverable**: Integration decision document

### 4.2 Configuration Design (2 hours)

**YAML config**:
```yaml
preprocessing:
 border_removal:
 enabled: true
 method: canny # or morph, hough
 min_area_ratio: 0.5 # minimum doc area vs image
 confidence_threshold: 0.8
 fallback_to_original: true
```

**Deliverable**: Configuration schema

### 4.3 Integration Guide (2 hours)

**Documentation**:
1. How to enable/disable border removal
2. Parameter tuning guide
3. Troubleshooting common failures
4. Performance benchmarks

**Deliverable**: Integration guide (markdown)

## Deliverables Summary

### Scripts
- `border_removal_canny.py` (Method 1)
- `border_removal_morph.py` (Method 2)
- `border_removal_hough.py` (Method 3, optional)
- `border_remover.py` (Unified interface)
- `test_border_removal.py` (Unit tests)

### Data Artifacts
- Border test set manifest (JSON)
- Synthetic border dataset (30 images)
- Baseline metrics (JSON)
- Method comparison results (JSON)
- Processed images (border-removed outputs)

### Documentation (EDS v1.0 compliant)
- Research notes (`.metadata/assessments/`)
- Baseline analysis (`.metadata/assessments/`)
- Method comparison (`.metadata/assessments/`)
- Skew validation (`.metadata/assessments/`)
- Integration guide (`.metadata/guides/`)
- Final report (`.metadata/reports/`)

## Dependencies

### Software
- OpenCV (cv2) - edge detection, contours, morphology
- NumPy - array operations
- Python 3.11+ - implementation language

### Data
- Image 000732 (confirmed border case)
- Test set (91+ images to search for more border cases)
- Clean images for synthetic generation

### Checkpoints
- epoch-18_step-001957.ckpt (for OCR validation)

## Risks & Mitigations

### Risk 1: Insufficient Border Cases
**Risk**: Only 1 confirmed border image (000732)
**Impact**: Can't validate generalization
**Mitigation**: Generate synthetic dataset (30 images), search test set thoroughly

### Risk 2: False Crops on Clean Images
**Risk**: Method too aggressive, crops content
**Impact**: OCR accuracy regression
**Mitigation**: Test on 10 border-free images, set confidence threshold, fallback to original

### Risk 3: Border Detection Fails
**Risk**: Complex borders (e.g., textured, gradient)
**Impact**: No improvement on problem images
**Mitigation**: Implement 3 methods, select best, add fallback logic

### Risk 4: Integration Complexity
**Risk**: Pipeline ordering unclear
**Impact**: Delays integration with main experiment
**Mitigation**: Test all 3 placement options (A, B, C), document trade-offs

## Timeline

| Day | Phase | Tasks | Deliverables |
|-----|-------|-------|--------------|
| 1 | Research | Literature review, test data collection | Research notes, test set |
| 2 | Baseline | Baseline metrics, synthetic data gen | Baseline assessment |
| 3 | Implementation | Method 1 (Canny), Method 2 (Morph) | 2 scripts |
| 4 | Implementation | Method 3 (Hough, opt), Unified interface | 2 scripts |
| 5 | Validation | Method comparison, skew validation | Comparison assessment |
| 6 | Validation | False positive analysis, VLM (opt) | Validation report |
| 7 | Integration | Pipeline placement, config, guide | Integration guide |
| 8 | Buffer | Contingency for issues, documentation | Final report |

**Total**: 7-8 days (can overlap with main experiment integration)

## Success Metrics

### Must-Have
- Resolve 000732 skew from -83° to <15°
- <5% false positive rate on border-free images
- <50ms processing time per image

### Nice-to-Have
- Works on 95% of border-affected images
- Handles colored/white borders (not just black)
- VLM validation score >4.0/5.0

## Handoff Information

### For New Conversation (Border Removal Implementation)
1. Start with: "Create new experiment 20251218_1900_border_removal_preprocessing"
2. Reference: This plan document
3. Begin with Phase 1 (Research & Baseline)
4. Use AgentQMS tools to create experiment structure

### For Integration Conversation (Main Pipeline)
1. Proceed with gray-world + Hough deskewing integration
2. Run OCR validation with current pipeline
3. Wait for border removal experiment results
4. Integrate border removal when ready (Day 7-8)
5. Rerun OCR validation with border removal included

---
*EDS v1.0 compliant | Implementation Plan | Ready for New Experiment Creation*
