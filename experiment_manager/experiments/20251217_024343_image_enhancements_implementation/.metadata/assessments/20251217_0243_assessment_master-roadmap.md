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
# Master Implementation Roadmap: 21-Step Office Lens Pipeline

**Experiment**: `20251217_024343_image_enhancements_implementation`
**Created**: 2025-12-17
**Vision**: Implement comprehensive Office Lens-style document enhancement pipeline incrementally

---

## 📊 Pipeline Overview

### Full 21-Step Sequence

```
[Geometry] → [Tone/Color] → [Background/Text] → [Edge/Detail] → [Cleanup]
  1-5           6-10            11-14              15-17          18-21
```

### Current Status (6/21 Complete - 29%)

| Phase | Steps | Status | Timeline | Notes |
|-------|-------|--------|----------|-------|
| **Geometry** | 1-5 | 4/5 ✅ | Complete | Missing: Step 4 (deskewing) |
| **Tone/Color** | 6-10 | 0/5 🚧 | Weeks 1-3 | Implementing: 7-8, 10 |
| **Background/Text** | 11-14 | 0/4 📋 | Phase 2 | Planned |
| **Edge/Detail** | 15-17 | 0/3 📋 | Phase 3 | Planned |
| **Cleanup** | 18-21 | 0/4 📋 | Phase 4 | Planned |

---

## 🎯 Phase Breakdown

### ✅ Phase 0: Geometric Foundation (COMPLETE)
**Status**: 4/5 steps complete (80%)
**Completion**: November 2025
**Experiment**: `20251129_173500_perspective_correction_implementation`

#### Implemented Steps
- [x] **Step 1**: Document boundary detection (rembg-based, 100% success)
- [x] **Step 2**: Page contour extraction (mask-based rectangle fitting)
- [x] **Step 3**: Perspective correction (Max-Edge + Lanczos4, 100% success)
- [ ] **Step 4**: Deskewing ← **In progress (Week 2)**
- [x] **Step 5**: Content cropping (bounding box refinement)

#### Key Achievements
- 100% success rate on 25 worst performers
- Robust multi-strategy fitting (standard/regression/dominant-extension)
- Coordinate transformation matrix tracking
- Production-ready perspective correction

#### Remaining Work
- **Step 4 (Deskewing)**: Text rotation correction
  - Timeline: Week 2 (5 days)
  - Methods: Projection profile + Hough transform
  - Target: <2° angle detection error

---

### 🚧 Phase 1: Tone & Color Normalization (IN PROGRESS)
**Status**: 0/5 steps (targeting 3/5 in Weeks 1-3)
**Timeline**: Weeks 1-3 (December 2025)
**Critical Issues**: Tinted backgrounds, illumination gradients

#### Implementation Plan

##### Week 1: White-Balance & Illumination (Steps 7-8)
- [ ] **Step 7**: White-balance correction
  - [ ] Gray-world white balance (Days 1-2)
  - [ ] Edge-based background estimation (Days 3-4)
  - [ ] Target: Background std dev <10

- [ ] **Step 8**: Illumination correction
  - [ ] Morphological background estimation (Day 4)
  - [ ] Division normalization (Day 5)
  - [ ] Target: Flatten lighting gradients

##### Week 2: Deskewing (Step 4 from Phase 0)
- [ ] **Step 4**: Deskewing/rotation correction
  - [ ] Projection profile angle detection (Days 1-2)
  - [ ] Hough transform angle detection (Days 3-4)
  - [ ] Rotation with coordinate tracking (Day 5)
  - [ ] Target: <2° angle detection error

##### Week 3: Local Contrast (Step 10) + Integration
- [ ] **Step 10**: Local contrast enhancement (CLAHE)
  - [ ] CLAHE implementation (Days 1-2)
  - [ ] Parameter tuning (clip_limit, tile_size)
  - [ ] Target: Improve faded text readability

- [ ] Integration & validation (Days 3-5)
  - [ ] Combined pipeline testing
  - [ ] Ablation study (4 configurations)
  - [ ] Performance profiling

#### Deferred Steps (Future)
- [ ] **Step 6**: Global tone adjustment (deferred - may not be needed)
- [ ] **Step 9**: Color space conversion (deferred - grayscale sufficient)

#### Success Criteria
- Background normalization: +5-10% OCR accuracy
- Deskewing: +10-20% OCR accuracy
- Combined: +15-30% OCR accuracy
- Processing time: <200ms total

---

### 📋 Phase 2: Background Whitening & Text Isolation (PLANNED)
**Status**: Not started (0/4 steps)
**Estimated Timeline**: Weeks 4-6 (January 2026)
**Dependencies**: Phase 1 completion

#### Planned Steps

##### Step 11: Adaptive Thresholding
**Purpose**: Binarize document for maximum text-background separation

**Methods to Evaluate**:
- Otsu's method (global)
- Niblack adaptive thresholding
- Sauvola adaptive thresholding
- Wolf adaptive thresholding

**Success Metrics**:
- Binary image preserves all text
- Background noise eliminated
- No text stroke damage

**Implementation Estimate**: 3-4 days

##### Step 12: Background Normalization
**Purpose**: Ensure uniform white background regardless of paper color/texture

**Methods to Evaluate**:
- Morphological reconstruction
- Background subtraction (adaptive)
- Color transfer to reference white

**Success Metrics**:
- Background uniformity: std dev <5
- Text contrast preserved or improved
- Works on patterned backgrounds

**Implementation Estimate**: 3-4 days

##### Step 13: Shadow Removal
**Purpose**: Eliminate shadow artifacts from lighting/camera angle

**Methods to Evaluate**:
- Illumination correction (may already be done in Step 8)
- Retinex-based shadow removal
- Deep learning shadow removal (if necessary)

**Success Metrics**:
- Shadowed regions match non-shadowed regions
- No text degradation in shadow areas
- OCR accuracy improvement in affected regions

**Implementation Estimate**: 4-5 days

**Note**: May be partially addressed by Step 8 (illumination correction). Assess necessity after Phase 1.

##### Step 14: Noise Suppression
**Purpose**: Remove camera noise, JPEG artifacts, speckles

**Methods to Evaluate**:
- Bilateral filter (edge-preserving)
- Non-local means denoising
- Morphological noise removal
- Median filtering

**Success Metrics**:
- Noise reduction without text blur
- Sharp text edges preserved
- OCR confidence improvement

**Implementation Estimate**: 2-3 days

#### Phase 2 Success Criteria
- Uniform white backgrounds across all images
- All text clearly separated from background
- Shadows eliminated or minimized
- Noise reduced without quality loss
- +5-10% additional OCR accuracy

---

### 📋 Phase 3: Edge & Detail Enhancement (PLANNED)
**Status**: Not started (0/3 steps)
**Estimated Timeline**: Weeks 7-8 (February 2026)
**Dependencies**: Phase 2 completion

#### Planned Steps

##### Step 15: Unsharp Mask Sharpening
**Purpose**: Enhance text edges for improved OCR recognition

**Methods to Evaluate**:
- Classic unsharp mask (blur + subtract + add)
- High-pass sharpening
- Laplacian sharpening

**Parameters to Tune**:
- Blur radius (1.0-3.0)
- Sharpening amount (0.5-2.0)
- Threshold (prevent noise amplification)

**Success Metrics**:
- Sharper text edges
- No ringing artifacts
- No noise amplification
- OCR confidence improvement

**Implementation Estimate**: 2-3 days

##### Step 16: Edge-Preserving Smoothing
**Purpose**: Smooth noise while preserving text structure

**Methods to Evaluate**:
- Bilateral filter (already used in Phase 2?)
- Guided filter
- Anisotropic diffusion

**Success Metrics**:
- Noise reduced
- Text edges sharp
- No blurring of characters
- Balance between Steps 14 & 15

**Implementation Estimate**: 2-3 days

**Note**: May overlap with Step 14 (noise suppression). Consider combining.

##### Step 17: Morphological Text Stroke Optimization
**Purpose**: Regularize text stroke width and connectivity

**Methods to Evaluate**:
- Morphological opening/closing
- Stroke width transform
- Connected component analysis + regularization

**Success Metrics**:
- Consistent stroke width
- Broken characters reconnected
- No character merging
- Improved character segmentation

**Implementation Estimate**: 3-4 days

#### Phase 3 Success Criteria
- Text edges sharp and well-defined
- Consistent text stroke appearance
- Noise eliminated without blur
- +3-5% additional OCR accuracy

---

### 📋 Phase 4: Layout Cleanup & Output Modes (PLANNED)
**Status**: Not started (0/4 steps)
**Estimated Timeline**: Weeks 9-10 (February-March 2026)
**Dependencies**: Phase 3 completion

#### Planned Steps

##### Step 18: Morphological Cleaning
**Purpose**: Remove small artifacts, speckles, border noise

**Methods to Evaluate**:
- Morphological opening (remove small objects)
- Morphological closing (fill small holes)
- Connected component filtering (area threshold)

**Success Metrics**:
- Isolated noise removed
- Text preserved
- Clean document appearance

**Implementation Estimate**: 2 days

##### Step 19: Artifact Removal
**Purpose**: Detect and remove non-text artifacts (stamps, logos, watermarks)

**Methods to Evaluate**:
- Connected component analysis
- Template matching (if common artifacts)
- Texture analysis
- Deep learning artifact detection (if necessary)

**Success Metrics**:
- Artifacts removed without text damage
- OCR confusion reduction
- Cleaner layout

**Implementation Estimate**: 4-5 days

**Note**: May require manual annotation of artifact types. Consider deferred if not critical.

##### Step 20: Border Cleanup
**Purpose**: Remove irregular borders, crop to document content

**Methods to Evaluate**:
- Content-based cropping (after Step 5)
- Border detection and removal
- Margin normalization

**Success Metrics**:
- Clean borders
- Consistent margins
- No content loss

**Implementation Estimate**: 2-3 days

##### Step 21: Multi-Mode Rendering
**Purpose**: Generate multiple output formats for different use cases

**Modes to Implement**:
1. **Photo Mode**: Color-preserved, minimal processing
2. **Document Mode**: High-contrast, optimized for text
3. **Whiteboard Mode**: Aggressive cleaning, ultra-white background
4. **Grayscale Mode**: Grayscale-optimized for OCR

**Success Metrics**:
- Each mode serves its purpose
- User selectable or auto-detected
- Consistent quality across modes

**Implementation Estimate**: 3-4 days

#### Phase 4 Success Criteria
- Professional document appearance
- Multiple output modes available
- Artifacts eliminated
- +2-5% additional OCR accuracy (cumulative: 25-50% total improvement)

---

## 📈 Progress Tracking

### Overall Pipeline Completion

```
Phase 0 (Geometry):         ████████████████░░░░ 80% (4/5)
Phase 1 (Tone/Color):       ░░░░░░░░░░░░░░░░░░░░  0% (targeting 3/5 → 60%)
Phase 2 (Background/Text):  ░░░░░░░░░░░░░░░░░░░░  0% (0/4)
Phase 3 (Edge/Detail):      ░░░░░░░░░░░░░░░░░░░░  0% (0/3)
Phase 4 (Cleanup):          ░░░░░░░░░░░░░░░░░░░░  0% (0/4)
─────────────────────────────────────────────────
Total:                      ████░░░░░░░░░░░░░░░░ 19% (4/21)
```

### After Phase 1 Completion (Target: Week 3)
```
Phase 0 (Geometry):         ████████████████████ 100% (5/5) ✅
Phase 1 (Tone/Color):       ████████████░░░░░░░░  60% (3/5)
Total:                      ████████░░░░░░░░░░░░  38% (8/21)
```

### Estimated Full Completion: Q1-Q2 2026 (10-12 weeks total)

---

## 🎓 Implementation Strategy

### Incremental Approach
1. **Implement 2-3 steps at a time** (weekly sprints)
2. **Validate each step independently** before integration
3. **Ablation studies** to measure individual contributions
4. **Go/no-go gates** after each phase

### Parallel Workstreams
- **Primary**: Implement next phase steps
- **Validation**: VLM-based quality assessment
- **Documentation**: Update experiment logs, create before/after reports
- **Integration**: Merge successful steps into main pipeline

### Risk Mitigation
- **Fallback**: Each step gated behind configuration flag
- **Rollback**: Can revert to previous phase if issues arise
- **Cherry-picking**: Can deploy successful steps independently

---

## 🧪 Validation Framework

### Per-Step Validation
1. **Unit testing**: Step works on controlled inputs
2. **Visual inspection**: Before/after comparisons
3. **OCR metrics**: Accuracy improvement measurement
4. **VLM assessment**: Structured quality reports
5. **Performance**: Processing time impact

### Phase-Level Validation
1. **Ablation study**: Test all combinations of phase steps
2. **Worst performers**: Full test on 25 images
3. **Production sample**: Test on random 100 images
4. **Regression testing**: Ensure no degradation on good images

### Pipeline-Level Validation
1. **End-to-end**: Full pipeline on diverse test set
2. **Performance profiling**: Total processing time budget
3. **Robustness**: Edge cases, failure modes
4. **Deployment readiness**: Integration testing

---

## 📋 Dependencies & Prerequisites

### Phase Dependencies
- Phase 1 requires: Phase 0 (Step 5) complete
- Phase 2 requires: Phase 1 (Steps 7-8) complete
- Phase 3 requires: Phase 2 (Steps 11-14) complete
- Phase 4 requires: Phase 3 (Steps 15-17) complete

### Step Dependencies
- Step 8 (Illumination) benefits from Step 7 (White-balance)
- Step 10 (CLAHE) works best after Step 8 (Illumination)
- Step 12 (Background norm) may obviate Step 13 (Shadow removal)
- Step 16 (Smoothing) balances Step 15 (Sharpening)

### Technical Prerequisites
- Python 3.11+
- OpenCV 4.8+
- NumPy 1.24+
- VLM tool for validation (optional but recommended)
- Test dataset with ground truth labels

---

## 🎯 Success Metrics (Cumulative)

### OCR Accuracy Targets
- **Phase 0 (Baseline)**: 60% accuracy on worst performers
- **Phase 1 (+15-30%)**: 75-90% accuracy
- **Phase 2 (+5-10%)**: 80-95% accuracy
- **Phase 3 (+3-5%)**: 83-98% accuracy
- **Phase 4 (+2-5%)**: 85-100% accuracy

**Total Target**: 85-100% accuracy on worst performers (25-40 point improvement)

### Processing Time Budget
- **Phase 0**: <50ms (perspective correction)
- **Phase 1**: +50ms (white-balance + deskew + CLAHE) = 100ms total
- **Phase 2**: +50ms (thresholding + background + shadow + denoise) = 150ms total
- **Phase 3**: +30ms (sharpening + smoothing + morphological) = 180ms total
- **Phase 4**: +20ms (cleaning + artifacts + borders) = 200ms total

**Total Budget**: <200ms per image (real-time viable)

### Quality Metrics
- Background uniformity: std dev <5
- Text alignment: angle error <1°
- Edge sharpness: increase in Laplacian variance
- Contrast ratio: text/background >10:1
- Noise level: decrease in noise power

---

## 🔄 Iteration & Refinement

### After Phase 1
- Review VLM assessments for unexpected issues
- Adjust Phase 2 priorities based on remaining problems
- Consider skipping steps if already addressed (e.g., shadow removal if illumination correction sufficient)

### After Phase 2
- Evaluate whether Phase 3 needed (may have reached accuracy target)
- Consider deep learning approaches if classical methods plateau

### After Phase 3
- Decide whether Phase 4 needed or if focus shifts to other priorities

### Continuous
- Monitor OCR accuracy on production data
- Collect new failure cases for future improvements
- Update roadmap based on learnings

---

## 📁 Documentation Structure

```
experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/
├── MASTER_ROADMAP.md                 # This document (overall 21-step plan)
├── PRIORITY_PLAN_REVISED.md          # Detailed Phase 1 implementation (Weeks 1-3)
├── README.md                          # Experiment overview
├── EXECUTIVE_SUMMARY.md              # Strategic summary
├── CURRENT_STATE_SUMMARY.md          # Current capabilities analysis
├── ENHANCEMENT_QUICK_REFERENCE.md    # Quick reference guide
├── phase_reports/
│   ├── phase0_complete.md            # Phase 0 completion report
│   ├── phase1_progress.md            # Phase 1 progress (updated weekly)
│   ├── phase2_plan.md                # Phase 2 detailed plan (when ready)
│   └── ...
├── vlm_reports/                       # VLM quality assessments
├── scripts/                           # Implementation scripts
└── artifacts/                         # Test outputs
```

---

## 🚦 Go/No-Go Decision Points

### After Phase 1 (Week 3)
**Measure**: OCR accuracy gain, processing time, robustness
**Go to Phase 2 if**: >10% accuracy gain, <150ms processing, no regressions
**Pivot if**: Gains not materializing; may need to revisit approach
**Stop if**: Critical issues in production integration

### After Phase 2 (Week 6)
**Measure**: Cumulative accuracy gain, background uniformity, shadow reduction
**Go to Phase 3 if**: >20% cumulative gain, backgrounds uniform, shadows minimal
**Pivot if**: Certain steps ineffective; skip or replace them
**Stop if**: Reached accuracy target (may not need Phases 3-4)

### After Phase 3 (Week 8)
**Measure**: Cumulative accuracy gain, text edge quality
**Go to Phase 4 if**: >25% cumulative gain, further improvement possible
**Stop if**: Reached accuracy plateau or target met

### After Phase 4 (Week 10)
**Measure**: Final accuracy, processing time, production readiness
**Deploy if**: >25% cumulative gain, <200ms processing, robust
**Iterate if**: Specific steps underperforming; refinement needed

---

## 🎯 Current Focus: Phase 1 (Weeks 1-3)

**See**: [PRIORITY_PLAN_REVISED.md](PRIORITY_PLAN_REVISED.md) for detailed implementation plan

**Current Steps**:
- Step 7: White-balance correction (Week 1)
- Step 8: Illumination correction (Week 1)
- Step 4: Deskewing (Week 2)
- Step 10: CLAHE (Week 3)

**Next Phase**: Phase 2 (Steps 11-14) planned for Weeks 4-6

---

## 📝 Maintenance & Updates

This roadmap is a **living document**. Update after:
- Each phase completion
- Significant findings from VLM assessments
- Changes in priorities based on production observations
- New techniques or methods discovered

**Last Updated**: 2025-12-17
**Next Review**: After Phase 1 completion (Week 3)
