# Image Enhancement Experiment: 21-Step Office Lens Pipeline

**Experiment ID**: `20251217_024343_image_enhancements_implementation`
**Created**: 2025-12-17T02:43:43
**Updated**: 2025-12-18T05:30:00
**Status**: Phase 1 Week 1 Day 1 Ready
**Branch**: `refactor/inference-module-consolidation`

---

## Executive Summary

Implement comprehensive 21-step Office Lens-style document enhancement pipeline for OCR preprocessing through incremental phases. **Current focus: Phase 1 (Weeks 1-3)** - Background normalization + text deskewing to address critical production issues (tinted backgrounds, text slant).

**Target**: +15-30% OCR accuracy improvement in Phase 1

---

## Current Status

### Pipeline Progress
- **Completed**: 4/21 steps (19%) - Perspective correction, background removal, resize/pad, normalization
- **Phase 1 Target**: 8/21 steps (38%) - Add background white-balance, deskewing, combined pipeline
- **Overall Target**: 21/21 steps (100%) by Q1-Q2 2026

### Critical Issues Identified
Based on production observations with 100% cropping success:
1. 🔴 **Tinted backgrounds** (cream, gray, yellow) → severe inference degradation
2. 🔴 **Text slant/skew** → no deskewing after perspective correction
3. ⚠️ **Shadow artifacts** → deferred to Phase 2

### Validation Infrastructure
- ✅ VLM prompts: quality assessment, enhancement validation, preprocessing diagnosis
- ✅ Helper scripts: baseline assessment, validation, aggregation
- ✅ Test data: 25 worst performers from parent experiment

---

## Master Documentation

### 🎯 Primary Reference (Start Here)
**[20251218_0530_implementation_plan_21step-image-enhancement-pipeline.md](20251218_0530_implementation_plan_21step-image-enhancement-pipeline.md)**

Complete 10-12 week implementation plan with:
- Week-by-week task breakdown
- Code examples and pseudocode
- VLM validation workflows
- Success criteria per phase
- Risk mitigation strategies

### 📋 Supporting Documents
| Document | Purpose | Use When |
|----------|---------|----------|
| [20251217_0243_assessment_priority-plan-revised.md](20251217_0243_assessment_priority-plan-revised.md) | 3-week detailed plan (Phase 1) | Implementing Week 1-3 |
| [20251217_0243_assessment_master-roadmap.md](20251217_0243_assessment_master-roadmap.md) | 21-step pipeline tracking | Tracking overall progress |
| [20251217_0243_assessment_current-state-summary.md](20251217_0243_assessment_current-state-summary.md) | Capabilities matrix | Understanding codebase |
| [20251217_0243_assessment_executive-summary.md](20251217_0243_assessment_executive-summary.md) | Strategic overview | High-level context |
| [20251217_0243_guide_vlm-integration-guide.md](20251217_0243_guide_vlm-integration-guide.md) | VLM workflows | Running validations |
| [20251217_0243_assessment_enhancement-quick-reference.md](20251217_0243_assessment_enhancement-quick-reference.md) | Quick reference | Quick lookups |

---

## Quick Start (Week 1 Day 1)

### Prerequisites
```bash
# 1. Verify VLM CLI
uv run python -m AgentQMS.vlm.cli.analyze_defects --help

# 2. Verify test images
ls -la data/zero_prediction_worst_performers/*.jpg | head -10

# 3. Create directories
cd experiment-tracker/experiments/20251217_024343_image_enhancements_implementation
mkdir -p vlm_reports/{baseline,phase1_validation,debugging} artifacts outputs/comparisons
```

### Run Baseline Assessment
```bash
# Step 1: VLM baseline (10 images, ~5 min)
bash scripts/vlm_baseline_assessment.sh

# Step 2: OCR baseline inference
uv run python runners/predict.py \
  --input data/zero_prediction_worst_performers \
  --output artifacts/baseline_predictions.json \
  --config configs/predict.yaml

# Step 3: Review VLM reports
cat vlm_reports/baseline/*_quality.md | grep "Overall Quality Score"
cat vlm_reports/baseline/*_quality.md | grep "Priority Ranking" -A5

# Step 4: Document baseline
# Record OCR accuracy + VLM scores in artifacts/phase1_baseline_metrics.json
```

### Next Steps
- **Day 2-3**: Implement `scripts/background_normalization.py` (gray-world, edge-based, illumination)
- **Day 4**: Test on 10 tinted images, VLM validation
- **Day 5**: Aggregate results, select best method
- **Week 2**: Implement text deskewing
- **Week 3**: Integration + ablation study

---

## Directory Structure

```
20251217_024343_image_enhancements_implementation/
├── README.md                                              # This file
├── 20251218_0530_implementation_plan_*.md                 # ⭐ MASTER PLAN
├── 20251217_0243_assessment_*.md                          # Supporting assessments
├── 20251217_0243_guide_vlm-integration-guide.md           # VLM workflows
│
├── scripts/                                               # Implementation scripts
│   ├── vlm_baseline_assessment.sh                         # ✅ Ready (Week 1 Day 1)
│   ├── vlm_validate_enhancement.sh                        # ✅ Ready
│   ├── aggregate_vlm_validations.py                       # ✅ Ready
│   │
│   ├── background_normalization.py                        # 📋 TODO (Week 1 Day 2-3)
│   ├── text_deskewing.py                                  # 📋 TODO (Week 2 Day 1-4)
│   ├── create_before_after_comparison.py                  # 📋 TODO (Week 1 Day 5)
│   ├── compute_accuracy.py                                # 📋 TODO (Week 1 Day 1)
│   ├── validate_coordinates.py                            # 📋 TODO (Week 3 Day 4)
│   │
│   └── [Copied from parent experiment]
│       ├── mask_only_edge_detector.py                     # ✅ 1339 lines
│       ├── perspective_transformer.py                     # ✅ Transform utils
│       ├── run_perspective_correction.py                  # ✅ Batch processing
│       └── test_worst_performers.py                       # ✅ Validation
│
├── artifacts/                                             # Generated outputs
│   ├── phase1_baseline_metrics.json                       # 📋 TODO (Week 1 Day 1)
│   ├── baseline_predictions.json                          # 📋 TODO (Week 1 Day 1)
│   ├── bg_norm_gray_world/                                # 📋 TODO (Week 1 Day 2-3)
│   ├── bg_norm_edge_based/                                # 📋 TODO (Week 1 Day 3)
│   ├── bg_norm_illumination/                              # 📋 TODO (Week 1 Day 4)
│   ├── phase1_enhanced/                                   # 📋 TODO (Week 2)
│   └── phase1_completion_report.md                        # 📋 TODO (Week 3 Day 5)
│
├── vlm_reports/                                           # VLM assessment reports
│   ├── baseline/                                          # 📋 TODO (Week 1 Day 1)
│   │   └── image_001_quality.md
│   ├── phase1_validation/                                 # 📋 TODO (Week 3 Day 4)
│   │   └── image_001_validation.md
│   ├── debugging/                                         # As needed
│   │   └── image_003_diagnosis.md
│   └── summaries/
│       └── phase1_vlm_summary.md
│
├── outputs/                                               # Visualization outputs
│   └── comparisons/                                       # Before/after images
│       ├── phase1_bg_norm/
│       ├── phase1_deskewing/
│       └── phase1_combined/
│
├── .metadata/                                             # Experiment metadata
│   └── [...tracking files...]
│
└── state.json                                             # Experiment state
```

---

## Implementation Phases

### Phase 1: Background Normalization + Deskewing (Weeks 1-3) ⏳ CURRENT
**Priority**: Critical (addresses production observations)
**Steps**: 4 → 8 (Steps 4, 7-8, 10)
**Expected Gain**: +15-30% OCR accuracy

#### Week 1: Background Normalization
- Day 1: Baseline assessment (VLM + OCR)
- Day 2-3: Implement gray-world + edge-based white balance
- Day 4: Implement illumination correction
- Day 5: Validation, method selection

#### Week 2: Text Deskewing
- Day 1-2: Projection profile angle detection
- Day 3-4: Hough transform angle detection
- Day 5: Validation, method selection

#### Week 3: Integration & Validation
- Day 1-2: Pipeline integration
- Day 3: Ablation study (4 configurations)
- Day 4-5: Full validation, completion report

### Phase 2: Background Whitening + Text Isolation (Weeks 4-6) 📋 PLANNED
**Steps**: 8 → 12 (Steps 11-14)
**Expected Gain**: +5-10% OCR accuracy (cumulative: +20-40%)

### Phase 3: Edge Enhancement + Detail (Weeks 7-8) 📋 PLANNED
**Steps**: 12 → 15 (Steps 15-17)
**Expected Gain**: +3-5% OCR accuracy (cumulative: +23-45%)

### Phase 4: Cleanup + Multi-Mode (Weeks 9-10) 📋 PLANNED
**Steps**: 15 → 21 (Steps 18-21)
**Pipeline Completion**: 100%

---

## Validation Framework

### VLM-Based Structured Assessment

#### Three Prompt Modes
1. **image_quality**: Baseline assessment (1-10 scoring, RGB estimates, angle detection)
2. **enhancement_validation**: Before/after Δ metrics with ✅/⚠️/❌ indicators
3. **preprocessing_diagnosis**: Root cause analysis for failures

#### Workflow
```bash
# Baseline
bash scripts/vlm_baseline_assessment.sh

# Validation after enhancement
bash scripts/vlm_validate_enhancement.sh phase1_bg_norm

# Aggregation
python scripts/aggregate_vlm_validations.py \
  --input vlm_reports/phase1_validation \
  --output summaries/phase1_vlm_summary.md
```

### OCR Accuracy Metrics
- Baseline: Measure accuracy per image on worst performers
- Incremental: Compare each enhancement vs. baseline
- Target: +15-30% improvement by Week 3

### Coordinate Alignment Validation
- Track transformation matrix composition: `M_total = M_perspective @ M_deskew @ M_bg`
- Validate coordinate mapping with <2px error threshold
- Visual overlay verification

---

## Success Criteria

### Phase 1 Completion (Week 3)
- ✅ OCR Accuracy: +15-30% from baseline
- ✅ Processing Time: <200ms total pipeline
- ✅ VLM Improvement: >+5 points average
- ✅ Coordinate Error: <2px
- ✅ Zero Predictions: Reduce by ≥10 images
- ✅ Success Rate: ≥80% pass rate on validation set

### Pipeline Quality Standards
- No preprocessing failures (100% processing success)
- Code reviewed and documented
- Integration tests passing
- Performance benchmarks documented

---

## Technical Foundation

### Existing Infrastructure (From Parent Experiment)
- ✅ **Perspective correction**: 100% success rate on 25 worst performers
- ✅ **Preprocessing pipeline**: Modular, coordinate-tracking (`preprocessing_pipeline.py`)
- ✅ **Test scripts**: Batch processing, worst-performer validation
- ✅ **VLM validation**: Structured prompts, CLI integration

### Pipeline Architecture
```python
# ocr/inference/preprocessing_pipeline.py
class PreprocessingPipeline:
    def process(image,
                enable_perspective=False,
                enable_background_norm=False,  # NEW - Phase 1
                enable_deskewing=False,        # NEW - Phase 1
                enable_grayscale=False):
        # Stage 0: Background normalization (NEW)
        # Stage 1: Perspective correction (EXISTING)
        # Stage 1.5: Text deskewing (NEW)
        # Stage 2: Grayscale (EXISTING - optional)
        # Stage 3-5: Resize, Pad, Normalize (EXISTING)
        return PreprocessingResult(...)
```

### Configuration
```yaml
# configs/preprocessing.yaml
preprocessing:
  enable_perspective: true
  enable_background_norm: true     # NEW
  bg_norm_method: "edge-based"     # NEW
  enable_deskewing: true           # NEW
  deskew_method: "hough"           # NEW
  enable_grayscale: false
```

---

## References

### Parent Experiment
- [20251129_173500_perspective_correction_implementation](../20251129_173500_perspective_correction_implementation/): 100% perspective correction success

### Code References
- [ocr/utils/perspective_correction.py](../../../../ocr/utils/perspective_correction.py): Perspective correction (461 lines)
- [ocr/inference/preprocessing_pipeline.py](../../../../ocr/inference/preprocessing_pipeline.py): Pipeline architecture (276 lines)
- [ocr/inference/preprocess.py](../../../../ocr/inference/preprocess.py): Transform helpers (154 lines)

### VLM Infrastructure
- [AgentQMS/vlm/prompts/markdown/](../../../../AgentQMS/vlm/prompts/markdown/): VLM prompts (8 modes total, 3 for image enhancements)
- [AgentQMS/vlm/cli/analyze_image_defects.py](../../../../AgentQMS/vlm/cli/analyze_image_defects.py): VLM CLI tool
- [AgentQMS/vlm/README.md](../../../../AgentQMS/vlm/README.md): VLM usage guide

#### Image Enhancement VLM Modes
1. **`image_quality`**: Baseline quality assessment (background tint, text slant, shadows, contrast)
2. **`enhancement_validation`**: Before/after preprocessing comparison
3. **`preprocessing_diagnosis`**: Failure root cause analysis

#### Quick Usage Examples
```bash
# Assess baseline image quality
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image data/zero_prediction_worst_performers/image_001.jpg \
  --mode image_quality \
  --output vlm_reports/baseline_image_001.md

# Validate enhancements (before/after comparison)
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image outputs/comparison_image_001.jpg \
  --mode enhancement_validation \
  --output vlm_reports/validation_image_001.md

# Diagnose preprocessing failures
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image outputs/failed_image.jpg \
  --mode preprocessing_diagnosis \
  --initial-description "White-balance applied but tint persists" \
  --output vlm_reports/diagnosis_image.md
```

#### Test VLM Integration
```bash
# Run VLM prompt functionality tests
uv run python scripts/test_vlm_prompts.py
```

### Standards
- [.ai-instructions/](../../../../.ai-instructions/): AgentQMS documentation standards

---

## Contact & Reporting

### Progress Updates
- Weekly reports: `summaries/week{N}_*.md`
- Phase completion: `artifacts/phase{N}_completion_report.md`
- Master roadmap updates: Track in `20251217_0243_assessment_master-roadmap.md`

### Issue Tracking
- VLM diagnosis reports: `vlm_reports/debugging/`
- Failure analysis: Document in weekly reports
- Coordinate alignment issues: Run `scripts/validate_coordinates.py`

---

**Last Updated**: 2025-12-18 05:30 KST
**Next Action**: Run baseline VLM assessment (Week 1 Day 1)
