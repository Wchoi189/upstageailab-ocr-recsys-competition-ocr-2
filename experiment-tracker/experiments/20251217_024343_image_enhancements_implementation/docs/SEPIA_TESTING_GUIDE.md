# Sepia Enhancement Testing Guide

**Experiment ID**: `20251217_024343_image_enhancements_implementation`
**Phase**: Week 2 Day 4-5
**Created**: 2025-12-21
**Status**: Testing In Progress

---

## Overview

Testing sepia tone enhancement as a superior alternative to gray-scale conversion and gray-world normalization for OCR preprocessing. User observations indicate that sepia provides more reliable OCR results than both methods combined.

### Hypothesis

**Sepia + Perspective Correction = Optimal Enhancement Pipeline**

Without advanced Office Lens-style processing, sepia combined with perspective correction provides the best OCR results.

### Reference Samples

- **Problematic**: `drp.en_ko.in_house.selectstar_000732` (current poor OCR results)
- **Target Quality**: `drp.en_ko.in_house.selectstar_000712_sepia.jpg` (desired output)

---

## Implementation

### Sepia Methods

Four sepia enhancement methods implemented:

1. **Classic Sepia** (`classic`)
   - Traditional sepia transformation matrix
   - Standard warm tone conversion
   - Fast processing (~50ms)

2. **Adaptive Sepia** (`adaptive`)
   - Intensity-based sepia strength
   - Preserves extreme darks and brights
   - Better detail retention

3. **Warm Sepia** (`warm`)
   - Enhanced warm tones for OCR
   - Stronger red/yellow channels
   - Reduces cold/blue tints

4. **Sepia + Contrast** (`contrast`)
   - Combines warm sepia with CLAHE
   - Enhanced text clarity
   - Best for low-contrast documents

### Scripts Created

```bash
experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/scripts/
├── sepia_enhancement.py              # Core sepia methods
├── compare_sepia_methods.py          # Isolated comparison vs alternatives
├── sepia_perspective_pipeline.py     # Full pipeline integration
└── vlm_validate_sepia.sh             # VLM validation
```

---

## Quick Start

### Step 1: Isolated Sepia Testing

Test all sepia methods on a single image:

```bash
cd experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/scripts

# Test all sepia methods
python sepia_enhancement.py \
  --input /path/to/drp.en_ko.in_house.selectstar_000732.jpg \
  --method all \
  --output ../outputs/sepia_tests/
```

This generates 4 sepia-enhanced images with metrics.

### Step 2: Comparison Against Alternatives

Compare sepia against gray-scale and normalization:

```bash
# Generate comparison grid
python compare_sepia_methods.py \
  --input /path/to/drp.en_ko.in_house.selectstar_000732.jpg \
  --output ../outputs/sepia_comparison/ \
  --save-metrics

# Output:
#  - Comparison grid image showing all 7 methods
#  - Individual enhanced images
#  - Metrics JSON file
```

Methods compared:
- Raw (baseline)
- Grayscale
- Gray-world normalization
- Sepia (classic, adaptive, warm, contrast)

### Step 3: Full Pipeline Testing

Test sepia with perspective correction:

```bash
# Sepia + perspective correction (recommended: warm method)
python sepia_perspective_pipeline.py \
  --input /path/to/drp.en_ko.in_house.selectstar_000732.jpg \
  --sepia-method warm \
  --output ../outputs/sepia_pipeline/ \
  --save-metrics

# Optional: Include deskewing (if needed)
python sepia_perspective_pipeline.py \
  --input /path/to/image.jpg \
  --sepia-method contrast \
  --deskew \
  --output ../outputs/sepia_pipeline_deskew/
```

Pipeline stages:
1. Perspective correction
2. Sepia enhancement
3. (Optional) Deskewing

### Step 4: VLM Validation

Validate results using VLM:

```bash
# Validate comparison grid
export DASHSCOPE_API_KEY='your_api_key'

./vlm_validate_sepia.sh ../outputs/sepia_comparison/comparison_grid.jpg

# Or validate entire directory
./vlm_validate_sepia.sh ../outputs/sepia_comparison/
```

VLM evaluates:
- Text clarity (1-10)
- Background quality (1-10)
- Color tint impact (1-10)
- OCR suitability (1-10)
- Ranking and recommendations

### Step 5: OCR End-to-End Validation

Test with actual OCR model:

```bash
# Use existing checkpoint
CHECKPOINT="outputs/experiments/train/ocr/pan_resnet18_add_polygons_canonical/20241019_0033_00/checkpoints/epoch-18_step-001957.ckpt"

# Run OCR on enhanced images
# (Use your existing OCR inference script here)
python runners/predict.py \
  --checkpoint $CHECKPOINT \
  --images ../experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/sepia_pipeline/ \
  --output ../outputs/sepia_ocr_results/
```

---

## Expected Results

### Metrics to Track

1. **Color Tint Score**
   - Target: < 20 (lower = better)
   - Baseline: 58.1 (gray-world achieved 14.6)

2. **Contrast**
   - Target: Maintain or improve
   - Watch for over-enhancement

3. **Edge Strength**
   - Indicator of text clarity
   - Higher = better text definition

4. **Processing Time**
   - Target: < 100ms per image
   - For pipeline viability

### Success Criteria

**Sepia enhancement is superior if:**

✅ Better OCR prediction accuracy than gray-scale
✅ More reliable results than gray-world normalization
✅ Comparable or faster processing time
✅ VLM validation score > 4.5/5
✅ Improved metrics on reference samples

---

## Directory Structure

```
experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/
├── scripts/
│   ├── sepia_enhancement.py
│   ├── compare_sepia_methods.py
│   ├── sepia_perspective_pipeline.py
│   └── vlm_validate_sepia.sh
├── outputs/
│   ├── sepia_tests/              # Isolated sepia tests
│   ├── sepia_comparison/         # Comparison results
│   ├── sepia_pipeline/           # Full pipeline outputs
│   └── sepia_vlm_reports/        # VLM validation results
├── docs/
│   └── SEPIA_TESTING_GUIDE.md   # This file
└── state.yml                     # Experiment state tracking
```

---

## Next Steps

### Testing Plan

1. **Isolated Testing** ✓ (scripts created)
   - Test all 4 sepia methods on reference samples
   - Generate metrics and comparison data

2. **Comparison Analysis** (pending)
   - Run comparison script on problematic images
   - Analyze metrics against gray-scale/normalization

3. **Pipeline Validation** (pending)
   - Test sepia + perspective correction
   - Compare against current pipeline

4. **VLM Validation** (pending)
   - Visual quality assessment
   - Ranking and recommendations

5. **OCR Testing** (pending)
   - End-to-end with epoch-18 checkpoint
   - Compare prediction accuracy

### Decision Points

After testing, decide:

1. **Best Sepia Method**: classic | adaptive | warm | contrast
2. **Pipeline Integration**: Replace gray-world or keep both?
3. **Deskewing**: Include or exclude based on OCR results?

### Integration Path

If sepia proves superior:

1. Update preprocessing pipeline configuration
2. Integrate sepia method into inference module
3. Update documentation and guides
4. Run full validation on test set
5. Update experiment state and close tasks

---

## Troubleshooting

### Common Issues

**Import errors**:
```bash
# Make sure you're in the scripts directory
cd experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/scripts
```

**Missing dependencies**:
```bash
# Install required packages
pip install opencv-python numpy
```

**VLM validation fails**:
```bash
# Check API key
echo $DASHSCOPE_API_KEY

# Test with curl
curl -X POST "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation" \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -H "Content-Type: application/json"
```

---

## References

- **Experiment State**: [state.yml](../state.yml)
- **Background Normalization Results**: Phase 1 Week 1 (4.75/5 score)
- **Deskewing Results**: Phase 1 Week 2 (4.6/5 score, excluded from pipeline)
- **Checkpoint**: epoch-18_step-001957.ckpt (97% hmean)

---

## Contact

For questions or issues:
- Review experiment state: `state.yml`
- Check VLM reports: `outputs/sepia_vlm_reports/`
- See comparison metrics: `outputs/sepia_comparison/*_metrics.json`
