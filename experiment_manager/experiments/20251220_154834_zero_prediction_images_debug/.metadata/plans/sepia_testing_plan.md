---
ads_version: "1.0"
type: implementation_plan
experiment_id: "20251220_154834_zero_prediction_images_debug"
title: "Sepia Enhancement Testing Plan"
created: "2025-12-21T02:23:00+09:00"
updated: "2025-12-21T03:25:00+09:00"
status: "completed"
priority: "high"
tags: ["sepia", "image-enhancement", "ocr-preprocessing"]
---

# Sepia Enhancement Testing Plan

## Objective

Test sepia color transformation as alternative to gray-scale/gray-world normalization for OCR preprocessing.

## Hypothesis

Sepia enhancement provides more reliable OCR results than gray-scale and gray-world normalization for problematic document images (000712, 000732).

## Test Workflow

### Phase 1: Isolated Method Testing
**Script**: `sepia_enhancement.py`
**Target**: Test all 5 sepia methods independently (classic, adaptive, warm, clahe, linear_contrast)
**Samples**:
- `drp.en_ko.in_house.selectstar_000712.jpg`
- `drp.en_ko.in_house.selectstar_000732.jpg`
- `drp.en_ko.in_house.selectstar_000732_REMBG.jpg`

**Commands**:
```bash
cd scripts/
python sepia_enhancement.py --input ../artifacts/reference_images/drp.en_ko.in_house.selectstar_000732_REMBG.jpg --method all --output ../outputs/sepia_tests/
python sepia_enhancement.py --input ../artifacts/reference_images/ --method all --output ../outputs/sepia_tests/
```

**Metrics**:
- Sepia tint intensity
- Contrast ratio
- Brightness level
- Edge strength

### Phase 2: Comparative Analysis
**Script**: `compare_sepia_methods.py`
**Target**: Compare sepia vs gray-scale vs gray-world
**Output**: Comparison grids + metrics JSON

**Commands**:
```bash
python compare_sepia_methods.py --input ../artifacts/reference_images/drp.en_ko.in_house.selectstar_000732_REMBG.jpg --output ../outputs/sepia_comparison/ --save-metrics
```

**Expected Output**:
- 7-method comparison grid
- Quantitative metrics per method
- Visual quality assessment baseline

### Phase 3: Pipeline Integration
**Script**: `sepia_perspective_pipeline.py`
**Target**: Test sepia + perspective correction

**Commands**:
```bash
python sepia_perspective_pipeline.py --input ../artifacts/reference_images/drp.en_ko.in_house.selectstar_000732.jpg --sepia-method clahe --output ../outputs/sepia_pipeline/
python sepia_perspective_pipeline.py --input ../artifacts/reference_images/drp.en_ko.in_house.selectstar_000732.jpg --sepia-method linear_contrast --output ../outputs/sepia_pipeline/
```

### Phase 4: VLM Validation
**Script**: `vlm_validate_sepia.sh`
**Target**: Use Qwen3 VL Plus for visual quality assessment

**Commands**:
```bash
export DASHSCOPE_API_KEY="your_key"
./vlm_validate_sepia.sh ../outputs/sepia_comparison/
```

### Phase 5: OCR End-to-End Testing
**Target**: Compare OCR accuracy with sepia-enhanced images
**Checkpoint**: `epoch-18` (existing model)

**Process**:
1. Run OCR inference on sepia-enhanced images
2. Compare predictions against gray-world baseline
3. Measure accuracy improvements on 000712/000732 samples

## Success Criteria

- ✅ All 5 sepia methods execute without errors
- ✅ Comparison grids generated successfully
- ✅ VLM reports confirm visual quality improvements
- ✅ OCR predictions improve on problematic samples

## Dependencies

**Python packages**: opencv-python, numpy
**API keys**: DASHSCOPE_API_KEY (for VLM validation)
**Model checkpoint**: epoch-18 (existing)

## Timeline

- Phase 1-2: 1 hour
- Phase 3: 30 minutes
- Phase 4: 1 hour (API calls)
- Phase 5: 2 hours (OCR inference)

**Total estimated time**: 4-5 hours
