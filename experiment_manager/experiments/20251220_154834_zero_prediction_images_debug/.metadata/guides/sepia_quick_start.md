---
ads_version: '1.0'
type: guide
experiment_id: 20251220_154834_zero_prediction_images_debug
title: Sepia Testing Quick Start
created: '2025-12-21T02:23:00+09:00'
tags:
- sepia
- quickstart
status: complete
updated: '2025-12-21T02:23:00+09:00'
commands: []
prerequisites: []
---
# Sepia Testing Quick Start

## Directory Structure

```
20251220_154834_zero_prediction_images_debug/
├── scripts/
│   ├── sepia_enhancement.py          # 5 sepia methods (classic, adaptive, warm, clahe, linear_contrast)
│   ├── compare_sepia_methods.py      # Comparison framework (includes artifacts validation)
│   ├── sepia_perspective_pipeline.py # Full pipeline (sepia + perspective)
│   └── vlm_validate_sepia.sh         # VLM quality assessment
├── artifacts/
│   └── reference_images/             # Test samples
│       ├── 000712.jpg
│       ├── 000732.jpg
│       └── 000732_REMBG.jpg
└── outputs/
    ├── sepia_tests/                  # Isolated method results
    ├── sepia_comparison/             # Comparison grids
    ├── sepia_pipeline/               # Pipeline results
    └── sepia_vlm_reports/            # VLM assessments
```

## Quick Test

**Run isolated sepia testing**:
```bash
cd scripts/
python sepia_enhancement.py \
  --input ../artifacts/reference_images/drp.en_ko.in_house.selectstar_000732_REMBG.jpg \
  --method all \
  --output ../outputs/sepia_tests/
```

**Expected output**:
- `*_sepia_classic.jpg`
- `*_sepia_adaptive.jpg`
- `*_sepia_warm.jpg`
- `*_sepia_clahe.jpg`
- `*_sepia_linear_contrast.jpg`
- `*_metrics.json`

## Method Comparison

```bash
python compare_sepia_methods.py \
  --input ../artifacts/reference_images/drp.en_ko.in_house.selectstar_000732_REMBG.jpg \
  --output ../outputs/sepia_comparison/ \
  --save-metrics
```

**Comparison includes**:
1. Raw image
2. Grayscale
3. Gray-world normalization
4. Sepia Classic
5. Sepia Adaptive
6. Sepia Warm
7. Sepia CLAHE (Adaptive Contrast)
8. Sepia Linear Contrast (Global Gain)

## Full Pipeline Test

```bash
python sepia_perspective_pipeline.py \
  --input ../artifacts/reference_images/drp.en_ko.in_house.selectstar_000732.jpg \
  --sepia-method clahe \
  --output ../outputs/sepia_pipeline/
```

## VLM Validation

```bash
export DASHSCOPE_API_KEY="your_key"
./vlm_validate_sepia.sh ../outputs/sepia_comparison/
```

## Troubleshooting

**Missing background_normalization.py**:
```bash
# Copy from project root
cp ../../../ocr/datasets/background_normalization.py scripts/
```

**Permission denied on vlm_validate_sepia.sh**:
```bash
chmod +x scripts/vlm_validate_sepia.sh
```

## Next Steps

1. Run isolated testing → verify all methods work
2. Run comparison → identify best sepia method
3. Run pipeline → test integration
4. VLM validation → confirm visual quality
5. OCR inference → measure accuracy improvements
