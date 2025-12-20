# Sepia Testing - Quick Commands

## Location
```bash
cd experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/scripts
```

## 1. Test Single Image with All Sepia Methods
```bash
python sepia_enhancement.py \
  --input /path/to/image.jpg \
  --method all \
  --output ../outputs/sepia_tests/
```

## 2. Compare Sepia vs Gray-scale/Normalization
```bash
python compare_sepia_methods.py \
  --input /path/to/drp.en_ko.in_house.selectstar_000732.jpg \
  --output ../outputs/sepia_comparison/ \
  --save-metrics
```

## 3. Full Pipeline (Sepia + Perspective Correction)
```bash
# Recommended: warm sepia method
python sepia_perspective_pipeline.py \
  --input /path/to/image.jpg \
  --sepia-method warm \
  --output ../outputs/sepia_pipeline/ \
  --save-metrics
```

## 4. VLM Validation
```bash
export DASHSCOPE_API_KEY='your_key'
./vlm_validate_sepia.sh ../outputs/sepia_comparison/
```

## Test on Reference Samples

### Problematic Image
```bash
# drp.en_ko.in_house.selectstar_000732
python compare_sepia_methods.py \
  --input /path/to/drp.en_ko.in_house.selectstar_000732.jpg \
  --output ../outputs/ref_000732/ \
  --save-metrics
```

### Target Quality Reference
```bash
# drp.en_ko.in_house.selectstar_000712_sepia.jpg
python compare_sepia_methods.py \
  --input /path/to/drp.en_ko.in_house.selectstar_000712_sepia.jpg \
  --output ../outputs/ref_000712/ \
  --save-metrics
```

## Batch Processing
```bash
# Process entire directory
python sepia_perspective_pipeline.py \
  --input /path/to/images_directory/ \
  --sepia-method warm \
  --output ../outputs/batch_results/ \
  --save-metrics
```

## Method Options

- `classic` - Traditional sepia tone
- `adaptive` - Adaptive intensity-based sepia
- `warm` - ⭐ Enhanced warm tones (recommended)
- `contrast` - Sepia + CLAHE enhancement
- `all` - Test all methods

## Output Files

Each run generates:
- Enhanced images: `*_sepia_<method>.jpg`
- Comparison grid: `*_comparison_grid.jpg`
- Metrics: `*_metrics.json`
- VLM validation: `*_vlm_validation.json`

## Next Steps After Testing

1. Review comparison grids in `outputs/sepia_comparison/`
2. Check metrics JSON for quantitative results
3. Run VLM validation for visual quality assessment
4. Test best method with OCR checkpoint
5. Update experiment state with findings
