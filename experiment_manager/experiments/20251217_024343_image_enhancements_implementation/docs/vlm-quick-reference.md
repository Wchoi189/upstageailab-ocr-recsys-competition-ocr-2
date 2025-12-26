# VLM Quick Reference - Image Enhancement Experiment

## 🎯 Available Modes

| Mode | Purpose | When to Use |
|------|---------|-------------|
| `image_quality` | Baseline quality assessment | Document initial problems in worst performers |
| `enhancement_validation` | Before/after comparison | Measure preprocessing effectiveness |
| `preprocessing_diagnosis` | Failure analysis | Debug why preprocessing didn't work |

## 🚀 Quick Commands

### 1. Assess Baseline Quality
```bash
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image data/zero_prediction_worst_performers/image_001.jpg \
  --mode image_quality \
  --output vlm_reports/baseline/image_001.md
```

**Output includes:**
- Background tint severity (1-10 scale)
- Text slant angle (degrees)
- Shadow presence and severity
- Contrast and readability scores
- Priority-ranked enhancement recommendations

### 2. Validate Enhancements
```bash
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image outputs/comparisons/image_001_before_after.jpg \
  --mode enhancement_validation \
  --output vlm_reports/phase1_validation/image_001.md
```

**Output includes:**
- Before/after metrics for each quality dimension
- Success assessment (achieved goals?)
- Remaining issues
- Deployment recommendation (Deploy/Tune/Reject)

### 3. Diagnose Failures
```bash
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image outputs/failed_preprocessing_image_005.jpg \
  --mode preprocessing_diagnosis \
  --initial-description "White-balance (gray-world) applied. Expected neutral white, got cream tint. Params: percentile=5" \
  --output vlm_reports/debugging/image_005_diagnosis.md
```

**Output includes:**
- Root cause hypotheses (ranked by likelihood)
- Quantitative evidence from the image
- Remediation strategies (code changes, parameter tuning, alternatives)

## 🔬 Batch Processing Scripts

### Baseline Assessment (All Worst Performers)
```bash
#!/bin/bash
for img in data/zero_prediction_worst_performers/*.jpg; do
  basename=$(basename "$img" .jpg)
  uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
    --image "$img" \
    --mode image_quality \
    --backend openrouter \
    --output vlm_reports/baseline/${basename}_quality.md
  echo "✅ Analyzed: $basename"
done
```

### Validation Assessment (All Comparisons)
```bash
#!/bin/bash
for cmp in outputs/comparisons/*.jpg; do
  basename=$(basename "$cmp" .jpg)
  uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
    --image "$cmp" \
    --mode enhancement_validation \
    --backend openrouter \
    --output vlm_reports/phase1_validation/${basename}_validation.md
  echo "✅ Validated: $basename"
done
```

## 📊 Output Structure

```
vlm_reports/
├── baseline/                     # image_quality mode outputs
│   ├── image_001_quality.md
│   ├── image_002_quality.md
│   └── ...
├── phase1_validation/           # enhancement_validation mode outputs
│   ├── image_001_validation.md
│   └── ...
└── debugging/                    # preprocessing_diagnosis mode outputs
    ├── image_005_diagnosis.md
    └── ...
```

## 🧪 Testing

Run functionality tests on all three modes:
```bash
cd experiment-tracker/experiments/20251217_024343_image_enhancements_implementation
uv run python scripts/test_vlm_prompts.py
```

Expected output:
```
✅ PASS: image_quality
✅ PASS: enhancement_validation
✅ PASS: preprocessing_diagnosis

🎉 All tests PASSED!
```

## 🛠️ Backend Options

| Backend | Model | Quality | Speed | Cost |
|---------|-------|---------|-------|------|
| `openrouter` (default) | Qwen-2-VL-72B | ⭐⭐⭐⭐⭐ | Medium | $ |
| `solar_pro2` | Solar-Pro2 | ⭐⭐⭐⭐ | Fast | $ |
| `cli` | Local Qwen | ⭐⭐⭐ | Slow | Free |

Use `--backend` flag to override:
```bash
--backend solar_pro2  # Faster alternative
--backend cli         # Local, no API costs
```

## 📝 Tips

1. **Use descriptive output filenames**: Include image ID and analysis type
2. **Provide context for diagnosis mode**: Use `--initial-description` to give preprocessing details
3. **Batch processing**: Save VLM costs by batching related images together
4. **Review outputs**: VLM reports are starting points - validate findings manually
5. **Aggregate results**: Create summary reports from multiple VLM analyses

## 🔗 Related Documentation

- [VLM Module README](../../../../AgentQMS/vlm/README.md)
- [VLM Integration Fixes](vlm-integration-fixes.md)
- [Experiment README](README.md)
- [Prompt Templates](../../../../AgentQMS/vlm/prompts/markdown/)

---

**Quick Help**: `uv run python -m AgentQMS.vlm.cli.analyze_image_defects --help`
