# Issues Fixed and Recommendations

**Date**: 2025-12-18
**Context**: VLM validation errors and naming convention violations

---

## ✅ Issues Fixed

### 1. Filename Convention Violation

**Problem**: `docs/EDS_COMPLIANCE_REPORT.md` used ALL CAPS naming

**Why it's wrong**:
- Project convention: lowercase-with-dashes
- ALL CAPS deprecated (see `/docs/artifacts/DEPRECATED-ALLCAPS-DOCS/`)
- AgentQMS naming standard: `{type}_{slug}.md` with lowercase slugs

**Fix Applied**:
```bash
mv docs/EDS_COMPLIANCE_REPORT.md docs/eds-compliance-report.md
```

**Convention Reference**:
- Tool catalog: `.ai-instructions/tier2-framework/tool-catalog.yaml`
- All artifact commands use lowercase slugs with dashes
- Example: `20251218_1415_report_baseline-metrics-summary.md` ✅
- Anti-pattern: `EDS_COMPLIANCE_REPORT.md` ❌

---

## ⚠️ VLM Backend Content Refusal Issue

### Problem Description

VLM calls to OpenRouter are being refused with content policy errors when analyzing receipt/invoice images:

```bash
# Failing command
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image data/zero_prediction_worst_performers/drp.en_ko.in_house.selectstar_000699.jpg \
  --mode image_quality \
  --backend openrouter \
  --output vlm_reports/baseline/000699_baseline_quality.md
```

### Root Cause

OpenRouter's content policy filters may be rejecting receipt/invoice images due to:
1. **Sensitive information**: Real financial documents contain PII, amounts, vendor info
2. **Model-specific policies**: Qwen-2-VL-72B may have stricter content filters
3. **Image classification**: Document scans might trigger financial/legal content filters

### Workarounds

#### Option 1: Use Alternative Backend (Recommended)

```bash
# Try Solar Pro 2 (may have different content policies)
--backend solar_pro2

# Or use local CLI backend (no content filtering)
--backend cli
```

#### Option 2: Sanitize Test Images

Before VLM analysis:
1. Redact sensitive information (amounts, names, addresses)
2. Use synthetic/mock receipts for testing
3. Apply blur to sensitive regions

#### Option 3: Manual Quality Assessment

For blocked images, document quality manually:

```markdown
# Manual Quality Assessment

**Image**: drp.en_ko.in_house.selectstar_000699.jpg

## Baseline Quality Metrics
- Tint Severity: 7/10 (cream/yellow background)
- Slant Angle: ~5° clockwise
- Shadow Severity: 3/10 (minimal)
- Contrast: 6/10 (adequate)
- Noise: 4/10 (moderate JPEG artifacts)

## Priority Enhancements
1. Background normalization (gray-world)
2. Text deskewing (projection profile)
3. Contrast enhancement (CLAHE)
```

#### Option 4: Use Image Quality Mode on Enhanced Images Only

Skip baseline VLM assessment and only validate enhancements:

```bash
# This works because enhanced images have sanitized backgrounds
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image outputs/bg_norm_gray_world/enhanced_image_001.jpg \
  --mode image_quality \
  --backend openrouter
```

---

## 📋 Naming Convention Reference

### Correct Patterns

| Type | Pattern | Example |
|------|---------|---------|
| Report | `YYYYMMDD_HHMM_report_{slug}.md` | `20251218_1415_report_baseline-metrics-summary.md` |
| Assessment | `YYYYMMDD_HHMM_assessment_{slug}.md` | `20251217_0243_assessment_master-roadmap.md` |
| Guide | `YYYYMMDD_HHMM_guide_{slug}.md` | `20251217_0243_guide_vlm-integration-guide.md` |
| Data artifact | `YYYYMMDD_HHMM_{slug}.json` | `20251218_1415_baseline-quality-metrics.json` |
| General docs | `{slug}.md` | `vlm-integration-fixes.md`, `vlm-quick-reference.md` |

### Incorrect Patterns ❌

- `EDS_COMPLIANCE_REPORT.md` (ALL CAPS)
- `VLM_Integration_Fixes.md` (mixed case with underscores)
- `20251218-Report-Baseline.md` (title case in slug)
- `Baseline_Metrics.json` (underscores in slug, title case)

### Rules

1. **Timestamps**: `YYYYMMDD_HHMM_` prefix for dated artifacts
2. **Slugs**: lowercase with dashes (kebab-case)
3. **Type prefix**: After timestamp, before slug (`report_`, `assessment_`, etc.)
4. **No underscores** in slugs (only in type prefix and timestamp separator)
5. **No capital letters** in slugs

---

## 🛠️ Recommended Actions

### Immediate

1. ✅ **Fixed**: Renamed `EDS_COMPLIANCE_REPORT.md` → `eds-compliance-report.md`
2. ⏳ **TODO**: Try alternative VLM backends (`solar_pro2` or `cli`)
3. ⏳ **TODO**: If VLM still fails, proceed with manual quality assessment

### Short-term

1. Create sanitized test dataset for VLM validation
2. Document manual quality metrics for baseline
3. Use VLM only on enhanced images (post-processing)
4. Add VLM backend fallback logic to validation scripts

### Long-term

1. Investigate OpenRouter content policy specifics
2. Consider self-hosted VLM for sensitive document analysis
3. Add pre-processing to auto-redact sensitive info before VLM
4. Create synthetic receipt dataset for testing

---

## 📚 Related Documentation

- [AgentQMS Tool Catalog](/.ai-instructions/tier2-framework/tool-catalog.yaml) - Naming conventions
- [VLM Integration Fixes](vlm-integration-fixes.md) - VLM mode additions
- [VLM Quick Reference](vlm-quick-reference.md) - Usage examples
- [EDS Compliance Report](eds-compliance-report.md) - Experiment structure

---

## 🎯 Next Steps

### For VLM Validation

```bash
# Option 1: Try alternative backend
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image outputs/bg_norm_gray_world/enhanced_001.jpg \
  --mode image_quality \
  --backend cli \
  --output vlm_reports/baseline/001_quality.md

# Option 2: Manual assessment (if VLM blocked)
# Document metrics in vlm_reports/baseline/manual_assessment.md

# Option 3: Focus on enhancement validation only
for cmp in outputs/*/comparison_*.jpg; do
  basename=$(basename "$cmp" .jpg | sed 's/comparison_//')
  uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
    --image "$cmp" \
    --mode enhancement_validation \
    --backend cli \
    --output vlm_reports/phase1_validation/${basename}_validation.md
done
```

### Verification

```bash
# Check all filenames follow conventions
find experiment-tracker/experiments/20251217_024343_image_enhancements_implementation \
  -name "*.md" -o -name "*.json" | grep -E '[A-Z_]{2,}'

# Should return nothing if all files follow lowercase-with-dashes
```

---

**Status**: Naming issues fixed ✅ | VLM issues documented ⚠️
**Recommendation**: Proceed with manual quality assessment or alternative VLM backend
