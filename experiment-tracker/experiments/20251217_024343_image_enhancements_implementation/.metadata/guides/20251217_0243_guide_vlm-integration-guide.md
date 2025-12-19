---
ads_version: "1.0"
type: "guide"
experiment_id: "20251217_024343_image_enhancements_implementation"
status: "complete"
created: "2025-12-17T17:59:48Z"
updated: "2025-12-17T17:59:48Z"
tags: ['image-enhancements', 'guide']
commands: []
prerequisites: []
---

# VLM Integration Guide for Image Enhancement Experiment

**Experiment**: `20251217_024343_image_enhancements_implementation`
**Purpose**: Integrate VLM-based image quality assessment into preprocessing validation workflow

---

## Overview

The VLM (Vision-Language Model) tool provides **structured, objective analysis** of document image quality issues and preprocessing effectiveness. It complements manual inspection and quantitative metrics with detailed technical assessments in natural language.

### Key Benefits
- ✅ **Objective validation**: Third-party assessment of preprocessing claims
- ✅ **Systematic documentation**: Structured reports for every test image
- ✅ **Hypothesis generation**: Identifies issues you might have missed
- ✅ **Quantitative scoring**: 1-10 scales for aggregation and comparison
- ✅ **Debugging insights**: Root cause analysis for preprocessing failures

---

## Setup

### 1. Verify VLM Tool Installation

```bash
# Test VLM CLI is available
uv run python -m AgentQMS.vlm.cli.analyze_defects --help

# Verify OpenRouter backend (default)
cat AgentQMS/vlm/config.yaml | grep -A5 backends
```

### 2. Create Experiment Directories

```bash
cd experiment-tracker/experiments/20251217_024343_image_enhancements_implementation

# Create VLM report directories
mkdir -p vlm_reports/baseline
mkdir -p vlm_reports/phase1_validation
mkdir -p vlm_reports/phase2_validation
mkdir -p vlm_reports/debugging

# Create comparison image directory
mkdir -p outputs/comparisons
```

### 3. Verify Prompts Exist

```bash
ls -la AgentQMS/vlm/prompts/markdown/

# Should show:
# - image_quality_analysis.md
# - enhancement_validation.md
# - preprocessing_diagnosis.md
```

---

## Usage Workflows

### Workflow 1: Baseline Assessment (Before Phase 1)

**Goal**: Document quality issues in worst performers before implementing fixes

#### Step 1: Identify Test Images
```bash
# Assuming you have worst performers in a directory
TEST_IMAGES_DIR="data/zero_prediction_worst_performers"
```

#### Step 2: Run Batch Assessment
```bash
# Process first 10 images for quick assessment
for img in $(ls $TEST_IMAGES_DIR/*.jpg | head -10); do
  basename=$(basename "$img" .jpg)
  echo "Analyzing baseline: $basename"

  uv run python -m AgentQMS.vlm.cli.analyze_defects \
    --image "$img" \
    --mode image_quality \
    --backend openrouter \
    --output-format markdown \
    --output "experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/vlm_reports/baseline/${basename}_quality.md"
done
```

#### Step 3: Review Reports
```bash
# Quick scan of all reports
cat vlm_reports/baseline/*_quality.md | grep "Overall Quality Score"
cat vlm_reports/baseline/*_quality.md | grep "Critical Issues"
```

#### Step 4: Aggregate Priorities
```bash
# Extract priority recommendations
cat vlm_reports/baseline/*_quality.md | grep -A10 "Priority Ranking"
```

**Expected Output**: 10 detailed quality assessment reports identifying tint severity, slant angles, shadow regions, contrast scores.

---

### Workflow 2: Enhancement Validation (After Phase 1)

**Goal**: Measure effectiveness of background normalization + deskewing

#### Step 1: Create Before/After Comparisons

```python
# Script: scripts/create_before_after_comparison.py
import cv2
import numpy as np
from pathlib import Path

def create_comparison(before_path, after_path, output_path):
    """Create side-by-side comparison image."""
    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path))

    # Resize to same height
    h = max(before.shape[0], after.shape[0])
    before_resized = cv2.resize(before, (int(before.shape[1] * h / before.shape[0]), h))
    after_resized = cv2.resize(after, (int(after.shape[1] * h / after.shape[0]), h))

    # Concatenate horizontally with separator
    separator = np.ones((h, 20, 3), dtype=np.uint8) * 128
    comparison = np.hstack([before_resized, separator, after_resized])

    # Add labels
    cv2.putText(comparison, "BEFORE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 0, 255), 3)
    cv2.putText(comparison, "AFTER", (before_resized.shape[1] + 70, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imwrite(str(output_path), comparison)

# Usage
baseline_dir = Path("data/zero_prediction_worst_performers")
enhanced_dir = Path("artifacts/phase1_enhanced")
output_dir = Path("outputs/comparisons")

for before_img in baseline_dir.glob("*.jpg"):
    after_img = enhanced_dir / before_img.name
    if after_img.exists():
        output_img = output_dir / f"comparison_{before_img.stem}.jpg"
        create_comparison(before_img, after_img, output_img)
```

#### Step 2: Run VLM Validation
```bash
# Analyze each comparison
for cmp in outputs/comparisons/*.jpg; do
  basename=$(basename "$cmp" .jpg | sed 's/comparison_//')
  echo "Validating: $basename"

  uv run python -m AgentQMS.vlm.cli.analyze_defects \
    --image "$cmp" \
    --mode enhancement_validation \
    --backend openrouter \
    --output-format markdown \
    --output "vlm_reports/phase1_validation/${basename}_validation.md"
done
```

#### Step 3: Extract Success Metrics
```bash
# Aggregate validation results
python scripts/aggregate_vlm_validations.py \
  --input vlm_reports/phase1_validation/ \
  --output summaries/phase1_vlm_summary.md
```

**Expected Output**: Validation reports showing quantitative improvements (background uniformity, text alignment, contrast) with ✅/⚠️/❌ success indicators.

---

### Workflow 3: Preprocessing Debugging (When Things Fail)

**Goal**: Understand why preprocessing failed on specific images

#### Step 1: Identify Failure Cases
```bash
# Review validation reports for ❌ failures
grep -l "Overall Success.*Fail" vlm_reports/phase1_validation/*.md
```

#### Step 2: Run Diagnostic Analysis
```bash
# Deep-dive analysis on failure cases
FAILED_IMAGE="image_005"

uv run python -m AgentQMS.vlm.cli.analyze_defects \
  --image "outputs/comparisons/comparison_${FAILED_IMAGE}.jpg" \
  --mode preprocessing_diagnosis \
  --backend openrouter \
  --output-format markdown \
  --output "vlm_reports/debugging/${FAILED_IMAGE}_diagnosis.md" \
  --context "Applied white-balance (gray-world, scale factors: B=1.1, G=1.0, R=0.9) and deskewing (detected angle: +7°, rotated CCW). Result shows over-correction with blue tint and text blur."
```

#### Step 3: Review Root Cause Analysis
```bash
cat vlm_reports/debugging/${FAILED_IMAGE}_diagnosis.md | grep -A20 "Root Cause"
cat vlm_reports/debugging/${FAILED_IMAGE}_diagnosis.md | grep -A10 "Remediation"
```

**Expected Output**: Technical diagnosis with root cause hypotheses, parameter recommendations, and alternative approaches.

---

## Prompt Modes

### Mode: `image_quality`
**Use**: Baseline assessment before preprocessing
**Prompt**: `AgentQMS/vlm/prompts/markdown/image_quality_analysis.md`
**Output**: Quality scores (1-10) for background, alignment, contrast, shadows, noise
**Key Sections**:
- Background assessment (tint severity, uniformity)
- Text orientation (slant angle, alignment quality)
- Shadow analysis (presence, severity, regions)
- Priority ranking of preprocessing techniques

### Mode: `enhancement_validation`
**Use**: Before/after comparison after preprocessing
**Prompt**: `AgentQMS/vlm/prompts/markdown/enhancement_validation.md`
**Output**: Improvement metrics (Δ scores), success criteria (✅/⚠️/❌)
**Key Sections**:
- Quantitative improvement tables
- Success criteria checklist
- Remaining issues
- Parameter tuning recommendations

### Mode: `preprocessing_diagnosis`
**Use**: Debugging preprocessing failures
**Prompt**: `AgentQMS/vlm/prompts/markdown/preprocessing_diagnosis.md`
**Output**: Root cause hypotheses, remediation strategies
**Key Sections**:
- Failure characterization
- Root cause analysis (with likelihood rankings)
- Parameter sensitivity analysis
- Alternative approaches

---

## Integration into Experiment Phases

### Phase 1 (Weeks 1-3): Background Normalization + Deskewing

#### Week 1 Day 1: Baseline Assessment
```bash
# Before implementing anything
./scripts/vlm_baseline_assessment.sh

# Expected: 10 quality reports identifying tinted backgrounds
```

#### Week 1 Day 5: Background Norm Validation
```bash
# After implementing white-balance + illumination correction
./scripts/vlm_validate_background_norm.sh

# Expected: Validation reports showing background uniformity improvement
```

#### Week 2 Day 5: Deskewing Validation
```bash
# After implementing text deskewing
./scripts/vlm_validate_deskewing.sh

# Expected: Validation reports showing text alignment improvement
```

#### Week 3 Day 3: Combined Validation
```bash
# After integrating both enhancements
./scripts/vlm_validate_combined.sh

# Expected: Cumulative improvement reports
```

#### Week 3 Day 5: Final Aggregation
```bash
# Aggregate all Phase 1 results
python scripts/aggregate_vlm_phase1.py

# Generate: summaries/phase1_vlm_results.md
```

### Phase 2 (Weeks 4-6): Background Whitening + Text Isolation

[Similar workflow structure for Steps 11-14]

---

## Helper Scripts

### Script 1: Batch Baseline Assessment
```bash
#!/bin/bash
# scripts/vlm_baseline_assessment.sh

EXPERIMENT_DIR="experiment-tracker/experiments/20251217_024343_image_enhancements_implementation"
TEST_IMAGES="data/zero_prediction_worst_performers"

echo "Running baseline VLM assessment on worst performers..."

for img in $(ls $TEST_IMAGES/*.jpg | head -10); do
  basename=$(basename "$img" .jpg)
  echo "  - Analyzing: $basename"

  uv run python -m AgentQMS.vlm.cli.analyze_defects \
    --image "$img" \
    --mode image_quality \
    --backend openrouter \
    --output "$EXPERIMENT_DIR/vlm_reports/baseline/${basename}_quality.md" \
    --quiet
done

echo "Baseline assessment complete. Reports in: $EXPERIMENT_DIR/vlm_reports/baseline/"
```

### Script 2: Validation After Enhancement
```bash
#!/bin/bash
# scripts/vlm_validate_enhancement.sh

PHASE=$1  # e.g., "phase1"
EXPERIMENT_DIR="experiment-tracker/experiments/20251217_024343_image_enhancements_implementation"

echo "Running VLM validation for $PHASE..."

# Create comparisons
python scripts/create_before_after_comparison.py \
  --baseline data/zero_prediction_worst_performers \
  --enhanced artifacts/${PHASE}_enhanced \
  --output outputs/comparisons/${PHASE}

# Analyze comparisons
for cmp in outputs/comparisons/${PHASE}/*.jpg; do
  basename=$(basename "$cmp" .jpg | sed 's/comparison_//')
  echo "  - Validating: $basename"

  uv run python -m AgentQMS.vlm.cli.analyze_defects \
    --image "$cmp" \
    --mode enhancement_validation \
    --backend openrouter \
    --output "$EXPERIMENT_DIR/vlm_reports/${PHASE}_validation/${basename}_validation.md" \
    --quiet
done

echo "Validation complete. Reports in: $EXPERIMENT_DIR/vlm_reports/${PHASE}_validation/"
```

### Script 3: Aggregate Results
```python
#!/usr/bin/env python3
# scripts/aggregate_vlm_validations.py

import re
from pathlib import Path
import argparse

def extract_score(text, pattern):
    """Extract numeric score from markdown text."""
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def aggregate_validations(input_dir, output_file):
    """Aggregate all VLM validation reports into summary."""
    reports = sorted(Path(input_dir).glob("*_validation.md"))

    results = []
    for report in reports:
        text = report.read_text()

        # Extract key metrics
        image_id = report.stem.replace("_validation", "")
        overall_improvement = extract_score(text, r"Overall Improvement.*?(\d+\.\d+)")
        background_delta = extract_score(text, r"Background.*?Δ = ([+-]?\d+\.\d+)")
        alignment_delta = extract_score(text, r"Alignment.*?Δ = ([+-]?\d+\.\d+)")

        # Extract success status
        success_match = re.search(r"Overall Success.*?(Pass|Partial|Fail)", text)
        success = success_match.group(1) if success_match else "Unknown"

        results.append({
            "image_id": image_id,
            "overall_improvement": overall_improvement,
            "background_delta": background_delta,
            "alignment_delta": alignment_delta,
            "success": success,
        })

    # Generate summary markdown
    with open(output_file, "w") as f:
        f.write("# VLM Validation Summary\n\n")
        f.write(f"**Total Images**: {len(results)}\n\n")

        # Success breakdown
        pass_count = sum(1 for r in results if r["success"] == "Pass")
        partial_count = sum(1 for r in results if r["success"] == "Partial")
        fail_count = sum(1 for r in results if r["success"] == "Fail")

        f.write(f"**Success Rate**: {pass_count}/{len(results)} ({100*pass_count/len(results):.1f}%)\n")
        f.write(f"**Partial Success**: {partial_count}\n")
        f.write(f"**Failures**: {fail_count}\n\n")

        # Average improvements
        avg_bg = sum(r["background_delta"] for r in results if r["background_delta"]) / len(results)
        avg_align = sum(r["alignment_delta"] for r in results if r["alignment_delta"]) / len(results)

        f.write(f"**Average Background Improvement**: {avg_bg:+.2f} points\n")
        f.write(f"**Average Alignment Improvement**: {avg_align:+.2f} points\n\n")

        # Detailed table
        f.write("## Detailed Results\n\n")
        f.write("| Image | Background Δ | Alignment Δ | Success |\n")
        f.write("|-------|--------------|-------------|--------|\n")
        for r in results:
            bg = f"{r['background_delta']:+.1f}" if r['background_delta'] else "N/A"
            align = f"{r['alignment_delta']:+.1f}" if r['alignment_delta'] else "N/A"
            status_emoji = {"Pass": "✅", "Partial": "⚠️", "Fail": "❌"}.get(r["success"], "❓")
            f.write(f"| {r['image_id']} | {bg} | {align} | {status_emoji} {r['success']} |\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with validation reports")
    parser.add_argument("--output", required=True, help="Output summary file")
    args = parser.parse_args()

    aggregate_validations(args.input, args.output)
    print(f"Summary written to: {args.output}")
```

---

## Expected Outputs

### Baseline Assessment Report Structure
```
vlm_reports/baseline/
├── image_001_quality.md       # Tint severity: 7/10, Slant: +5°
├── image_002_quality.md       # Tint severity: 4/10, Slant: -8°
├── image_003_quality.md       # Tint severity: 9/10, Slant: +12°
└── ...
```

### Validation Report Structure
```
vlm_reports/phase1_validation/
├── image_001_validation.md    # Background Δ: +6 points, Alignment Δ: +4 points ✅
├── image_002_validation.md    # Background Δ: +3 points, Alignment Δ: +7 points ✅
├── image_003_validation.md    # Background Δ: -1 points (regression), ❌
└── ...
```

### Aggregated Summary
```
summaries/phase1_vlm_results.md

# VLM Validation Summary

**Total Images**: 10
**Success Rate**: 8/10 (80%)
**Partial Success**: 1
**Failures**: 1

**Average Background Improvement**: +4.2 points
**Average Alignment Improvement**: +5.1 points

## Detailed Results
| Image | Background Δ | Alignment Δ | Success |
|-------|--------------|-------------|---------|
| image_001 | +6.0 | +4.0 | ✅ Pass |
| image_002 | +3.0 | +7.0 | ✅ Pass |
| image_003 | -1.0 | +2.0 | ❌ Fail |
...
```

---

## Best Practices

### 1. Sample Size
- **Quick validation**: 5-10 images
- **Phase validation**: 10-20 images
- **Full validation**: 25+ images (all worst performers)

### 2. VLM Backend Selection
- **OpenRouter** (default): Good balance of quality and speed (~2-3s per image)
- **Solar Pro 2**: Faster but may be less detailed (~1s per image)
- **CLI**: For testing prompts without API calls

### 3. Batch Processing
- Run VLM analysis in parallel (up to 5 concurrent to avoid rate limits)
- Use `--quiet` flag to reduce terminal output
- Save reports immediately (don't wait for all to finish)

### 4. Prompt Tuning
- If VLM reports are too verbose, shorten prompts
- If VLM misses issues, add more specific questions to prompts
- Update scoring rubrics if scale interpretations vary

### 5. Integration with OCR Metrics
- Run VLM validation **in addition to** OCR accuracy measurement
- VLM provides qualitative insights OCR metrics miss (e.g., "background improved but text blur introduced")
- Use VLM to understand **why** OCR accuracy changed

---

## Troubleshooting

### Issue: VLM reports are generic/unhelpful
**Fix**: Make prompts more specific, provide example assessments inline

### Issue: VLM calls are slow (>5s per image)
**Fix**: Switch to faster backend (Solar Pro 2), or process only critical subset

### Issue: VLM scores are inconsistent across images
**Fix**: Add calibration examples to prompts showing what "7/10" looks like

### Issue: VLM misses obvious quality issues
**Fix**: Update prompts with specific questions about those issues

---

## Next Steps

1. **Run baseline assessment** on 10 worst performers before Week 1
2. **Create before/after comparison script** for Week 1 Day 5
3. **Set up aggregation scripts** for Week 3 Day 5
4. **Integrate VLM summaries** into experiment reports

VLM validation provides the **structured evidence** needed to confidently claim preprocessing improvements and justify deployment decisions.
