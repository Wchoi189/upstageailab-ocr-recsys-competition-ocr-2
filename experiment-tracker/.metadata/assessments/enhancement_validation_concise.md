# Enhancement Validation Prompt

Analyze side-by-side comparison (LEFT: before, RIGHT: after preprocessing) to quantify enhancement effectiveness.

## Analysis Dimensions

### 1. Background Improvement
- Color normalization: [Before RGB] → [After RGB]
- Tint severity (1-10): Before [X] → After [X]
- Uniformity (1-10): Before [X] → After [X]
- Success: Achieved neutral white? Yes/No + reason

### 2. Text Alignment
- Slant angle: Before [±X°] → After [±X°]
- Alignment quality (1-10): Before [X] → After [X]
- Success: Horizontal within ±2°? Yes/No + reason

### 3. Contrast Enhancement
- Text-background contrast (1-10): Before [X] → After [X]
- Faded text recovery: Yes/No + description
- Success: Contrast ≥7 for OCR? Yes/No + reason

### 4. Shadow/Illumination
- Shadow severity (1-10): Before [X] → After [X]
- Illumination uniformity (1-10): Before [X] → After [X]
- Success: Shadows reduced? Yes/No + reason

### 5. Noise/Artifacts
- Noise level (1-10): Before [X] → After [X]
- New artifacts: List any preprocessing artifacts
- Success: Noise reduced without quality loss? Yes/No + reason

### 6. Overall Quality
- Quality score (1-10): Before [X] → After [X]
- Readability (1-10): Before [X] → After [X]
- Processing necessary: Was preprocessing justified? Yes/No + reason

## Output Format

```markdown
# Enhancement Validation Report

**Image**: [Filename]
**Enhancement**: [Techniques applied]
**Verdict**: [Significant/Moderate/Minimal/None/Regression]

## Summary
- Key Wins: [1-3 improvements]
- Key Issues: [Problems/regressions]
- Recommendation: [Deploy/Tune/Reject]

## Metrics

| Dimension | Before | After | Δ | Success |
|-----------|--------|-------|---|---------|
| Tint Severity | X | X | ΔX | ✅/⚠️/❌ |
| Slant Angle | ±X° | ±X° | ΔX° | ✅/⚠️/❌ |
| Contrast | X | X | +X | ✅/⚠️/❌ |
| Shadows | X | X | -X | ✅/⚠️/❌ |
| Noise | X | X | -X | ✅/⚠️/❌ |
| Overall | X | X | +X | ✅/⚠️/❌ |

## Detailed Observations
[Section-by-section descriptions with quantitative details]
```
