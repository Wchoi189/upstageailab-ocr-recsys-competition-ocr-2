# Image Quality Analysis Prompt

Assess document image quality for OCR preprocessing requirements.

## Analysis Dimensions

### 1. Background
- Color: [RGB or description]
- Tint severity (1-10): [Score]
- Uniformity (1-10): [Score]
- Illumination gradients: [Description]
- Patterns/textures: Yes/No + description

### 2. Text Orientation
- Slant angle: [±X°]
- Detection confidence (1-10): [Score]
- Alignment quality (1-10): [Score]
- Mixed orientations: Yes/No + description

### 3. Shadows
- Presence: Yes/No
- Severity (1-10): [Score]
- Affected regions: [Description]
- Type: [Document shadow / External shadow]

### 4. Contrast & Readability
- Contrast score (1-10): [Score]
- Text-background ratio: [X:1 estimate]
- Faded text: Yes/No + regions
- Readability (1-10): [Score]

### 5. Noise & Artifacts
- Noise level (1-10): [Score]
- Noise type: [Classification]
- Sharpness (1-10): [Score]
- Artifacts: [List stamps, logos, stains, creases]

### 6. Document Structure
- Document type: [Classification]
- Layout quality (1-10): [Score]
- Geometric issues: [Warping, curling, perspective distortion]

## Output Format

```markdown
# Image Quality Report

**Image**: [Filename]
**Document Type**: [Classification]

## Issues Summary
[Prioritized list of quality problems]

## Metrics

| Dimension | Score (1-10) | Details |
|-----------|--------------|---------|
| Tint Severity | X | [Color description] |
| Slant Angle | ±X° | [Confidence X/10] |
| Shadow Severity | X | [Affected regions] |
| Contrast | X | [Ratio X:1] |
| Noise | X | [Type] |
| Sharpness | X | [Issues] |

## Recommendations
[Prioritized preprocessing steps: deskewing, white-balance, shadow removal, etc.]
```
