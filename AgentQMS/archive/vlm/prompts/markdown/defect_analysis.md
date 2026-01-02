# Defect Analysis Prompt

Analyze the provided image for visual defects and artifacts in the output.

## Task

Examine the image carefully and identify any visual defects, distortions, or artifacts that may indicate processing failures or quality issues.

## Analysis Focus

1. **Distortion Types**: Look for:
   - Shearing (skewed text or objects)
   - Pincushion distortion (curved edges)
   - Barrel distortion (bulging center)
   - Stretching (elongated elements)
   - Blank or empty output regions
   - Cropped or missing content

2. **Key Features**: Identify:
   - Text alignment issues (diagonal, rotated, misaligned)
   - Pixel smearing or blurring
   - ROI (Region of Interest) problems
   - Color artifacts
   - Edge artifacts
   - Compression artifacts

3. **Comparison**: Compare with expected output:
   - Is this worse than baseline?
   - Is this a regression from previous version?
   - Are there improvements despite defects?

## Output Format

Provide a structured analysis in the following format:

- **Distortion Type:** [Type of distortion observed]
- **Key Features:** [Specific features that indicate the defect]
- **Comparison:** [How this compares to baseline or expected output]
