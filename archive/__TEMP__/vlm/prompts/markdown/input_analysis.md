# Input Characteristics Analysis Prompt

Analyze the source input image to identify unique properties that may affect processing.

## Task

Examine the input image and describe its characteristics that could impact processing quality or cause failures.

## Analysis Focus

1. **ROI Coverage**: Assess:
   - How much of the frame does the subject fill?
   - Is the ROI centered or off-center?
   - Are there multiple subjects or a single focus?
   - Is the ROI clearly defined or ambiguous?

2. **Contrast/Lighting**: Evaluate:
   - Contrast between foreground and background
   - Lighting conditions (bright, dim, uneven)
   - Shadow presence and impact
   - Color saturation levels
   - Exposure quality

3. **Geometry**: Examine:
   - Is the image already cropped or rectified?
   - Perspective angle (straight-on, angled, extreme)
   - Aspect ratio and orientation
   - Edge conditions (touching borders, isolated)
   - Geometric complexity

## Output Format

Provide a structured analysis in the following format:

- **ROI Coverage:** [Description of subject coverage]
- **Contrast/Lighting:** [Lighting and contrast assessment]
- **Geometry:** [Geometric characteristics]
