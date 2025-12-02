# Example Images and Results

This directory contains example images and detection results for documentation and demos.

## 📁 Directory Structure

```
examples/
├── inputs/          # Input receipt images
├── detections/      # Detection result visualizations
├── recognitions/    # Recognition result visualizations (future)
└── comparisons/     # Before/after comparisons
```

## 📸 Adding Examples

### Input Images

Place original receipt images in `inputs/`:
- Use descriptive filenames (e.g., `receipt_001.jpg`, `complex_layout_001.jpg`)
- Include diverse examples (different layouts, qualities, languages)
- Ensure images are properly licensed or anonymized

### Detection Results

Place detection visualizations in `detections/`:
- Show bounding boxes/polygons overlaid on images
- Use consistent naming: `{input_name}_detection.png`
- Include confidence scores in visualizations

### Recognition Results (Future)

When text recognition is implemented:
- Place recognition results in `recognitions/`
- Show detected text with bounding boxes
- Include accuracy metrics

## 🎨 Visualization Guidelines

- Use consistent color schemes
- Include legends for different elements
- Maintain image quality (PNG format recommended)
- Keep file sizes reasonable for web display

## 📝 Usage in Documentation

Reference examples in documentation:

```markdown
!Example Detection
```

---

*Example images will be added as the project develops.*
