# Annotation Rendering in OCR Inference Console

## Overview

The OCR Inference Console displays OCR predictions as polygon overlays on images. This document describes the coordinate system, image preprocessing, and rendering approach.

## Data Contract

The console follows the same data contract as `playground-console` Inference Studio, defined in [`docs/pipeline/inference-data-contracts.md`](../../docs/pipeline/inference-data-contracts.md).

### Key Components

1. **Preview Image**: The backend provides `preview_image_base64` - a preprocessed image in `processed_size` space (typically 640x640) that matches the coordinate system of predictions.

2. **Metadata**: The `meta` field contains:
   - `processed_size`: Dimensions of the preprocessed image (e.g., [640, 640])
   - `original_size`: Dimensions of the source image
   - `padding`: Black padding added during preprocessing (top, bottom, left, right)
   - `coordinate_system`: "pixel" or "normalized"
   - `scale`: Scaling factor applied during preprocessing

3. **Predictions**: Polygon coordinates are in `processed_size` space, matching the preview image.

## Rendering Approach

### Using Preview Image (Recommended)

When `preview_image_base64` is available:

1. **Load Preview Image**: Decode base64 and create an object URL for display
2. **Trim Padding**: Calculate content area from padding metadata and use SVG `viewBox` to crop black padding
3. **Direct Coordinate Mapping**: Since both image and coordinates are in `processed_size` space, no transformation is needed

```typescript
// Content area calculation (top-left padding)
const contentW = processedW - padding.right;
const contentH = processedH - padding.bottom;
const contentArea = { x: 0, y: 0, w: contentW, h: contentH };

// SVG viewBox crops to content area
<svg viewBox={`${contentArea.x} ${contentArea.y} ${contentArea.w} ${contentArea.h}`}>
  <image width={processedW} height={processedH} /> {/* Full preview image */}
  {/* Polygons use coordinates directly - no transformation */}
</svg>
```

### Fallback: Original Image

If preview image is unavailable:

1. **Load Original Image**: Use the uploaded image file
2. **Transform Coordinates**: Scale from `processed_size` to actual image dimensions
3. **No Padding Trim**: Original image doesn't have preprocessing padding

```typescript
const scaleX = actualWidth / processedW;
const scaleY = actualHeight / processedH;
transformedPoints = points.map(([x, y]) => [x * scaleX, y * scaleY]);
```

## Visual Styling

Annotations use subtle styling to avoid visual clutter:

- **Fill**: Light red (pink) with low opacity: `rgba(255, 182, 193, 0.2)`
- **Stroke**: Light red (tomato) with medium opacity: `rgba(255, 99, 71, 0.8)`
- **Stroke Width**: Very thin (`0.5px`) for clean appearance

## Implementation Files

- **Component**: [`src/components/PolygonOverlay.tsx`](../src/components/PolygonOverlay.tsx)
- **API Client**: [`src/api/ocrClient.ts`](../src/api/ocrClient.ts)
- **Data Contracts**: [`docs/pipeline/inference-data-contracts.md`](../../docs/pipeline/inference-data-contracts.md)

## Related Documentation

- [Inference Data Contracts](../../docs/pipeline/inference-data-contracts.md) - Coordinate system and padding contracts
- [Playground Console Implementation](../../apps/frontend/src/components/inference/InferencePreviewCanvas.tsx) - Reference implementation using Canvas API
