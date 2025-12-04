# BUG-002: Inference Studio Visual Padding Mismatch

**Date**: 2025-12-03
**Status**: In Progress — Visual padding appears uneven despite correct calculations
**Priority**: Medium (affects visual presentation, not functionality)
**Related**: BUG-001 (resolved overlay alignment issue)

---

## 1. Summary

The **Next.js Inference Studio** canvas correctly calculates equal padding (40px on all sides for 640x640 images on 720x720 canvas), but the visual presentation may appear uneven. The backend preprocessing uses **top-left padding** (content at `(0,0)`, padding on right/bottom), which means the content is positioned in the top-left corner of the 640x640 processed image, not centered.

When this 640x640 image (with top-left content) is displayed on a 720x720 canvas with 40px padding on all sides, the visual result may appear left-aligned rather than centered, depending on the content distribution within the processed image.

---

## 2. Symptoms

- **Affected UI**: Next.js Inference Studio (`apps/frontend/src/components/inference/InferencePreviewCanvas.tsx`)
- **Observed Behavior**:
  - Console logs show correct padding calculations: `40.0px on all sides`
  - Canvas dimensions: 720x720
  - Image dimensions: 640x640
  - Calculated offsets: `dx=40.0px, dy=40.0px`
  - Backend meta padding: `{"top":0,"bottom":0,"left":0,"right":160}` (for portrait images)
- **Visual Issue**: Despite correct canvas padding, content may appear left-aligned or unevenly padded

---

## 3. Root Cause Analysis

### 3.1 Backend Preprocessing (Expected Behavior)

The backend preprocessing uses **top-left padding** to match training configuration:

```python
# ui/utils/inference/preprocess.py
# PadIfNeeded: pad to target_size x target_size with top_left position
# Content is at [0, resized_w] x [0, resized_h] within [0, 640] x [0, 640]
pad_h = target_size - scaled_h
pad_w = target_size - scaled_w
processed_image = cv2.copyMakeBorder(
    processed_image,
    0, pad_h, 0, pad_w,  # top, bottom, left, right (top_left padding)
    cv2.BORDER_CONSTANT,
    value=[0, 0, 0],
)
```

**Result**: For a portrait image (e.g., 428x960 → 285x640 after scaling):
- Content occupies: `[0, 285] x [0, 640]` (top-left area)
- Padding: `right=355px, bottom=0px`
- Content is **not centered** within the 640x640 frame

### 3.2 Frontend Canvas Display

The frontend correctly centers the 640x640 image on a 720x720 canvas:

```typescript
// Canvas: 720x720
// Image: 640x640
// Padding: (720 - 640) / 2 = 40px on all sides
const dx = (canvasSize - imageWidth) / 2;  // 40px
const dy = (canvasSize - imageHeight) / 2; // 40px
```

**Result**: The 640x640 image (with top-left content) is centered on the canvas, but the content within that image is still positioned at top-left.

### 3.3 Visual Perception

The visual result depends on content distribution:
- **Portrait images**: Content in top-left of 640x640 → appears left-aligned on canvas
- **Square images**: Content fills most of 640x640 → appears more centered
- **Landscape images**: Content in top-left of 640x640 → appears left-aligned

---

## 4. Example Data

**Console Output**:
```
BUG-001: Canvas centering calculation:
  Image dimensions: 640x640
  Canvas dimensions: 720x720
  Horizontal offset (dx): 40.0px (should be equal left/right padding)
  Vertical offset (dy): 40.0px (should be equal top/bottom padding)
  Expected padding: 40.0px on all sides
  Meta padding (backend): {"top":0,"bottom":0,"left":0,"right":160}
```

**Backend Metadata** (for portrait image):
- `original_size`: `[428, 960]` (width x height)
- `processed_size`: `[640, 640]`
- `padding`: `{"top":0,"bottom":0,"left":0,"right":160}`
- `scale`: `0.666...`
- `coordinate_system`: `"pixel"`

---

## 5. Technical Details

### 5.1 Current Implementation

**Backend** (`ui/utils/inference/engine.py`):
- Preprocessing: LongestMaxSize + PadIfNeeded with `position: "top_left"`
- Content positioned at `(0, 0)` in processed frame
- Padding on right/bottom only

**Frontend** (`apps/frontend/src/components/inference/InferencePreviewCanvas.tsx`):
- Canvas: Fixed 720x720
- Image centering: `dx = (720 - 640) / 2 = 40px`
- Polygon coordinates: Already in processed_size frame, only need canvas centering offset

### 5.2 Coordinate Flow

1. **Original image**: 428x960 (portrait)
2. **Preprocessing**: Scale to 285x640, pad to 640x640 (top-left)
   - Content area: `[0, 285] x [0, 640]`
   - Padding: `right=355px, bottom=0px`
3. **Polygon mapping**: Original space → processed_size frame (640x640)
   - Polygons in `[0, 285] x [0, 640]` range
4. **Frontend display**: 640x640 image centered on 720x720 canvas
   - Canvas padding: 40px on all sides
   - Content appears at: `[40, 325] x [40, 680]` on canvas
   - But content is still left-aligned within the 640x640 image

---

## 6. Potential Solutions

### Option 1: Center Content in Backend Preprocessing (Recommended)

**Approach**: Modify preprocessing to center content within the 640x640 frame instead of top-left positioning.

**Pros**:
- Content appears centered on canvas
- Better visual presentation
- Matches user expectations

**Cons**:
- Requires coordinate remapping (polygons need offset adjustment)
- May affect model performance if training used top-left padding
- More complex implementation

**Implementation**:
```python
# Center content instead of top-left
pad_left = pad_w // 2
pad_right = pad_w - pad_left
pad_top = pad_h // 2
pad_bottom = pad_h - pad_top
processed_image = cv2.copyMakeBorder(
    processed_image,
    pad_top, pad_bottom, pad_left, pad_right,  # Centered padding
    cv2.BORDER_CONSTANT,
    value=[0, 0, 0],
)
# Update polygon coordinates: add pad_left to x, pad_top to y
```

### Option 2: Visual Centering in Frontend (Current + Enhancement)

**Approach**: Keep backend as-is, but add visual indication or adjust canvas padding to account for content position.

**Pros**:
- No backend changes
- Maintains training compatibility
- Simpler implementation

**Cons**:
- Doesn't actually center content
- May require asymmetric padding calculations
- Less intuitive

**Implementation**:
- Option 2a: Add visual guide (grid/outline) showing content area
- Option 2b: Calculate content bounds from metadata and adjust canvas padding
- Option 2c: Accept current behavior (content is correctly positioned, just not visually centered)

### Option 3: Hybrid Approach

**Approach**: Use centered padding for display, but keep top-left for model inference.

**Pros**:
- Best visual presentation
- Maintains model compatibility

**Cons**:
- Requires two preprocessing paths (inference vs display)
- More complex coordinate mapping
- Potential for inconsistencies

---

## 7. Recommended Next Steps

1. **Clarify Requirements**:
   - Is visual centering required, or is current behavior acceptable?
   - Does model training use top-left padding (must match)?

2. **If Centering Required**:
   - Implement Option 1 (center content in backend)
   - Update polygon coordinate mapping to account for padding offsets
   - Verify model performance is not affected

3. **If Current Behavior Acceptable**:
   - Document that content is positioned at top-left (training compatibility)
   - Add visual indicator showing content area bounds
   - Update user documentation

4. **Testing**:
   - Test with various image aspect ratios (portrait, landscape, square)
   - Verify polygon alignment remains correct after any changes
   - Check model inference accuracy if preprocessing changes

---

## 8. Related Files

- `ui/utils/inference/preprocess.py` - Preprocessing logic (top-left padding)
- `ui/utils/inference/engine.py` - Coordinate mapping
- `apps/frontend/src/components/inference/InferencePreviewCanvas.tsx` - Canvas rendering
- `configs/transforms/base.yaml` - Training configuration (padding position)

---

## 9. Notes

- This is a **visual presentation issue**, not a functional bug
- Overlay alignment (BUG-001) is correct - polygons align with text
- The padding calculations are mathematically correct (40px on all sides)
- The perceived unevenness is due to content positioning within the processed image
- Backend preprocessing matches training configuration (top-left padding) - changing this may affect model performance

---

## 10. Status Updates

**2025-12-03**: Initial report created
- Identified visual padding mismatch despite correct calculations
- Documented root cause (top-left padding in preprocessing)
- Proposed three solution options
- Awaiting decision on requirements (centering vs training compatibility)
