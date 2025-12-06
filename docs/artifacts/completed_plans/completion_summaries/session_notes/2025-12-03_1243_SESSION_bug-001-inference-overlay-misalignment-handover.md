---
title: "2025 12 03 Bug 001 Inference Overlay Misalignment Handover"
date: "2025-12-06 18:09 (KST)"
type: "session_note"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['session_note', 'troubleshooting', 'handover']
---





# Session Handover: BUG-001 Inference Overlay Misalignment

**Date**: 2025-12-03
**Bug ID**: BUG-001
**Status**: In Progress - Fixes Not Taking Effect
**Priority**: Medium-High (affects visual QA and user trust)

## Context Rebuild Prompt

```
I'm working on fixing BUG-001: inference overlay misalignment in the Next.js Inference Studio application.
The issue is that prediction overlays (polygon bounding boxes) are consistently deviated to the right for
portrait images (720x1280, 960x1280), while square images (1280x1280) align correctly.

The OCR pipeline uses top_left padding (content in top-left, padding on right/bottom) which matches the
training configuration. However, for display purposes, users expect centered content.

I've attempted fixes in both backend (inference engine) and frontend (visualization component), but the
fixes don't seem to have an effect. The inference viewer may be receiving data from an unexpected source.

I need to:
1. Identify where the inference viewer is actually getting its data from
2. Add debugging to trace the data flow
3. Verify which component is actually rendering the overlays
4. Determine if there are multiple inference viewers (Next.js frontend vs Streamlit UI)
```

## Problem Summary

### Symptoms
- **Portrait images** (720x1280, 960x1280): Prediction overlays appear right-aligned/deviated to the right
- **Square images** (1280x1280): Overlays align correctly
- Overlays appear as if they need padding to the right or to be centered with the image
- Canvas style uses `max-width: 100%` and `height: auto`, which may cause stretching when window is maximized

### Affected Images
- `drp.en_ko.in_house.selectstar_002362.jpg` (720x1280) - enlarged, misaligned
- `drp.en_ko.in_house.selectstar_002374.jpg` (720x1280) - misaligned
- `drp.en_ko.in_house.selectstar_000275.jpg` (960x1280) - misaligned
- `drp.en_ko.in_house.selectstar_002432.jpg` (960x1280) - misaligned
- `drp.en_ko.in_house.selectstar_003263.jpg` (1280x1280) - **aligned correctly**

## Attempted Fixes

### 1. Backend Fixes (Reverted)
**Location**: `ui/utils/inference/engine.py`

- ✅ Fixed preprocessing to use LongestMaxSize + PadIfNeeded matching postprocessing
- ✅ Fixed coordinate mapping to use exact processed image from preprocessing
- ✅ Added forward coordinate transformation from original space to preview space
- ❌ Attempted to center preview image in backend (reverted - wrong layer)
- ❌ Attempted to add centering offset to polygon coordinates in backend (reverted)

**Status**: Backend now correctly returns raw 640x640 preview images with top_left padding and polygons in preview coordinate space.

### 2. Frontend Fixes (Current)
**Location**: `apps/frontend/src/components/inference/InferencePreviewCanvas.tsx`

- ✅ Added `getCenteringOffset()` function to detect portrait images and calculate horizontal centering
- ✅ Added centering logic in image drawing (lines 276-290)
- ✅ Added centering offset to polygon coordinates in `drawPolygon()` (line 320)
- ❓ **Issue**: Fixes don't seem to have effect - may not be the component receiving data

**Current Implementation**:
```typescript
// Lines 224-246: Centering offset calculation
const getCenteringOffset = (): number => {
  if (!displayBitmap || !result?.preview_image_base64) return 0;
  if (displayBitmap.width === 640 && displayBitmap.height === 640) {
    if (result.regions && result.regions.length > 0) {
      const allX = result.regions.flatMap((r) => r.polygon.map((p) => p[0]));
      if (allX.length > 0) {
        const maxX = Math.max(...allX);
        if (maxX < 600) { // Portrait detection
          const contentWidth = maxX;
          const padWidth = 640 - contentWidth;
          return Math.round(padWidth / 2);
        }
      }
    }
  }
  return 0;
};
```

## Critical Investigation Needed

### 1. Multiple Inference Viewers?
There appear to be **two different inference viewers**:

#### A. Next.js Frontend Viewer
- **Component**: `apps/frontend/src/components/inference/InferencePreviewCanvas.tsx`
- **Page**: `apps/frontend/src/pages/Inference.tsx`
- **API Endpoint**: `/inference/preview` (FastAPI backend)
- **Data Flow**: `runInferencePreview()` → `InferencePreviewResponse` → `InferencePreviewCanvas`

#### B. Streamlit UI Viewer
- **Component**: `ui/apps/unified_ocr_app/components/inference/results_viewer.py`
- **Page**: `ui/apps/unified_ocr_app/pages/2_🤖_Inference.py`
- **Data Source**: `state.inference_results` (Streamlit state)
- **Visualization**: Uses OpenCV (`cv2.polylines`) to draw polygons

**Question**: Which viewer is the user actually seeing? The fixes were applied to the Next.js component, but the user might be using the Streamlit UI.

### 2. Data Flow Investigation

#### Next.js Frontend Flow
```
User uploads image
  → InferencePreviewCanvas.tsx
  → runInferencePreview() (apps/frontend/src/api/inference.ts)
  → POST /inference/preview
  → apps/backend/services/playground_api/routers/inference.py
  → engine.predict_array()
  → ui/utils/inference/engine.py
  → Returns InferencePreviewResponse with:
     - regions: TextRegion[] (polygons in preview space)
     - preview_image_base64: string (640x640 PNG)
  → InferencePreviewCanvas renders
```

#### Streamlit UI Flow
```
User uploads image
  → unified_ocr_app/pages/2_🤖_Inference.py
  → Inference engine (ui/utils/inference/engine.py)
  → Stores in state.inference_results
  → results_viewer.py renders
  → Uses result.image and result.polygons directly
```

**Key Difference**: Streamlit viewer uses `result.image` (original image?) while Next.js uses `preview_image_base64`.

### 3. Debugging Steps Needed

1. **Identify Active Viewer**:
   - Check browser URL: `/inference` (Next.js) vs Streamlit port
   - Check which component is actually rendering
   - Add console.log to both viewers to see which one executes

2. **Trace Data Flow**:
   - Add logging in `InferencePreviewCanvas.tsx` to log:
     - `result?.preview_image_base64` presence
     - `displayBitmap` dimensions
     - `getCenteringOffset()` return value
     - Polygon coordinates before/after offset

   - Add logging in `results_viewer.py` to log:
     - `result.image` shape and type
     - `result.polygons` coordinate ranges
     - Coordinate space (original vs preview)

3. **Verify Coordinate Space**:
   - Check if polygons are in original image space vs preview space
   - Verify `preview_image_base64` is actually being used
   - Check if `displayBitmap` is falling back to original image

4. **Check for Caching**:
   - Browser cache might be serving old JavaScript
   - Check if frontend needs rebuild (`npm run build` or dev server restart)
   - Verify backend changes are deployed

## Relevant Files

### Backend (Python)
- `ui/utils/inference/engine.py` - Main inference engine (lines 350-500)
- `ui/utils/inference/preprocess.py` - Image preprocessing (lines 44-109)
- `ui/utils/inference/postprocess.py` - Coordinate mapping (lines 63-184)
- `apps/backend/services/playground_api/routers/inference.py` - API endpoint (lines 220-302)

### Frontend (TypeScript/React)
- `apps/frontend/src/components/inference/InferencePreviewCanvas.tsx` - **Main viewer component** (lines 35-454)
- `apps/frontend/src/pages/Inference.tsx` - Inference page wrapper
- `apps/frontend/src/api/inference.ts` - API client (lines 171-178)

### Streamlit UI (Python)
- `ui/apps/unified_ocr_app/components/inference/results_viewer.py` - **Alternative viewer** (lines 15-284)
- `ui/apps/unified_ocr_app/pages/2_🤖_Inference.py` - Streamlit inference page

### Configuration
- `configs/transforms/base.yaml` - OCR pipeline config (uses `position: "top_left"`)

### Bug Reports
- `docs/artifacts/bug_reports/2025-12-02_2313_BUG_001_inference-studio-overlay-misalignment.md`
- `docs/artifacts/bug_reports/2025-12-03_0003_BUG_001_inference-resize-misalignment.md`

## Key Code Locations

### Coordinate Mapping (Backend)
```python
# ui/utils/inference/engine.py, lines 427-486
def _map_polygons_to_preview_space(payload):
    # Maps polygons from original_shape to resized/padded preview space (640x640)
    forward_scale_x = resized_w / float(original_w)
    forward_scale_y = resized_h / float(original_h)
    transformed_coords = coords_2d * np.array([forward_scale_x, forward_scale_y])
```

### Centering Logic (Frontend)
```typescript
// apps/frontend/src/components/inference/InferencePreviewCanvas.tsx, lines 224-290
const getCenteringOffset = (): number => { /* ... */ }
// Lines 276-290: Image centering
// Line 320: Polygon offset application
```

### Streamlit Visualization
```python
# ui/apps/unified_ocr_app/components/inference/results_viewer.py, lines 127-143
cv2.polylines(viz_image, [poly_array], True, polygon_color, polygon_thickness)
# Uses result.image and result.polygons directly - coordinate space unclear
```

### 4. Fix Attempts Summary (Chronological)

| Date        | Layer      | Change / Hypothesis                                                                                         | Status              | Outcome / Notes                                                                                                                                                           |
|------------:|-----------:|-------------------------------------------------------------------------------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2025-11-xx  | Backend    | Align preprocessing with training via **LongestMaxSize + PadIfNeeded** (top_left), and map polygons from original space → preview (640×640). | ✅ Landed           | `preprocess_image()` now pads to 640×640 with content in top-left; `_map_polygons_to_preview_space()` scales polygons into this preview frame (no viewer centering yet). |
| 2025-11-xx  | Frontend   | Add `getCenteringOffset()` and apply horizontal centering to both image and polygons in `InferencePreviewCanvas`.                              | ❌ Reverted         | Produced double-offset behavior and right-shift for portrait images when combined with backend mapping; centering moved to the wrong abstraction layer.                 |
| 2025-12-02  | Streamlit  | Update legacy unified OCR Streamlit app to consume `preview_image_base64` instead of original image for visualization.                         | ✅ Landed           | Streamlit viewer now renders the same 640×640 preprocessed image used by the model, eliminating one source of misalignment there; Next.js Inference Studio still off.  |
| 2025-12-03  | Backend    | Added **Inference Engine Data Contract** metadata (`original_size`, `processed_size`, `padding`, `scale`, `coordinate_system`) to inference output. | ✅ Landed           | `ui/utils/inference/engine.py` attaches `meta` alongside `preview_image_base64`; FastAPI router exposes it via `InferenceMetadata`/`Padding` models for frontend use.   |
| 2025-12-03  | Frontend   | Updated Next.js APIs and `InferencePreviewCanvas` to consume `meta`, verify `displayBitmap` vs `processed_size`, and branch on `coordinate_system`. | ✅ Landed (diagnostic) | Viewer now logs contract mismatches and uses `coordinate_system` instead of pure heuristics; **root bug persists** (horizontal shift), indicating a deeper mapping issue. |

> **Current Status (2025-12-03):** The model’s polygons are believed to be correct in their own preview coordinate system, but we still lack a single, trusted description of (1) how padding is applied to content vs. preview canvas, and (2) whether the final offset responsibility belongs in the inference pipeline, the visualizer, or both under a clear contract.

## Next Steps

1. **Immediate**: Add console logging to `InferencePreviewCanvas.tsx`:
   ```typescript
   console.log('BUG-001 Debug:', {
     hasPreviewImage: !!result?.preview_image_base64,
     displayBitmapSize: `${displayBitmap?.width}x${displayBitmap?.height}`,
     centeringOffset: getCenteringOffset(),
     polygonCount: result?.regions?.length,
     firstPolygonCoords: result?.regions?.[0]?.polygon
   });
   ```

2. **Verify Active Viewer**: Check which URL/port the user is accessing

3. **Check Streamlit Viewer**: If using Streamlit, apply fixes to `results_viewer.py` instead

4. **Coordinate Space Verification**: Add logging to verify polygon coordinates are in expected space

5. **Rebuild/Refresh**: Ensure frontend changes are deployed (check if dev server needs restart)

## Questions to Answer

1. Which inference viewer is actually being used? (Next.js or Streamlit)
2. Is `preview_image_base64` being received and used?
3. Are polygon coordinates in preview space (640x640) or original space?
4. Is the centering offset being calculated correctly?
5. Is the offset being applied to both image and polygons?
6. Are there any browser console errors?
7. Is the frontend code actually executing (check Network tab for API calls)?

## Testing Checklist

- [ ] Verify which viewer is active (check URL/port)
- [ ] Add console logging to trace data flow
- [ ] Test with known problematic image (e.g., `selectstar_002374.jpg`)
- [ ] Check browser console for errors
- [ ] Verify API response includes `preview_image_base64`
- [ ] Check polygon coordinates in Network response
- [ ] Verify `displayBitmap` dimensions match preview image
- [ ] Test centering offset calculation with debug logs
- [ ] If using Streamlit, test fixes in `results_viewer.py`

## Related Issues

- BUG-001: Initial overlay misalignment (general)
- BUG-001: Resize misalignment (transform mismatch)
- BUG-001: Rightward deviation (portrait images)
- Canvas stretching with `max-width: 100%`

## Notes

- The backend coordinate mapping math has been verified correct
- The issue appears to be in the visualization/display layer
- Multiple viewers may exist - need to identify which one is active
- Frontend fixes may not be executing if wrong component is being used
