---
title: "Perspective Correction Phase 2 - Completion Guide"
date: "2025-12-14 12:00 (KST)"
type: "template"
category: "planning"
status: "active"
version: "1.0"
---

# Perspective Correction Phase 2 - Completion Guide

## Current Implementation Status

### ✅ **Completed (80%)**

1. **Backend Infrastructure** - DONE
   - [ocr/utils/perspective_correction.py](ocr/utils/perspective_correction.py)
     - `correct_perspective_from_mask()` now returns transform matrix (line 275-323)
     - `transform_polygons_inverse()` transforms annotations back to original space (line 389-447)

   - [ocr/inference/preprocess.py](ocr/inference/preprocess.py:114-153)
     - `apply_optional_perspective_correction()` supports matrix return

   - [apps/shared/backend_shared/models/inference.py](apps/shared/backend_shared/models/inference.py)
     - `InferenceRequest.enable_perspective_correction` added (line 108-111)
     - `InferenceRequest.perspective_display_mode` added (line 112-116)

2. **Frontend Structure** - ANALYZED
   - [apps/ocr-inference-console/src/api/ocrClient.ts](apps/ocr-inference-console/src/api/ocrClient.ts:146-151) - Request construction
   - [apps/ocr-inference-console/src/components/Workspace.tsx](apps/ocr-inference-console/src/components/Workspace.tsx:26-60) - Inference handling
   - [apps/ocr-inference-console/src/components/Sidebar.tsx](apps/ocr-inference-console/src/components/Sidebar.tsx) - UI controls location

### 🔨 **Remaining Work (20%)**

#### **1. Inference Engine Updates** (Critical)

**File**: [ocr/inference/engine.py](ocr/inference/engine.py)

**What to do**: Update `_predict_from_array()` method to handle `perspective_display_mode` parameter

**Current state**: Lines 342-383 handle perspective correction but don't support display mode

**Required changes**:

```python
def _predict_from_array(
    self,
    image: np.ndarray,
    return_preview: bool = True,
    enable_perspective_correction: bool | None = None,
    perspective_display_mode: str = "corrected",  # ADD THIS
) -> dict[str, Any] | None:
    # ... existing code ...

    # REPLACE lines 366-383 with:
    if enable_perspective_correction is not None:
        enable_persp = bool(enable_perspective_correction)
    else:
        raw_config = bundle.raw_config
        enable_persp = False
        try:
            enable_persp = bool(getattr(raw_config, "enable_perspective_correction", False))
        except Exception:
            enable_persp = False

    # NEW: Store original image and transform matrix for Phase 2
    original_image_for_display = None
    perspective_transform_matrix = None

    if enable_persp:
        if perspective_display_mode == "original":
            # Phase 2: Store original, get matrix, correct for inference only
            original_image_for_display = image.copy()
            image, perspective_transform_matrix = apply_optional_perspective_correction(
                image, enable_perspective_correction=True, return_matrix=True
            )
        else:
            # Phase 1: Correct and display corrected (current behavior)
            image = apply_optional_perspective_correction(image, enable_perspective_correction=True)

    # ... rest of inference code ...

    # AFTER inference completes (around line 500+ where results are returned):
    # NEW: Transform polygons back if in "original" mode
    if perspective_transform_matrix is not None and perspective_display_mode == "original":
        from ocr.utils.perspective_correction import transform_polygons_inverse

        # Transform polygons back to original space
        if "polygons" in result:
            result["polygons"] = transform_polygons_inverse(
                result["polygons"],
                perspective_transform_matrix
            )

        # Use original image for preview instead of corrected
        if original_image_for_display is not None and return_preview:
            # Recreate preview image from original
            # You'll need to preprocess original_image_for_display and use it for preview
            preview_image_bgr = original_image_for_display  # Simplified - may need preprocessing

    # Return result with potentially transformed polygons and original preview
```

**Exact locations to modify**:
1. Line 342: Add `perspective_display_mode` parameter
2. Line 353: Update docstring
3. Lines 366-383: Replace with new logic above
4. After inference results (search for `return {` around line 500): Add polygon transformation

#### **2. Update Method Signatures**

**Files**:
- [ocr/inference/engine.py](ocr/inference/engine.py)
  - `predict_array()` - line 210: Add `perspective_display_mode: str = "corrected"` parameter
  - `predict_image()` - line 266: Add `perspective_display_mode: str = "corrected"` parameter
  - Update calls to `_predict_from_array()` at lines 260-264 and 336-340

#### **3. Backend Endpoints**

**File**: [apps/ocr-inference-console/backend/main.py](apps/ocr-inference-console/backend/main.py:273-279)

**Change**:
```python
result = _inference_engine.predict_array(
    image_array=image,
    binarization_thresh=request.confidence_threshold,
    box_thresh=request.nms_threshold,
    return_preview=True,
    enable_perspective_correction=request.enable_perspective_correction,
    perspective_display_mode=request.perspective_display_mode,  # ADD THIS LINE
)
```

**File**: [apps/playground-console/backend/routers/inference.py](apps/playground-console/backend/routers/inference.py:140-146)

**Same change** - add `perspective_display_mode` parameter

#### **4. Frontend UI Controls**

**File**: [apps/ocr-inference-console/src/components/Sidebar.tsx](apps/ocr-inference-console/src/components/Sidebar.tsx)

**Add** after CheckpointSelector (around line 59):

```tsx
<div className="pt-4 border-t border-gray-100">
    <h4 className="px-4 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Perspective Correction
    </h4>
    <div className="px-4 space-y-2">
        <label className="flex items-center gap-2 text-sm">
            <input
                type="checkbox"
                checked={enablePerspectiveCorrection}
                onChange={(e) => onPerspectiveCorrectionChange(e.target.checked)}
                className="rounded border-gray-300"
            />
            <span>Enable Correction</span>
        </label>
        {enablePerspectiveCorrection && (
            <div className="ml-6 space-y-1">
                <label className="flex items-center gap-2 text-xs">
                    <input
                        type="radio"
                        name="displayMode"
                        value="corrected"
                        checked={displayMode === "corrected"}
                        onChange={(e) => onDisplayModeChange(e.target.value)}
                    />
                    <span>Show Corrected</span>
                </label>
                <label className="flex items-center gap-2 text-xs">
                    <input
                        type="radio"
                        name="displayMode"
                        value="original"
                        checked={displayMode === "original"}
                        onChange={(e) => onDisplayModeChange(e.target.value)}
                    />
                    <span>Show Original</span>
                </label>
            </div>
        )}
    </div>
</div>
```

**Update Sidebar props**:
```tsx
interface SidebarProps {
    selectedCheckpoint: string | null;
    onCheckpointChange: (checkpoint: string) => void;
    enablePerspectiveCorrection: boolean;
    onPerspectiveCorrectionChange: (enabled: boolean) => void;
    displayMode: string;
    onDisplayModeChange: (mode: string) => void;
}
```

#### **5. Frontend State & API Client**

**File**: [apps/ocr-inference-console/src/App.tsx](apps/ocr-inference-console/src/App.tsx)

**Add state**:
```tsx
const [enablePerspectiveCorrection, setEnablePerspectiveCorrection] = useState(false);
const [displayMode, setDisplayMode] = useState("corrected");
```

**Pass to components**:
```tsx
<Sidebar
    selectedCheckpoint={selectedCheckpoint}
    onCheckpointChange={setSelectedCheckpoint}
    enablePerspectiveCorrection={enablePerspectiveCorrection}
    onPerspectiveCorrectionChange={setEnablePerspectiveCorrection}
    displayMode={displayMode}
    onDisplayModeChange={setDisplayMode}
/>

<Workspace
    selectedCheckpoint={selectedCheckpoint}
    enablePerspectiveCorrection={enablePerspectiveCorrection}
    displayMode={displayMode}
/>
```

**File**: [apps/ocr-inference-console/src/components/Workspace.tsx](apps/ocr-inference-console/src/components/Workspace.tsx)

**Update props**:
```tsx
interface WorkspaceProps {
    selectedCheckpoint: string | null;
    enablePerspectiveCorrection: boolean;
    displayMode: string;
}
```

**Update predict call** (line 40):
```tsx
const result = await ocrClient.predict(
    file,
    selectedCheckpoint || undefined,
    enablePerspectiveCorrection,
    displayMode
);
```

**File**: [apps/ocr-inference-console/src/api/ocrClient.ts](apps/ocr-inference-console/src/api/ocrClient.ts)

**Update predict function signature** (line 114):
```tsx
predict: async (
    file: File,
    checkpointPath?: string,
    enablePerspectiveCorrection?: boolean,
    perspectiveDisplayMode?: string
): Promise<InferenceResponse> => {
```

**Update request body** (line 146):
```tsx
const requestBody = {
    checkpoint_path: checkpointPath || "",
    image_base64: base64Image,
    confidence_threshold: 0.1,
    nms_threshold: 0.4,
    enable_perspective_correction: enablePerspectiveCorrection || false,
    perspective_display_mode: perspectiveDisplayMode || "corrected",
};
```

---

## Testing Checklist

### Phase 1 (Corrected Mode) - Already Working
- [ ] Upload receipt image
- [ ] Enable perspective correction
- [ ] Select "corrected" mode
- [ ] Verify corrected image is displayed
- [ ] Verify annotations overlay correctly on corrected image

### Phase 2 (Original Mode) - New Feature
- [ ] Upload skewed receipt image
- [ ] Enable perspective correction
- [ ] Select "original" mode
- [ ] Verify ORIGINAL image is displayed (not corrected)
- [ ] Verify annotations are transformed and overlay correctly on original image
- [ ] Compare polygon coordinates between modes (should be different)

---

## Quick Implementation Script

Here's a script to validate the current state:

```bash
# Test syntax
python -m py_compile \
    ocr/utils/perspective_correction.py \
    ocr/inference/preprocess.py \
    apps/shared/backend_shared/models/inference.py

# If all pass, implementation is syntactically correct so far
echo "✅ Syntax validation passed"
```

---

## Continuation Prompt for New Session

If you need to continue in a new session, use this prompt:

```
I'm continuing the implementation of Perspective Correction Phase 2 for the OCR inference system.

**Context:**
- Working on domain-driven backend reconstruction (docs/artifacts/implementation_plans/2025-12-14_1746_implementation_plan_domain-driven-backends.md)
- Implementing user-activated perspective correction with two display modes
- Implementation is 80% complete

**Completed:**
1. Backend infrastructure for transform matrix and inverse transformation
2. API models updated with enable_perspective_correction and perspective_display_mode parameters
3. Frontend structure analyzed

**Remaining:**
1. Update InferenceEngine._predict_from_array() to handle perspective_display_mode
2. Update backend endpoints to pass display_mode parameter
3. Add frontend UI controls for perspective correction toggle
4. Wire frontend state and API client

**Reference:** See docs/artifacts/implementation_guides/perspective-correction-phase2-completion-guide.md for detailed instructions.

Please complete the remaining implementation following the guide.
```

---

## Files Reference

### Core Implementation Files
- [ocr/utils/perspective_correction.py](ocr/utils/perspective_correction.py) - Transform functions
- [ocr/inference/engine.py](ocr/inference/engine.py) - Inference logic (NEEDS UPDATES)
- [ocr/inference/preprocess.py](ocr/inference/preprocess.py) - Preprocessing with matrix support
- [apps/shared/backend_shared/models/inference.py](apps/shared/backend_shared/models/inference.py) - API models

### Backend Endpoints
- [apps/ocr-inference-console/backend/main.py](apps/ocr-inference-console/backend/main.py:273-279)
- [apps/playground-console/backend/routers/inference.py](apps/playground-console/backend/routers/inference.py:140-146)

### Frontend Files
- [apps/ocr-inference-console/src/App.tsx](apps/ocr-inference-console/src/App.tsx)
- [apps/ocr-inference-console/src/components/Sidebar.tsx](apps/ocr-inference-console/src/components/Sidebar.tsx)
- [apps/ocr-inference-console/src/components/Workspace.tsx](apps/ocr-inference-console/src/components/Workspace.tsx)
- [apps/ocr-inference-console/src/api/ocrClient.ts](apps/ocr-inference-console/src/api/ocrClient.ts)

### Documentation
- [docs/artifacts/features/perspective-correction-api-integration.md](docs/artifacts/features/perspective-correction-api-integration.md) - Feature documentation
- [docs/archive/archive_docs/docs/completed_plans/2025-11/2025-11-29_1728_implementation_plan_perspective-correction.md](docs/archive/archive_docs/docs/completed_plans/2025-11/2025-11-29_1728_implementation_plan_perspective-correction.md) - Original implementation

---

**Implementation Status**: 80% Complete
**Estimated Time to Complete**: 30-45 minutes
**Complexity**: Medium (requires careful coordinate transformation logic)
