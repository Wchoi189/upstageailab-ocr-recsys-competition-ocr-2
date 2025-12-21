---
ads_version: "1.0"
title: "Handover 2025 12 16"
date: "2025-12-16 22:11 (KST)"
type: "session_note"
category: "reference"
status: "active"
version: "1.0"
tags: ['session_note', 'reference']
---



# Session Handover: OCR Inference Console Enhancement

## ✅ Completed Work Summary

### Status: All Critical Bugs Fixed & Verified

**Date Completed:** 2025-12-16
**Build Status:** ✅ PASSING (40s build time, 313KB bundle)
**Testing Status:** ✅ Manual testing completed, all issues resolved
**Chrome Console:** ✅ Clean (no port 7242 errors)

### Bug Fixes Completed (5 issues)

#### 1. TypeScript Errors in ocrClient.ts (CRITICAL) ✅
- **Issue:** Undefined `sharedListCheckpoints`, implicit `any` types, unused variables
- **Fix:**
  - Replaced `sharedListCheckpoints` with direct `/health` endpoint fetch
  - Added `ApiRegion` interface for type safety
  - Removed 5 unused timing variables
- **File:** [apps/ocr-inference-console/src/api/ocrClient.ts](../apps/ocr-inference-console/src/api/ocrClient.ts)
- **Lines Changed:** ~30 lines

#### 2. Checkpoint Discovery Race Condition (CRITICAL) ✅
- **Issue:** UI showed "No checkpoints found" until manual browser refresh
- **Root Cause:** Frontend calling API before backend initialized, single attempt with timeout
- **Fix:**
  - Implemented exponential backoff retry logic (1s, 2s, 5s, 10s, 20s delays)
  - Added retry counter in UI ("Waiting for backend... attempt X/6")
  - Automatic retry on empty response or error
  - Fixed `NodeJS.Timeout` type error (changed to `number` for browser env)
- **File:** [apps/ocr-inference-console/src/components/CheckpointSelector.tsx](../apps/ocr-inference-console/src/components/CheckpointSelector.tsx)
- **Lines Changed:** ~60 lines
- **Impact:** No more manual refresh required!

#### 3. Perspective Correction Toggle (HIGH) ✅
- **Issue:** Radio button to toggle original/corrected display not working
- **Root Cause:** Backend not passing `perspective_display_mode` parameter to engine
- **Fix:**
  - Backend: Added `perspective_display_mode=request.perspective_display_mode` to `predict_array()` call
  - Documentation: Updated with Phase 2 implementation details, API examples, coordinate behavior
- **Files:**
  - [apps/ocr-inference-console/backend/main.py](../apps/ocr-inference-console/backend/main.py) (line 332)
  - [docs/artifacts/features/perspective-correction-api-integration.md](../docs/artifacts/features/perspective-correction-api-integration.md)
- **Lines Changed:** 1 line backend, ~80 lines documentation

#### 4. Port 7242 Connection Errors (MEDIUM) ✅
- **Issue:** Chrome console spammed with ERR_CONNECTION_REFUSED to port 7242
- **Root Cause:** Debug telemetry code attempting to POST to non-existent service
- **Fix:**
  - Removed all `// #region agent log ... // #endregion` blocks
  - Removed 27 fetch calls across 4 files
  - Verified 0 references to port 7242 remain
- **Files:**
  - [apps/ocr-inference-console/src/components/PolygonOverlay.tsx](../apps/ocr-inference-console/src/components/PolygonOverlay.tsx) - 15 blocks removed
  - [apps/ocr-inference-console/src/components/Workspace.tsx](../apps/ocr-inference-console/src/components/Workspace.tsx) - 5 blocks removed
  - [packages/console-shared/src/api/inference.ts](../packages/console-shared/src/api/inference.ts) - 3 blocks removed
  - [packages/console-shared/src/api/client.ts](../packages/console-shared/src/api/client.ts) - 4 blocks removed
- **Lines Changed:** ~200 lines removed

#### 5. Build Verification (BONUS) ✅
- **Action:** Fixed final TypeScript compilation issue
- **Issue:** `NodeJS.Timeout` type not available in browser environment
- **Fix:** Changed `let timeoutId: NodeJS.Timeout` to `let timeoutId: number`
- **Result:** Clean TypeScript build with Vite production bundle

### Files Modified Summary

**Total:** 8 files modified, ~400 lines changed

**Backend (1 file):**
- `apps/ocr-inference-console/backend/main.py`

**Frontend (6 files):**
- `apps/ocr-inference-console/src/api/ocrClient.ts`
- `apps/ocr-inference-console/src/components/CheckpointSelector.tsx`
- `apps/ocr-inference-console/src/components/PolygonOverlay.tsx`
- `apps/ocr-inference-console/src/components/Workspace.tsx`
- `packages/console-shared/src/api/inference.ts`
- `packages/console-shared/src/api/client.ts`

**Documentation (1 file):**
- `docs/artifacts/features/perspective-correction-api-integration.md`

---

## 🎯 Next Tasks: Feature Enhancements

### Priority 1: Upload UX Improvement (Issue 5)

**Priority:** Medium
**Complexity:** Low-Medium
**Estimated Time:** 1-2 hours

#### Problem Statement

After uploading an image, the uploaded image covers the upload button located in the center of the preview panel. This blocks consecutive image uploads and requires awkward workarounds. There is no "clear" button to reset the workspace.

#### Design References

**Visual mockups available at:**
- **Thumbnail Box Design:** `.vlm_cache/2025-12-15 12_10_15-AgentQMS-Manager-Dashboard [SSH_ ocr-dev] - Antigravity - Makefile.png`
  - Shows small box with "+" icon at top of preview panel
  - Upload button located in this thumbnail area

- **Modal Design:** `.vlm_cache/2025-12-15 12_10_36-agentqms-manager-dashboard 및 1개 탭 - 파일 탐색기.png`
  - Two-panel modal: Checkpoint dropdown (left) + File upload (right)
  - Entire upload area is drag-and-drop enabled
  - Support for multiple formats: JPEG, PNG, BMP, PDF, TIFF, HEIC, DOCX, XLSX, PPTX (up to 50MB)

#### Implementation Options

**Option A: Move Upload to Thumbnail Box** ⭐ RECOMMENDED (Simpler)

**Key File:** `apps/ocr-inference-console/src/components/Workspace.tsx`

**Current Structure:**
```
Workspace
  ├─ Left Panel (Preview)
  │   ├─ Thumbnail box (top) - currently just shows thumbnail
  │   └─ Main preview area - currently has upload button in center
  └─ Right Panel (Settings/Sidebar)
```

**Proposed Changes:**
1. Move `<input type="file">` from center preview area to thumbnail box
2. Make thumbnail box clickable (wrap in `<label>`)
3. Show "+" icon when no image uploaded
4. Show thumbnail preview in adjacent box after upload
5. Keep main preview area clean for PolygonOverlay display

**Pseudo-code:**
```tsx
{/* Thumbnail box section */}
<div className="flex gap-2 mb-2">
    <label className="w-16 h-16 border-2 border-dashed border-gray-300 rounded flex items-center justify-center cursor-pointer hover:border-blue-500 transition-colors">
        <input
            type="file"
            accept="image/*"
            onChange={handleImageSelected}
            className="hidden"
        />
        <Plus className="text-gray-400" size={24} />
    </label>
    {imageUrl && (
        <div className="w-16 h-16 border rounded overflow-hidden">
            <img src={imageUrl} alt="Thumbnail" className="w-full h-full object-cover" />
        </div>
    )}
</div>

{/* Main preview area */}
{imageUrl ? (
    <PolygonOverlay {...props} />
) : (
    <div className="flex items-center justify-center h-full text-gray-400">
        Select an image using the + button above
    </div>
)}
```

**Option B: Implement Upload Modal** (More Complex)

**New Component:** `apps/ocr-inference-console/src/components/UploadModal.tsx`

**Features:**
- Left panel: Checkpoint dropdown selection
- Right panel: Drag-and-drop file upload area (entire area clickable)
- File format validation (JPEG, PNG, BMP, PDF, TIFF, HEIC, DOCX, XLSX, PPTX)
- File size validation (up to 50MB)
- Close button (X) and Cancel button
- Modal backdrop with click-outside-to-close

**Trigger:** Replace center upload button with "Upload New Image" button that opens modal

**State Management:**
```tsx
const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

<UploadModal
    isOpen={isUploadModalOpen}
    onClose={() => setIsUploadModalOpen(false)}
    onFileSelected={(file, checkpointPath) => {
        handleImageSelected(file);
        setSelectedCheckpoint(checkpointPath);
        setIsUploadModalOpen(false);
    }}
    checkpoints={checkpoints}
    selectedCheckpoint={selectedCheckpoint}
/>
```

**Recommendation:** Start with **Option A** (simpler), gather user feedback, then implement Option B if needed.

---

### Priority 2: Grayscale Preprocessing Option (New Feature)

**Priority:** High (User-requested)
**Complexity:** Medium
**Estimated Time:** 2-3 hours

#### User Requirement

> "Grayscale images are incredibly effective at converting zero prediction images to full prediction capable images."

Add a toggle option for grayscale preprocessing that can be combined with perspective correction to improve OCR accuracy on low-contrast documents.

#### Feature Specification

**UI Control:**
- Add checkbox to Sidebar: "Enable Grayscale Preprocessing"
- Group with existing perspective correction controls
- Should work independently OR in combination with perspective correction

**Processing Pipeline Options:**

**Option A: Grayscale Before Perspective Correction**
```
Input Image → Grayscale Conversion → Perspective Correction → Model Inference
```
- **Pros:** Simpler pipeline, reduced channels earlier
- **Cons:** Rembg (background removal) may behave differently on grayscale

**Option B: Grayscale After Perspective Correction** ⭐ RECOMMENDED
```
Input Image → Perspective Correction → Grayscale Conversion → Model Inference
```
- **Pros:** Rembg works on color image, cleaner separation of concerns
- **Cons:** Slightly more complex pipeline

**Option C: Configurable Order**
- Add radio button group: "Apply grayscale before/after perspective correction"
- **Pros:** Maximum flexibility
- **Cons:** UI complexity, may confuse users

**Recommendation:** Start with **Option B** (post-correction), add Option C only if user testing reveals need.

#### Implementation Plan

Follow the pattern established by perspective correction:

**1. Frontend UI Control**
- **File:** `apps/ocr-inference-console/src/components/Sidebar.tsx`
- **Add:** Checkbox control below perspective correction options

```tsx
{/* Grayscale Preprocessing */}
<label className="flex items-center gap-2">
    <input
        type="checkbox"
        checked={enableGrayscale}
        onChange={(e) => onGrayscaleChange(e.target.checked)}
        className="w-4 h-4"
    />
    <span className="text-sm">Enable Grayscale Preprocessing</span>
</label>
```

**2. State Management**
- **File:** `apps/ocr-inference-console/src/App.tsx`
- **Add:** `const [enableGrayscale, setEnableGrayscale] = useState(false);`
- **Pass to:** Sidebar and Workspace components

**3. API Client**
- **File:** `apps/ocr-inference-console/src/api/ocrClient.ts`
- **Update:** `predict()` function signature

```typescript
predict: async (
    file: File,
    checkpointPath?: string,
    enablePerspectiveCorrection?: boolean,
    perspectiveDisplayMode?: string,
    enableGrayscale?: boolean  // NEW
): Promise<InferenceResponse>
```

**4. Request Model**
- **File:** `apps/shared/backend_shared/models/inference.py`
- **Add:** New field to `InferenceRequest`

```python
class InferenceRequest(BaseModel):
    enable_perspective_correction: bool = Field(default=False)
    perspective_display_mode: str = Field(default="corrected", pattern="^(corrected|original)$")
    enable_grayscale: bool = Field(
        default=False,
        description="Enable grayscale conversion before inference"
    )
```

**5. Backend Endpoint**
- **File:** `apps/ocr-inference-console/backend/main.py`
- **Update:** Pass parameter to engine (line ~332)

```python
result = _inference_engine.predict_array(
    image_array=image,
    binarization_thresh=request.confidence_threshold,
    box_thresh=request.nms_threshold,
    return_preview=True,
    enable_perspective_correction=request.enable_perspective_correction,
    perspective_display_mode=request.perspective_display_mode,
    enable_grayscale=request.enable_grayscale,  # NEW
)
```

**6. Engine Delegation**
- **File:** `ocr/inference/engine.py`
- **Update:** `predict_array()` and `predict_image()` signatures
- **Delegate:** Pass parameter to orchestrator

```python
def predict_array(
    self,
    image_array: np.ndarray,
    # ... existing params
    enable_grayscale: bool = False,
) -> dict[str, Any] | None:
    # ...
    return self._orchestrator.predict(
        image=image_array,
        return_preview=return_preview,
        enable_perspective_correction=enable_perspective_correction or False,
        perspective_display_mode=perspective_display_mode,
        enable_grayscale=enable_grayscale,  # NEW
    )
```

**7. Orchestrator Coordination**
- **File:** `ocr/inference/orchestrator.py`
- **Update:** `predict()` method
- **Pass to:** Preprocessing pipeline

```python
def predict(
    self,
    image: np.ndarray,
    return_preview: bool = True,
    enable_perspective_correction: bool = False,
    perspective_display_mode: str = "corrected",
    enable_grayscale: bool = False,  # NEW
) -> dict[str, Any] | None:
    # ...
    preprocess_result = self.preprocessing_pipeline.process(
        image,
        enable_perspective_correction=enable_perspective_correction,
        perspective_display_mode=perspective_display_mode,
        enable_grayscale=enable_grayscale,  # NEW
    )
```

**8. Preprocessing Pipeline Implementation**
- **File:** `ocr/inference/preprocessing_pipeline.py`
- **Update:** `process()` method
- **Add:** Grayscale conversion logic

```python
def process(
    self,
    image: np.ndarray,
    enable_perspective_correction: bool = False,
    perspective_display_mode: str = "corrected",
    enable_grayscale: bool = False,  # NEW
) -> PreprocessingResult | None:
    # ... existing perspective correction code ...

    # Grayscale conversion (after perspective correction if enabled)
    if enable_grayscale:
        LOGGER.info("Converting image to grayscale")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Expand back to 3 channels for model compatibility
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        LOGGER.info(f"Grayscale conversion complete, shape: {image.shape}")

    # ... continue with existing preprocessing ...
```

**Important Considerations:**

1. **Model Input Format:**
   - Most OCR models expect 3-channel (BGR) input
   - After `COLOR_BGR2GRAY`, expand back: `COLOR_GRAY2BGR`
   - This maintains compatibility while providing grayscale preprocessing benefits

2. **Preview Image:**
   - If grayscale enabled, preview should also be grayscale
   - Store original grayscale image before expanding to BGR
   - Use grayscale for preview generation

3. **EXIF Normalization:**
   - Grayscale conversion should happen AFTER EXIF normalization
   - Current flow: Load → EXIF normalize → Perspective correct → **Grayscale (NEW)** → Resize/Normalize → Model

4. **Metadata Accuracy:**
   - Ensure `original_size`, `processed_size` reflect correct dimensions
   - Grayscale conversion doesn't change dimensions, only channels

#### Testing Strategy

1. **Unit Tests:**
   - Test grayscale conversion with sample images
   - Verify 3-channel output after conversion
   - Test combination with perspective correction

2. **Integration Tests:**
   - Upload low-contrast color image without grayscale → note predictions
   - Upload same image with grayscale enabled → compare predictions
   - Expected: Improved detection on low-contrast documents

3. **Edge Cases:**
   - Already-grayscale images (should be no-op or gracefully handle)
   - Grayscale + perspective correction + original display mode
   - Very large images (performance impact of extra conversion)

---

## 📚 Key Documentation References

### Architecture & System Design

1. **Inference Pipeline Overview**
   - **File:** `docs/architecture/inference-overview.md`
   - **Purpose:** System architecture diagram, component interaction
   - **Relevance:** Understanding where grayscale conversion fits in pipeline

2. **Perspective Correction API** (Reference Pattern)
   - **File:** `docs/artifacts/features/perspective-correction-api-integration.md`
   - **Purpose:** Complete implementation guide for preprocessing feature
   - **Relevance:** Use as template for grayscale feature documentation
   - **Key Sections:**
     - Data flow diagram (adapt for grayscale)
     - API usage examples
     - Coordinate space behavior
     - Implementation status tracking

3. **Shared Backend Contract**
   - **File:** `docs/artifacts/specs/shared-backend-contract.md`
   - **Purpose:** API contract specification
   - **Relevance:** Ensure `InferenceRequest` changes follow contract

### API & Data Models

1. **Inference Request/Response Models**
   - **File:** `apps/shared/backend_shared/models/inference.py`
   - **Classes:** `InferenceRequest`, `InferenceResponse`, `InferenceMetadata`
   - **Relevance:** Add `enable_grayscale` field here

2. **Inference Data Contracts**
   - **File:** `docs/pipeline/inference-data-contracts.md`
   - **Purpose:** Data type specifications for pipeline stages
   - **Relevance:** Understand preprocessing result structure

### Code Structure & Components

1. **Frontend Components**
   - `apps/ocr-inference-console/src/components/Sidebar.tsx` - UI controls
   - `apps/ocr-inference-console/src/components/Workspace.tsx` - Upload area, preview
   - `apps/ocr-inference-console/src/App.tsx` - Root state management

2. **Backend Pipeline**
   - `apps/ocr-inference-console/backend/main.py` - `/api/inference/preview` endpoint
   - `ocr/inference/engine.py` - Thin wrapper (delegates to orchestrator)
   - `ocr/inference/orchestrator.py` - Pipeline coordination
   - `ocr/inference/preprocessing_pipeline.py` - **WHERE GRAYSCALE GOES**

3. **Image Processing Utilities**
   - `ocr/utils/perspective_correction.py` - Reference for image preprocessing
   - `ocr/inference/image_loader.py` - EXIF normalization, loading

### Testing & Validation

1. **Test Images**
   - Low-contrast documents (test grayscale benefit)
   - Skewed documents (test perspective + grayscale combo)
   - Already-grayscale images (test edge case)

---

## 🚀 Continuation Prompt for Next Session

```markdown
# OCR Inference Console - Feature Enhancement Session

## Quick Context
I'm continuing work on the OCR Inference Console after completing critical bug fixes (TypeScript errors, checkpoint retry logic, perspective correction toggle, port 7242 telemetry cleanup). All bugs are verified fixed, build passes, and manual testing is complete.

## Current Tasks

### Task 1: Upload UX Improvement (Issue 5)
**Problem:** After uploading an image, the upload button in the center of the preview panel is blocked by the image. Consecutive uploads are difficult.

**Solution Options:**
1. **Option A (Simpler):** Move upload input to thumbnail box at top of preview panel
   - Reference image: `.vlm_cache/2025-12-15 12_10_15-AgentQMS-Manager-Dashboard [SSH_ ocr-dev] - Antigravity - Makefile.png`
   - Key file: `apps/ocr-inference-console/src/components/Workspace.tsx`

2. **Option B (Complex):** Implement modal dialog with checkpoint selection + file upload
   - Reference image: `.vlm_cache/2025-12-15 12_10_36-agentqms-manager-dashboard 및 1개 탭 - 파일 탐색기.png`
   - Create: `apps/ocr-inference-console/src/components/UploadModal.tsx`

**Recommendation:** Start with Option A, then Option B if user feedback requires it.

### Task 2: Add Grayscale Preprocessing Option (New Feature - High Priority)
**User Insight:** "Grayscale images are incredibly effective at converting zero prediction images to full prediction capable images."

**Requirements:**
1. Add UI checkbox to Sidebar: "Enable Grayscale Preprocessing"
2. Should work independently AND with perspective correction
3. Add `enable_grayscale: bool` field to `InferenceRequest` model
4. Implement grayscale conversion in preprocessing pipeline
5. Preview image should reflect grayscale state
6. Document feature similar to perspective-correction-api-integration.md

**Key Questions to Address:**
- Should grayscale conversion happen before or after perspective correction?
  - **Recommended:** After (so rembg works on color image)
- Should preview image be grayscale or color?
  - **Recommended:** Match processing (grayscale preview if enabled)
- How to handle model input format?
  - **Solution:** Convert BGR→GRAY→BGR to maintain 3-channel input

**Implementation Pattern (Follow Perspective Correction):**
1. Frontend UI control (Sidebar.tsx) →
2. State management (App.tsx) →
3. API call with parameter (ocrClient.ts) →
4. Backend request model (inference.py) →
5. Endpoint handler (main.py) →
6. Engine delegation (engine.py) →
7. Orchestrator coordination (orchestrator.py) →
8. Preprocessing pipeline (preprocessing_pipeline.py) ← **IMPLEMENT HERE**

**Reference Documentation:**
- Pattern: `docs/artifacts/features/perspective-correction-api-integration.md`
- Pipeline: `docs/architecture/inference-overview.md`
- Models: `apps/shared/backend_shared/models/inference.py`
- Preprocessing: `ocr/inference/preprocessing_pipeline.py`

## Implementation Order
1. **Start with:** Review design references for Upload UX (both images)
2. **Then:** Propose implementation approach for both features
3. **Ask:** Any clarifying questions about requirements
4. **Consider:** Should features be implemented sequentially or in parallel?

## Session Goals
- ✅ Upload UX improved (users can easily upload consecutive images)
- ✅ Grayscale preprocessing option functional
- ✅ Both features work in combination with existing perspective correction
- ✅ Documentation updated
- ✅ Build passes, no regressions

Let's begin! What questions do you have before we start?
```

---

## 📊 Session Statistics

| Metric | Value |
|--------|-------|
| **Session Duration** | ~3 hours |
| **Issues Resolved** | 5 bugs + 1 build fix |
| **Files Modified** | 8 files |
| **Lines Changed** | ~400 lines |
| **TypeScript Errors Fixed** | 7 errors |
| **Telemetry Blocks Removed** | 27 blocks |
| **Build Status** | ✅ PASSING |
| **Build Time** | 40.11s |
| **Bundle Size** | 313KB (100KB gzipped) |
| **Console Errors** | 0 (clean) |
| **Manual Tests** | ✅ PASSED |

---

## 🎯 Success Criteria for Next Session

### Upload UX Improvement
- ✅ User can upload consecutive images without obstruction
- ✅ Clear visual feedback for upload area (hover states, cursor changes)
- ✅ Optional: Drag-and-drop support
- ✅ Optional: File format validation with user-friendly error messages
- ✅ No breaking changes to existing upload workflow

### Grayscale Preprocessing Feature
- ✅ Checkbox control works independently
- ✅ Checkbox works in combination with perspective correction
- ✅ Preview image displays correctly (grayscale or color based on toggle)
- ✅ Inference accuracy improves for low-contrast documents
- ✅ API documented with request/response examples
- ✅ No breaking changes to existing API contracts
- ✅ Model still receives 3-channel input (BGR format)

### Quality Gates
- ✅ Frontend builds without errors
- ✅ Backend starts without errors
- ✅ No console warnings or errors
- ✅ Manual testing confirms both features work
- ✅ Documentation is complete and accurate

---

## ⚠️ Potential Gotchas & Recommendations

### Upload UX
1. **File Input Accessibility:**
   - Ensure `<label>` wraps `<input>` for proper keyboard navigation
   - Add `aria-label` for screen readers
   - Test with keyboard-only navigation (Tab, Enter)

2. **Thumbnail Display:**
   - Use `object-cover` or `object-contain` for proper aspect ratio
   - Handle very wide/tall images gracefully
   - Consider lazy loading for large images

3. **State Management:**
   - Clear previous predictions when new image uploaded
   - Reset inference state (loading, errors, results)
   - Handle rapid consecutive uploads (debounce or disable during processing)

### Grayscale Preprocessing
1. **Conversion Timing:**
   - **Recommended:** Grayscale AFTER perspective correction
   - Reason: Rembg (background removal) works better on color images
   - If needed before: Test rembg behavior on grayscale inputs first

2. **Preview Image Format:**
   - If grayscale enabled, preview should be grayscale (consistent UX)
   - Store original grayscale before expanding to BGR for model
   - Use grayscale image for JPEG preview encoding

3. **Model Input Compatibility:**
   - Models typically expect 3-channel (BGR) input
   - After `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`, expand back:
     ```python
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
     ```
   - This maintains compatibility while providing preprocessing benefits

4. **EXIF Normalization Order:**
   - Current: Load → EXIF normalize → Perspective correct → Resize → Model
   - Updated: Load → EXIF normalize → Perspective correct → **Grayscale** → Resize → Model
   - Grayscale conversion should NOT affect EXIF-normalized orientation

5. **Edge Cases:**
   - Already-grayscale images (1-channel): Handle gracefully (no-op or expand to 3-channel)
   - Alpha channel images (RGBA): Convert to BGR first, then grayscale
   - Very large images: Profile performance impact of extra conversion step

6. **Metadata Accuracy:**
   - `original_size`, `processed_size` should remain accurate
   - Grayscale conversion changes channels, not dimensions
   - Update any channel-related metadata if exposed

### Testing Strategy
1. **Upload UX:**
   - Upload multiple images in sequence without errors
   - Drag-and-drop if implemented
   - Keyboard navigation (Tab to upload button, Enter to trigger)
   - Mobile/touch devices (if applicable)

2. **Grayscale:**
   - Low-contrast color document → grayscale ON → compare predictions
   - High-contrast color document → grayscale ON → ensure no regression
   - Already-grayscale image → grayscale ON → verify no errors
   - Grayscale + perspective correction combined → verify both work
   - Grayscale + perspective correction + original display mode → complex case

3. **Performance:**
   - Large images (4K+) with grayscale enabled → measure latency
   - Profile memory usage with grayscale conversion
   - Compare inference time: color vs grayscale

---

## 📝 Next Session Checklist

Before starting implementation:
- [ ] Review both design reference images (`.vlm_cache/2025-12-15 12_10_*`)
- [ ] Read `docs/artifacts/features/perspective-correction-api-integration.md` (implementation pattern)
- [ ] Read `ocr/inference/preprocessing_pipeline.py` (where grayscale goes)
- [ ] Decide: Upload UX Option A or B? (Recommend A first)
- [ ] Decide: Grayscale before or after perspective correction? (Recommend after)

During implementation:
- [ ] Create todo list with TodoWrite tool for tracking progress
- [ ] Test each component individually before integration
- [ ] Update documentation as features are implemented (not at the end)
- [ ] Run `npm run build` after each frontend change
- [ ] Test manually after each milestone

After implementation:
- [ ] Full manual testing of both features
- [ ] Verify no regressions in existing features
- [ ] Update this handover document with completion status
- [ ] Create PR or commit with descriptive message

---

## 🔗 Quick Links

**Implementation Plan:** `/home/vscode/.claude/plans/jolly-beaming-thunder.md`

**Key Files to Modify:**
- Frontend: `apps/ocr-inference-console/src/components/Workspace.tsx`
- Frontend: `apps/ocr-inference-console/src/components/Sidebar.tsx`
- Frontend: `apps/ocr-inference-console/src/App.tsx`
- Frontend: `apps/ocr-inference-console/src/api/ocrClient.ts`
- Backend: `apps/shared/backend_shared/models/inference.py`
- Backend: `apps/ocr-inference-console/backend/main.py`
- Backend: `ocr/inference/engine.py`
- Backend: `ocr/inference/orchestrator.py`
- Backend: `ocr/inference/preprocessing_pipeline.py`

**Documentation to Create:**
- `docs/artifacts/features/grayscale-preprocessing.md` (similar to perspective-correction)

---

## 🏁 Ready for Next Session!

This handover document contains all context needed to continue feature development. The next developer should:

1. Read continuation prompt above
2. Review design references
3. Ask clarifying questions
4. Begin implementation with Upload UX (Option A recommended)
5. Then implement Grayscale preprocessing
6. Test thoroughly
7. Update documentation

**Current State:** ✅ All critical bugs fixed, build passing, ready for feature work
**Next State:** 🎯 Upload UX + Grayscale preprocessing implemented and tested

Good luck with the next session! 🚀
