---
type: "bug_report"
category: "troubleshooting"
status: "active"
severity: "medium"
version: "1.0"
tags: ['bug', 'issue', 'troubleshooting']
title: "Inference Studio overlay misaligned with original image"
date: "2025-12-02 23:13 (KST)"
---

# Bug Report: Inference Studio overlay misaligned with original image

## Bug ID
BUG-001

<!-- REQUIRED: Fill these sections when creating the initial bug report -->
## Summary
Inference Studio's visual overlay of OCR predictions is horizontally misaligned with the underlying original image when running single-image inference; rotation appears correct, but bounding boxes and labels are shifted sideways, as if the overlay has a different horizontal padding or crop than the displayed image.

## Environment
- **OS**: Linux (WSL2 dev container)
- **Python Version**: 3.x (project virtualenv, see `Makefile` / `uv`-managed env)
- **Backend App**: `apps.backend.services.playground_api.app:app` (FastAPI)
- **Frontend App**: Next.js "Inference Studio" under `apps/frontend`
- **Browser**: Chromium-based browser (e.g., Chrome/Edge) during local development
- **Relevant Configs**:
  - `configs/ui/inference.yaml`
  - AgentQMS-enabled docs/artifacts workflow

## Steps to Reproduce
1. From the project root, start the full stack:
   - `make fs` (alias for `make stack-dev`) to launch the FastAPI backend and Next.js frontend.
2. Open the Inference Studio page in the browser (Next.js route `Inference`, reachable via the main playground UI).
3. Upload a receipt or document image (e.g., a vertical receipt similar to `logs/ui/image.png`) using the "Upload Image" control.
4. Select any available OCR checkpoint in the "Checkpoint Picker".
5. Wait for inference to complete and observe the "Inference Preview" canvas on the right-hand side of the page.

## Expected Behavior
- The prediction overlay (green polygons, confidence labels, and text labels) should align exactly with the underlying original image drawn in the preview canvas.
- Bounding boxes should tightly enclose the corresponding text regions, with no visible horizontal or vertical offset between boxes and printed text.

## Actual Behavior
- Rotation of the preview image appears correct, and the general shape of the detections matches the document.
- However, all overlay polygons and labels are shifted horizontally by a roughly constant offset relative to the underlying image (e.g., boxes appear displaced to the right or left of the true text positions).
- The misalignment is most noticeable near the left/right edges of the receipt, where boxes consistently miss the printed text they are supposed to cover.

## Error Messages
```
No explicit frontend or backend error messages are raised.
The issue manifests purely as a visual misalignment between the rendered image and the polygon overlay.
```

## Screenshots/Logs
- Example captured overlay image: `logs/ui/image.png`.
- Backend inference router:
  - `apps/backend/services/playground_api/routers/inference.py`
- Frontend components:
  - `apps/frontend/src/pages/Inference.tsx`
  - `apps/frontend/src/components/inference/InferencePreviewCanvas.tsx`

## Impact
- **Severity**: Medium
- **Affected Users**: Anyone using the Inference Studio web UI to visually inspect OCR predictions, debug checkpoints, or validate perspective correction.
- **Workaround**:
  - Numerical metrics (e.g., detection counts, latency) still function, but visual inspection of prediction quality is unreliable.
  - As a temporary measure, developers can inspect overlays using other tooling that renders polygons in the correct coordinate system (e.g., offline scripts) or by manually reprojecting coordinates.

<!-- OPTIONAL: Resolution sections - fill these during investigation and fixing -->
## Investigation

### Root Cause Analysis
- **Cause**:
  - The frontend assumes that polygon coordinates from the `/inference/preview` API are expressed in the same pixel coordinate system as the raw uploaded image.
  - In practice, the inference engine applies preprocessing (including optional perspective correction, resizing, and potential padding/cropping) before running the model, and polygons are produced in this *preprocessed* image space.
  - The API forwards these polygon coordinates directly to the frontend without mapping them back to the original image coordinate system.
- **Location**:
  - **Frontend overlay rendering**:
    - `apps/frontend/src/components/inference/InferencePreviewCanvas.tsx`
      - Sets `canvas.width` / `canvas.height` to the raw `ImageBitmap` dimensions and draws the original image at `(0, 0)`.
      - Iterates over `result.regions` and calls `drawPolygon` with each `TextRegion.polygon`, using coordinates as-is.
  - **Backend polygon generation**:
    - `apps/backend/services/playground_api/routers/inference.py`
      - `_load_image` decodes the uploaded image.
      - `InferenceEngine.predict_array(...)` runs with `image_array` that may have undergone perspective correction and resizing (see `ui/utils/inference/preprocess.py`).
      - `_parse_inference_result` converts model output (preprocessed-space polygons) into API-level `TextRegion` objects without coordinate remapping.
- **Trigger**:
  - Any Inference Studio request where preprocessing alters the effective field of view compared to the raw upload, particularly when:
    - perspective correction or background removal is enabled, or
    - resizing with padding/cropping changes horizontal margins.

### Related Issues
- Historical bugs around coordinate mismatches and polygon handling (see archived bug reports under `docs/artifacts/bug_reports/archive/`, e.g., empty predictions / invalid polygon coordinates).
- Perspective-correction and preprocessing behaviour documented in `ui/utils/inference/preprocess.py` and related implementation plans under `docs/artifacts/implementation_plans/`.

## Proposed Solution

### Fix Strategy
- Normalize coordinate systems so that the frontend always overlays polygons in the same space as the displayed image.
- Preferred approach:
  - Track the transformation from original image → preprocessed image (including perspective correction, resizing, and any padding/cropping) inside the inference pipeline.
  - Apply the inverse of this transform to polygon vertices before returning them from `/inference/preview`, so `TextRegion.polygon` is expressed in original-image coordinates.
- Alternative approach (if tracking/inverting transforms is non-trivial):
  - Return a preprocessed image (e.g., as a base64 PNG or static file URL) along with polygons in that coordinate system, and update `InferencePreviewCanvas` to display this preprocessed image instead of the raw upload.

### Implementation Plan
1. **Backend analysis**:
   - Inspected `InferenceEngine.predict_array` and related preprocessing utilities (e.g., `ui/utils/inference/preprocess.py`) to document the exact image transforms applied before inference, including perspective correction and resizing.
2. **Coordinate mapping design**:
   - Decided to treat the engine’s **preprocessed BGR image (after optional perspective correction and resize)** as the canonical coordinate space for polygons and previews.
3. **API update**:
   - Updated the inference pipeline so `_predict_from_array` in `ui/utils/inference/engine.py` attaches a `preview_image_base64` PNG representing the exact image used for polygon decoding (BUG-001).
   - Extended `/inference/preview` to expose this `preview_image_base64` field via `InferencePreviewResponse`, ensuring frontends can render overlays on the correct image without additional transforms.
4. **Frontend verification**:
   - Updated `InferencePreviewCanvas.tsx` to prefer the backend-provided preview image when available, sizing the canvas to that image and drawing polygons in the same coordinate system.
   - Performed manual checks using known problematic and non-problematic images (including `logs/ui/image.png`) to confirm that horizontal misalignment is eliminated when the preview image is used.
5. **Documentation**:
   - Documented the canonical preview/coordinate contract for `/inference/preview` in this bug report and the resize-focused companion BUG report, referencing the BUG-001 changes in engine, API, and frontend.

### Testing Plan
1. **Unit tests (backend)**:
   - Add tests for the new coordinate-mapping utilities, verifying that a synthetic polygon in preprocessed space is mapped back to the correct position in original space for common transformations (pure resize, crop + resize, perspective warp).
2. **Integration tests (API)**:
   - Create a deterministic test image (e.g., synthetic grid or rectangle markers) and a mocked inference result with known polygon coordinates in preprocessed space; assert that `/inference/preview` returns polygons aligned with expected original-image coordinates.
3. **End-to-end tests (frontend)**:
   - Extend Playwright (or existing E2E) tests for Inference Studio to:
     - Upload a known test image.
     - Capture the rendered canvas and compare the overlay positions against reference expectations within a pixel tolerance.
4. **Regression validation**:
   - Manually validate with real receipts and documents (including `logs/ui/image.png`) to confirm that horizontal misalignment is eliminated across multiple checkpoints and preprocessing settings.

## Status
- [x] Confirmed
- [x] Investigating
- [x] Fix in progress
- [x] Fixed
- [ ] Verified

## Assignee
- TBD (to be assigned by maintainers or current sprint owner)

## Priority
- Medium (important for developer productivity and visual QA, but does not break core inference pipeline)

---

*This bug report follows the project's standardized format for issue tracking.*
