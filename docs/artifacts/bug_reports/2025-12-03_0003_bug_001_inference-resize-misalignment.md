---
ads_version: "1.0"
title: "001 Inference Resize Misalignment"
date: "2025-12-04 12:43 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---



# Bug Report: Inference preview misalignment for resized images

## Bug ID
BUG-001

<!-- REQUIRED: Fill these sections when creating the initial bug report -->
## Summary
Inference preview overlays in Inference Studio show **size-dependent alignment**: images that are enlarged by the preprocessing/resizing pipeline exhibit horizontal misalignment between polygons and text, while images that remain close to their native resolution are correctly aligned.

## Environment
- **OS**: Linux (WSL2 dev container)
- **Python Version**: 3.x (project virtualenv, `uv`-managed)
- **Backend App**: `apps.backend.services.playground_api.app:app` (FastAPI)
- **Frontend App**: Next.js "Inference Studio" under `apps/frontend`
- **Model / Engine**:
  - `ui/utils/inference/engine.py`
  - Preprocessing configuration derived from checkpoint configs via `ui/utils/inference/config_loader.py` (`PreprocessSettings.image_size`)
- **Browser**: Chromium-based browser (e.g., Chrome/Edge) during local development

## Steps to Reproduce
1. Start the stack from the project root:
   - `make fs` (alias for `make stack-dev`) to launch the FastAPI backend and Next.js frontend.
2. Open the Inference Studio page (`Inference` route in the Next.js frontend).
3. Upload a **validation** image known to be heavily resized, such as `drp.en_ko.in_house.selectstar_003062.jpg`, and run inference with any available checkpoint.
4. Observe the preview canvas: the document appears enlarged (e.g., filling the viewport), and polygon overlays show a noticeable horizontal deviation from the underlying text.
5. Repeat with several **training** images:
   - Example of misaligned enlarged image: `drp.en_ko.in_house.selectstar_000858.jpg` (train).
   - Another misaligned enlarged example: `drp.en_ko.in_house.selectstar_002678.jpg` (train).
   - Example of correctly aligned, reasonably sized image: `drp.en_ko.in_house.selectstar_002123.jpg` (train).
   - Example of correct alignment despite being from train: `drp.en_ko.in_house.selectstar_002723.jpg`.
6. Compare the overlay alignment between these images; note that misalignment correlates strongly with images that are aggressively resized/enlarged by the preprocessing pipeline.

## Expected Behavior
- Regardless of input image size or any preprocessing/resizing, the preview overlay should remain **pixel-perfectly aligned** with the displayed image.
- Images that are upscaled or downscaled by the inference pipeline should still have polygons rendered at the correct positions in the preview.

## Actual Behavior
- Images that remain close to their native "reasonable" resolution in the preview (e.g., `drp.en_ko.in_house.selectstar_002123.jpg`, train) show **correct alignment** between polygons and text.
- Images that appear **excessively enlarged** in the preview (e.g., `drp.en_ko.in_house.selectstar_003062.jpg` from validation, `drp.en_ko.in_house.selectstar_000858.jpg` and `002678.jpg` from train) exhibit a consistent horizontal shift between polygons and underlying text.
- The amount of misalignment appears to depend on the degree of resizing performed: more aggressive resize / padding leads to more noticeable horizontal deviation.

## Error Messages
```
No explicit errors are raised in the frontend or backend.
The issue manifests as visual misalignment tied to image resizing, not as an exception.
```

## Screenshots/Logs
- Representative overlay image (from previous investigation): `logs/ui/image.png`.
- Relevant code paths:
  - Preprocess configuration extraction:
    - `ui/utils/inference/config_loader.py` (`_extract_preprocess_settings`, `PreprocessSettings.image_size`)
  - Preprocess + resize pipeline:
    - `ui/utils/inference/preprocess.py` (`build_transform` → `transforms.Resize(settings.image_size)`)
  - Inference engine:
    - `ui/utils/inference/engine.py` (`InferenceEngine._predict_from_array`)

## Impact
- **Severity**: Medium
- **Affected Users**:
  - Developers and analysts using Inference Studio to visually inspect OCR predictions across mixed-resolution datasets (train/val/test).
- **Workaround**:
  - For debugging, focus on images whose preview size is closer to native resolution (these tend to align correctly).
  - When investigating problematic checkpoints, cross-check overlays using offline scripts that render polygons in the engine’s native coordinate system.

<!-- OPTIONAL: Resolution sections - fill these during investigation and fixing -->
## Investigation

### Root Cause Analysis
- **Cause (confirmed)**:
  - **Critical mismatch**: The preprocessing pipeline uses `transforms.Resize(image_size)` which directly resizes images to a fixed size (potentially distorting aspect ratio), while the postprocessing code (`decode_polygons_with_head`, `fallback_postprocess`) assumes **LongestMaxSize + PadIfNeeded** transforms (which preserve aspect ratio and pad to a square, typically 640x640).
  - This mismatch causes the inverse matrix computation in postprocessing to be incorrect, leading to polygon coordinates being mapped to the wrong coordinate space.
  - Some inputs (e.g., tall receipts like `drp.en_ko.in_house.selectstar_003051.jpg`) are affected more severely because the aspect ratio distortion from `transforms.Resize` differs significantly from the aspect-ratio-preserving LongestMaxSize transform that postprocessing expects.
  - The preview image coordinate system (currently `image.shape` after perspective correction) matches what postprocessing maps polygons back to, but the underlying coordinate mapping is incorrect due to the transform mismatch.
- **Location**:
  - Preprocess configuration and resizing:
    - `ui/utils/inference/config_loader.py` – `_extract_preprocess_settings` sets `image_size` from `preprocessing.target_size` or transform configs.
    ```151:183:ui/utils/inference/config_loader.py def _extract_preprocess_settings(config: Any) -> PreprocessSettings:
        image_size = DEFAULT_IMAGE_SIZE
        ...
        preprocessing = _get_attr(config, "preprocessing")
        if preprocessing and (target_size := _coerce_tuple(_get_attr(preprocessing, "target_size"))):
            image_size = target_size
    ```
    - `ui/utils/inference/preprocess.py` – `build_transform` uses `transforms.Resize(settings.image_size)`, which normalizes all inputs to the configured target size.
    ```22:33:ui/utils/inference/preprocess.py def build_transform(settings: PreprocessSettings):
        ...
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(settings.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=settings.normalization.mean, std=settings.normalization.std),
            ]
        )
    ```
  - Inference engine:
    - `ui/utils/inference/engine.py` – `_predict_from_array` applies optional perspective correction and then `preprocess_image` / `transforms.Resize`, passing the resized image into the model and post-processing, which returns polygons in the resized space.
  - Frontend preview:
    - `apps/frontend/src/components/inference/InferencePreviewCanvas.tsx` – draws the preview image and overlays; previous fixes (BUG-001) ensure the canvas uses the engine’s preprocessed image when available, but size-dependent quirks indicate that some images still experience scaling mismatches between backend resizing and frontend rendering.
- **Trigger**:
  - Input images whose dimensions or aspect ratios force large scaling factors or asymmetric padding when resized to the configured `target_size` (e.g., very tall receipts or highly elongated documents).

### Related Issues
- **BUG-001 (overlay misalignment)** – earlier bug report documenting general overlay misalignment between original images and polygons; this resize-focused bug refines that analysis to cases driven specifically by aggressive resizing and scaling behaviour.
- Historical issues around geometry and polygon coordinate handling documented in:
  - `docs/artifacts/implementation_plans/2025-11-12_0226_data-contract-enforcement-implementation.md`
  - `docs/artifacts/bug_reports/archive/*polygon*` (e.g., out-of-bounds coordinates).

## Proposed Solution

### Fix Strategy
- Ensure that **a single, well-defined resize/scale factor** is used consistently from preprocessing through to preview rendering:
  - Track the exact transformation (scale factors and padding/crop) applied when resizing from original resolution to `PreprocessSettings.image_size`.
  - Use this metadata to:
    - (A) map polygons back to the preview image’s coordinate system, or
    - (B) generate a preview image that is guaranteed to share the same coordinate system as the decoded polygons (preferred, and partially addressed in BUG-001).
- Audit and, if necessary, standardize `preprocessing.target_size` and transform definitions across training/validation configs to minimize surprise scaling differences between splits.

### Implementation Plan
1. **Config / resize survey**:
   - Enumerate model configs used in Inference Studio and document their `preprocessing.target_size` / transform chains (including any dataset-specific overrides).
   - Identify which configs correspond to the problematic images (e.g., `drp.en_ko.in_house.selectstar_*` samples).
2. **Transform metadata capture**:
   - Extend the preprocessing pipeline (or `PreprocessSettings`) to compute and expose the transformation parameters for each image:
     - original width/height
     - resized width/height
     - scale factors and padding offsets.
3. **Coordinate normalization**:
   - Update the inference engine to either:
     - return polygons already mapped to the preview image resolution, or
     - return polygons plus explicit transformation metadata that the frontend can use to scale coordinates before drawing.
4. **Preview behaviour review**:
   - Confirm that `InferencePreviewCanvas` always uses the correct image (original vs. preprocessed) for a given mode, avoiding any extra client-side resizes that reintroduce drift.
5. **Config harmonization**:
   - Where feasible, align `preprocessing.target_size` and related resize settings across train/validation configs so that similar documents receive similar scale factors.

### Testing Plan
1. **Unit tests (geometry / transforms)**:
   - Add unit tests around resize + padding utilities (e.g., using `ocr.utils.geometry_utils`) to confirm that the computed transformation metadata accurately reproduces the engine’s behaviour for a range of aspect ratios.
2. **API-level tests**:
   - For synthetic images of known size and positions, run `/inference/preview` and assert that polygons scale linearly with changes in input resolution and `target_size`, without introducing extra offsets.
3. **Dataset-sampled regression tests**:
   - Build a small curated set of images:
     - misaligned examples: `drp.en_ko.in_house.selectstar_003062.jpg`, `000858.jpg`, `002678.jpg`
     - correctly aligned examples: `002123.jpg`, `002723.jpg`.
   - Add automated checks (or visual golden tests) to assert that overlays remain aligned after fixes, across both train and validation splits.
4. **Manual validation in Inference Studio**:
   - Re-run Inference Studio with the curated sample set and visually confirm that previously enlarged, misaligned images now show correct alignment.

## Implementation Summary (BUG-001)

**Fixed**: Updated preprocessing pipeline to use LongestMaxSize + PadIfNeeded matching postprocessing assumptions.

**Changes made**:
1. **`ui/utils/inference/preprocess.py`**:
   - Removed `transforms.Resize` from `build_transform()`
   - Implemented LongestMaxSize + PadIfNeeded in `preprocess_image()` using OpenCV:
     - Scales longest side to `target_size` preserving aspect ratio
     - Pads to `target_size x target_size` with top_left position (padding at bottom/right)
   - Added `target_size` parameter to `preprocess_image()` (defaults to 640, matching postprocessing)
   - Fixed to work on a copy of the input image to avoid in-place modification
   - Added `return_processed_image` parameter to return the exact processed BGR image for preview

2. **`ui/utils/inference/engine.py`**:
   - Updated to extract `target_size` from `PreprocessSettings.image_size` and pass to `preprocess_image()`
   - Fixed to capture `original_shape` BEFORE preprocessing (since `preprocess_image` now works on a copy)
   - **Consistent output resolution**: Uses exact processed image from `preprocess_image()` (640x640) instead of reconstructing it
   - **Polygon coordinate mapping**: Maps polygons from original space (where postprocessing returns them) forward to resized/padded preview space using direct scaling (simplified from matrix transform)
   - Added dimension verification to ensure preview image matches expected target_size

3. **`apps/frontend/src/components/inference/InferencePreviewCanvas.tsx`**:
   - **Canvas CSS fixes**: Added `display: block` and centered container to prevent stretching issues
   - **Canvas clearing**: Added explicit `clearRect()` before drawing to prevent artifacts
   - Canvas internal dimensions match image exactly for 1:1 pixel mapping

**Expected impact**:
- Aspect ratio preserved during resize (eliminates "enlarged" appearance)
- Consistent coordinate mapping between preprocessing and postprocessing
- Polygon overlays should align correctly with preview images
- **Output resolution consistency**: All preview images are now consistently 640x640 (resized/padded), eliminating size variations that caused "enlarged" appearance
- Preview images match the coordinate space of polygons (both in resized/padded 640x640 space)

**Verification needed**: Test with problematic images (e.g., `drp.en_ko.in_house.selectstar_003051.jpg`) to confirm alignment and that "enlarged" appearance is resolved.

**Known Issues**:
- Portrait images (720x1280, 960x1280) still show rightward deviation in overlays
- Coordinate mapping math verified correct (forward scales are exact inverses of postprocessing scales)
- Frontend fixes applied but not taking effect - may be wrong viewer component
- **CRITICAL**: Need to identify which inference viewer is active (Next.js vs Streamlit)
- See session handover: `docs/artifacts/session_handovers/2025-12-03_BUG-001_inference-overlay-misalignment-handover.md`

## Status
- [x] Confirmed
- [x] Investigating
- [x] Fix in progress
- [x] Fixed (preprocessing updated to use LongestMaxSize+PadIfNeeded matching postprocessing)
- [ ] Verified

## Assignee
- TBD (to be assigned by maintainers / sprint owner)

## Priority
- Medium (important for trustworthy visual QA across mixed-resolution datasets, but core inference pipeline remains functional)

---

*This bug report follows the project's standardized format for issue tracking.*
