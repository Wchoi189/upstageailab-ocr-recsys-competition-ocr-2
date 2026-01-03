# Archived Tests

This directory contains tests that have been archived due to dependency issues or obsolescence.

## test_phase2_validation.py

**Archived:** 2026-01-03
**Reason:** Broken dependency chain

This test suite requires `pylsd` (Python Line Segment Detector), which has a broken dependency on the `lsd` module. The `pylsd` package (v0.0.2) attempts to import `from lsd import lsd` but the `lsd` module is not properly packaged or available.

**Tests Affected:**
- `test_salt_pepper_noise_elimination` - Salt & pepper noise reduction validation
- `test_shadow_noise_elimination` - Shadow noise removal validation
- `test_bright_image_darkening` - Brightness adjustment validation
- `test_overall_brightness_validation` - Overall brightness quality validation

**Dependencies Required:**
- `pylsd` (broken)
- `ocr.datasets.preprocessing.advanced_noise_elimination`
- `ocr.datasets.preprocessing.document_flattening`
- `ocr.datasets.preprocessing.intelligent_brightness`

**To Re-enable:**
1. Fix the `pylsd` dependency issue (may require forking and fixing the package)
2. Or refactor `ocr/datasets/preprocessing/detector.py` to not use `pylsd`
3. Move tests back to `tests/integration/`

## test_advanced_noise_elimination.py

**Archived:** 2026-01-03
**Reason:** Same dependency issue as above

Unit tests for the Advanced Noise Elimination module, also requires `pylsd` through the preprocessing pipeline imports.

**To Re-enable:**
Same as above - fix `pylsd` dependency or refactor to avoid it.

---

**Note:** These tests were originally mentioned in CI/test failure reports but could not run due to the missing/broken `pylsd` dependency. The original error reports may have been from a different environment where the dependency was available.
