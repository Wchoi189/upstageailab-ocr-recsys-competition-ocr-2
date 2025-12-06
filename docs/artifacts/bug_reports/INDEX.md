# 🐛 Bug Reports

Documentation of bugs, issues, and their resolution.

**Last Updated**: 2025-12-06 20:09:12
**Total Artifacts**: 55

## Active (55)

- [001 Dominant Edge Extension Failure](2025-11-28_0000_BUG_001_dominant-edge-extension-failure.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report)
- [001 Inference Studio Offsets Data Contract](2025-12-03_1100_BUG_001_inference-studio-offsets-data-contract.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date**: 2025-12-03 **Status**: ✅ RESOLVED (2025-12-03) — Overlay alignment fixed, coordinate system contract established
- [002 Inference Studio Visual Padding Mismatch](2025-12-03_2300_BUG_002_inference-studio-visual-padding-mismatch.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date**: 2025-12-03 **Status**: In Progress — Visual padding appears uneven despite correct calculations
- [001 Inference Padding Scaling Mismatch](2025-01-01_1200_BUG_001_inference-padding-scaling-mismatch.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - Last Updated: 2025-10-21 During inference, clean images occasionally return 0 predictions and the UI displays extremely large images. Model inputs are correctly resized to 640 and padded to 640×640, b...
- [002 Fix Findings](2025-01-02_1200_BUG_002_fix-findings.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date:** October 10, 2025 **Bug ID:** BUG-2025-002
- [002 Pil Image Transform Crash](2025-01-02_1200_BUG_002_pil-image-transform-crash.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - def __call__(self, image, polygons): if isinstance(image, Image.Image):
- [003 Albumentations Contract Violation](2025-01-03_1200_BUG_003_albumentations-contract-violation.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - class GoodTransform(A.ImageOnlyTransform): def __init__(self, param, always_apply=False, p=1.0):
- [003 Fix Findings](2025-01-03_1200_BUG_003_fix-findings.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date:** October 10, 2025 **Bug ID:** BUG-2025-003
- [004 Polygon Shape Dimension Error](2025-01-04_1200_BUG_004_polygon-shape-dimension-error.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - def __call__( self,
- [010 Empty Predictions Inference](2025-01-10_1200_BUG_010_empty-predictions-inference.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report)
- [010 Empty Predictions Resolution Plan](2025-01-10_1200_BUG_010_empty-predictions-resolution-plan.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - This document outlines a systematic approach to resolve the empty predictions issue identified in [BUG-2025-010_empty_predictions_inference.md](../bug_reports/BUG-2025-010_empty_predictions_inference....
- [011 Inference Ui Coordinate Transformation](2025-01-11_1200_BUG_011_inference-ui-coordinate-transformation.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - return self._remap_predictions_if_needed( decoded,
- [Bug 2025 012 Streamlit Duplicate Element Key](BUG-2025-012_streamlit_duplicate_element_key.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date:** October 21, 2025 **Bug ID:** BUG-2025-012
- [Bug 2025 10 09 001 Canonical Size Typeerror](BUG-2025-10-09-001_canonical_size_typeerror.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Bug ID**: BUG-2025-10-09-001 **Status**: ✅ **RESOLVED**
- [Bug 2025 10 12 001 Checkpoint Naming Duplication](BUG-2025-10-12-001_checkpoint_naming_duplication.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - epoch = trainer.current_epoch step = trainer.global_step
- [Bug 20251109 002 Code Changes](BUG-20251109-002-code-changes.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - This document tracks all code changes made to fix BUG-20251109-002. All changes are indexed with the bug ID for proper tracking and version control. - **Bug ID:** BUG-20251109-002
- [Bug 20251109 002 Cuda Illegal Memory Access In Bce Loss Computation](BUG-20251109-002-cuda-illegal-memory-access-in-bce-loss-computation.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - BUG-20251109-002 CUDA illegal memory access error occurs during training in BCE loss computation when computing `positive.sum().item()` at line 31 of `ocr/models/loss/bce_loss.py`. The error causes tr...
- [Bug 20251110 001 Dice Loss Assertion Stops Training](BUG-20251110-001_dice-loss-assertion-stops-training.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report)
- [Bug 20251110 001 Out Of Bounds Polygon Coordinates In Training Dataset](BUG-20251110-001_out-of-bounds-polygon-coordinates-in-training-dataset.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - BUG-20251110-001 Training dataset contains 867 images (26.5%) with out-of-bounds Y coordinates exceeding image height, causing training errors.
- [Bug 20251110 002 Code Changes](BUG-20251110-002-code-changes.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - This document tracks all code changes made to fix BUG-20251110-002. All changes are indexed with the bug ID for proper tracking and version control. - **Bug ID:** BUG-20251110-002
- [Bug 20251110 002 Nan Gradients From Step Function Overflow](BUG-20251110-002-nan-gradients-from-step-function-overflow.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - BUG-20251110-002 Training crashes at step ~122 with widespread NaN/Inf gradients propagating from the differentiable binarization step function in `DBHead._step_function()`. The step function uses `to...
- [Bug 20251110 002 Cudnn Status Execution Failed During Training](BUG-20251110-002_cudnn-status-execution-failed-during-training.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report)
- [Bug 20251110 003 Cudnn Workspace Memory Exhaustion](BUG-20251110-003-cudnn-workspace-memory-exhaustion.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - BUG-20251110-003 Training fails with cuDNN "FIND was unable to find an engine to execute this computation" error on RTX 3060 12GB, while the same configuration worked on RTX 3090 24GB. The error occur...
- [Bug 20251112 001 001 Dice Loss Assertion Error](BUG-20251112-001_001-dice-loss-assertion-error.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - pred = pred.clamp(0, 1) intersection = (pred * gt * mask).sum()
- [Bug 20251112 002 002 Mixed Precision Performance](BUG-20251112-002_002-mixed-precision-performance.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - val/hmean: 0.8839 val/hmean: 0.5530  # 37% performance drop
- [Bug 20251112 003 003 Run Id Confusion](BUG-20251112-003_003-run-id-confusion.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - Baseline run (9evam0xb): precision=32-true, hmean=0.8839 Optimized run (b1bipuoz): precision=16-mixed, hmean=0.7816
- [Bug 20251112 004 004 Caching Performance Impact](BUG-20251112-004_004-caching-performance-impact.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - val/hmean: 0.8839 val/hmean: 0.7816  # 11.6% performance drop
- [Bug 20251112 005 004 Streamlit Pandas Import Deadlock](BUG-20251112-005_004-streamlit-pandas-import-deadlock.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date Reported:** 2025-10-19 **Date Fixed:** 2025-10-20
- [Bug 20251112 006 004 Streamlit Viewer Hanging](BUG-20251112-006_004-streamlit-viewer-hanging.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date Reported**: 2025-10-18 **Status**: ✅ FIXED
- [Bug 20251112 007 005 Map Cache Invalidation](BUG-20251112-007_005-map-cache-invalidation.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Status**: IDENTIFIED **Severity**: Medium
- [Bug 20251112 008 005 Rbf Interpolation Hang](BUG-20251112-008_005-rbf-interpolation-hang.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date Reported**: 2025-10-18 **Severity**: 🔴 CRITICAL
- [Bug 20251112 009 001 Dice Loss Assertion Error](BUG-20251112-009_001-dice-loss-assertion-error.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - pred = pred.clamp(0, 1) intersection = (pred * gt * mask).sum()
- [Bug 20251112 010 Wandb Step Logging](BUG-20251112-010_wandb-step-logging.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - The PerformanceProfilerCallback was logging metrics to WandB with non-monotonic step values during testing phase, causing WandB to reject the logs with warnings: "Tried to log to step X that is less t...
- [Bug 20251112 011 Issues Resolution 2025 10 14](BUG-20251112-011_issues-resolution-2025-10-14.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date**: 2025-10-14 **Investigator**: Claude Code
- [Bug 20251112 012 Cache Analysis](BUG-20251112-012_cache-analysis.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date:** October 10, 2025 **Analyst:** GitHub Copilot
- [Bug 20251112 013 Cuda Illegal Memory Access In Bce Loss Computation](BUG-20251112-013_cuda-illegal-memory-access-in-bce-loss-computation.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - BUG-20251109-002 CUDA illegal memory access error occurs during training in BCE loss computation when computing `positive.sum().item()` at line 31 of `ocr/models/loss/bce_loss.py`. The error causes tr...
- [Bug 20251112 014 Cuda Cudnn Execution Error In Fpn Decoder During Training](BUG-20251112-014_cuda-cudnn-execution-error-in-fpn-decoder-during-training.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - model.encoder.model_name=resnet18 dataloaders.test_dataloader.num_workers=0
- [Bug 20251112 014 Cuda Illegal Instruction Error And Missing Directory In Exception Handler](BUG-20251112-014_cuda-illegal-instruction-error-and-missing-directory-in-exception-handler.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - BUG-20251112-014 Brief description of the bug.
- [Bug 20251112 015 Wandb Continues Running When Disabled In Configuration](BUG-20251112-015_wandb-continues-running-when-disabled-in-configuration.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - BUG-20251112-015 WandB continues to initialize and run even when explicitly disabled in configuration files. This causes WandB-related messages to appear in logs and may contribute to CUDA errors duri...
- [Bug 20251116 001 Annotation Fixes Applied](BUG-20251116-001_annotation_fixes_applied.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date**: 2025-11-16 **Script**: `scripts/data/fix_polygon_coordinates.py`
- [Bug 20251116 001 Candidate Files](BUG-20251116-001_CANDIDATE_FILES.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Why:** Core coordinate transformation logic - **Line 240:** `box = self.__transform_coordinates(box, inverse_matrix)` - Transforms polygons back to original coordinates
- [Bug 20251116 001 Debugging Handover](BUG-20251116-001_DEBUGGING_HANDOVER.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Bug ID:** BUG-20251116-001 **Date:** 2025-11-16
- [Bug 20251116 001 Excessive Invalid Polygons During Training](BUG-20251116-001_excessive-invalid-polygons-during-training.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - BUG-20251116-001 Training pipeline is dropping an extremely high number of invalid polygons due to out-of-bounds coordinate validation. Polygons are being rejected for:
- [Bug 20251116 001 Investigation Summary](BUG-20251116-001_investigation_summary.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Root Cause Identified**: `polygons_in_canonical_frame()` tolerance is too strict (1.5 pixels), causing polygons that are already in canonical frame to be incorrectly remapped, resulting in out-of-bo...
- [Bug 20251116 001 Quick Summary](BUG-20251116-001_QUICK_SUMMARY.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - 1. **Tolerance increase in `polygons_in_canonical_frame()`** - Changed from 1.5 to 3.0 pixels
- [Bug 20251116 001 Test Results Analysis](BUG-20251116-001_test_results_analysis.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - All tests pass, confirming that: 1. ✅ The tolerance fix is working correctly
- [Bug 20251116 001 Tolerance Explanation](BUG-20251116-001_tolerance_explanation.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Why do we need tolerance if images are expected to be the same when they are already in their canonical form? Don't rotated images need a tolerance factor?** Yes, tolerance is needed even for canoni...
- [Bug 20251116 001 Wandb Validation Images Show Black Background Due To Pil Opencv Color Format Mismatch](BUG-20251116-001_wandb-validation-images-show-black-background-due-to-pil-opencv-color-format-mismatch.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - BUG-20251116-001 WandB validation images display completely black backgrounds with green annotation overlays and almost no red prediction overlays. This indicates a critical bug in the train/validatio...
- [2025 004 Streamlit Pandas Import Deadlock](BUG_2025_004_STREAMLIT_PANDAS_IMPORT_DEADLOCK.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date Reported:** 2025-10-19 **Date Fixed:** 2025-10-20
- [2025 004 Streamlit Viewer Hanging](BUG_2025_004_STREAMLIT_VIEWER_HANGING.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date Reported**: 2025-10-18 **Status**: ✅ FIXED
- [2025 005 Rbf Interpolation Hang](BUG_2025_005_RBF_INTERPOLATION_HANG.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - **Date Reported**: 2025-10-18 **Severity**: 🔴 CRITICAL
- [Report Template](BUG_REPORT_TEMPLATE.md) (📅 2025-12-06 18:08 (KS, 🐛 bug_report) - class PerformanceProfilerCallback(Callback): def __init__(self, ...):
- [Inference preview misalignment for resized images](2025-12-03_0003_BUG_001_inference-resize-misalignment.md) (📅 2025-12-03 00:03 (KS, 🐛 bug_report) - BUG-001 <!-- REQUIRED: Fill these sections when creating the initial bug report -->
- [AgentQMS Bootstrap Test Bug](2025-12-02_2335_BUG_001_bootstrap-bug.md) (📅 2025-12-02 23:35 (KS, 🐛 bug_report) - BUG-001 <!-- REQUIRED: Fill these sections when creating the initial bug report -->
- [Inference Studio overlay misaligned with original image](2025-12-02_2313_BUG_001_inference-studio-overlay-misalignment.md) (📅 2025-12-02 23:13 (KS, 🐛 bug_report) - BUG-001 <!-- REQUIRED: Fill these sections when creating the initial bug report -->

## Summary

| Status | Count |
|--------|-------|
| Active | 55 |

---

*This index is automatically generated. Do not edit manually.*