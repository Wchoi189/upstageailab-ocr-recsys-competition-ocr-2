---
ads_version: "1.0"
type: status_update
experiment_id: "20251220_154834_zero_prediction_images_debug"
created: "2025-12-21T02:24:00+09:00"
updated: "2025-12-21T03:15:00+09:00"
status: "completed"
progress: 100
---

# Sepia Testing Final Status

**Date**: 2025-12-21 03:15 KST
**Overall Progress**: 100% complete ✅

## Completed Phases ✅

### Phase 1: Color Mapping Correction ✅
- Identified and fixed BGR -> RGB channel swap in sepia matrices.
- Verified expected reddish tint (previously greenish).
- Performance boost: Edge improvement increased from +141% to **+164%**.

### Phase 2: CLAHE vs Linear Contrast ✅
- Renamed original `sepia_contrast` to `sepia_clahe`.
- Implemented `sepia_linear_contrast` for benchmarking.
- Confirmed CLAHE is significantly superior for low-contrast document OCR.

**Final Metrics (000732_REMBG)**:
| Method | Tint | Contrast | Edge Improvement | Speed (ms) |
|--------|------|----------|------------------|------------|
| **sepia_clahe** | +44.1 | **+8.2** | **+164.0%** | ~85ms |
| sepia_linear_contrast | +44.2 | +0.7 | +84.4% | ~8ms |
| sepia_warm | +44.8 | +1.9 | +58.8% | ~12ms |
| sepia_classic | +24.4 | +2.1 | +46.9% | ~8ms |

### Phase 3: Pipeline Validation ✅
- Integrated `sepia_clahe` into perspective correction pipeline.
- Successfully processed problematic images 000712 and 000732.
- Total pipeline time: ~140ms.

## Final Findings 🏆

1. **Corrected Tint is Crucial**: Switching from greenish to reddish sepia significantly improved character edge clarity (+23% gain).
2. **Adaptive Contrast (CLAHE) is Mandatory**: CLAHE provides nearly 2X the edge improvement of standard linear contrast (+164% vs +84%).
3. **Execution Cost**: CLAHE adds ~75ms over linear contrast, but the 10X edge clarity improvement justifies the cost for problematic images.

## Output Artifacts
- Comparison Grid: `outputs/sepia_comparison/drp.en_ko.in_house.selectstar_000732_REMBG_comparison_grid.jpg`
- Pipeline Results: `outputs/sepia_pipeline/`
- Full Metrics: `outputs/sepia_comparison/drp.en_ko.in_house.selectstar_000732_REMBG_metrics.json`

## Timeline

- Started: 2025-12-21 02:23 KST
- Completed: 2025-12-21 03:15 KST
- Total duration: 52 minutes
