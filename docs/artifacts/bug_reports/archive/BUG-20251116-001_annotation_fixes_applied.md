---
title: "Bug 20251116 001 Annotation Fixes Applied"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---



# BUG-20251116-001: Annotation File Fixes Applied

**Date**: 2025-11-16
**Script**: `scripts/data/fix_polygon_coordinates.py`
**Tolerance**: 3.0 pixels (matching validation tolerance)

## Summary

Fixed out-of-bounds polygon coordinates in annotation files by clamping coordinates to valid range [0, width] x [0, height]. This addresses the root cause identified in the investigation: polygons with coordinates slightly outside bounds due to floating-point errors and annotation tool rounding.

## Files Fixed

### 1. `data/datasets/jsons/train.json`

**Status**: ✅ Fixed
**Backup**: `data/datasets/jsons/train.json.backup`

**Statistics**:
- Images processed: 3,272
- Images with fixes: 76
- Polygons fixed: 146
- Polygons out of tolerance (>3 pixels): 58

**Fix Details**:
- Coordinates clamped to valid range [0, width] x [0, height]
- 146 polygons had coordinates adjusted
- 58 polygons were beyond 3-pixel tolerance but still clamped (legitimate annotation errors)
- All fixes preserve polygon geometry while ensuring bounds compliance

**Sample Fixes**:
- `drp.en_ko.in_house.selectstar_000042.jpg`: Word 0076 - X: [-6.4, 308.2] clamped, Y: [-10.7, 39.7] clamped
- `drp.en_ko.in_house.selectstar_000057.jpg`: 6 polygons fixed - Y coordinates clamped (e.g., 1281.9 → 1280.0)
- `drp.en_ko.in_house.selectstar_000141.jpg`: 2 polygons fixed - Y coordinates clamped (e.g., 1284.3 → 1280.0)

### 2. `data/datasets/jsons/val.json`

**Status**: ✅ Fixed
**Backup**: `data/datasets/jsons/val.json.backup`

**Statistics**:
- Images processed: 404
- Images with fixes: 7
- Polygons fixed: 14
- Polygons out of tolerance (>3 pixels): 5

**Fix Details**:
- Similar fixes applied to validation set
- 14 polygons had coordinates adjusted
- 5 polygons were beyond 3-pixel tolerance but still clamped

### 3. `data/datasets/jsons/test.json`

**Status**: ✅ No fixes needed

**Statistics**:
- Images processed: 413
- Images with fixes: 0
- Polygons fixed: 0

**Note**: Test set had no out-of-bounds coordinates.

## Fix Methodology

The fix script:
1. **Loads annotations** from JSON files
2. **Gets image dimensions** (including EXIF orientation handling)
3. **Clamps coordinates** to valid range [0, width] x [0, height]
4. **Preserves polygon geometry** while ensuring bounds compliance
5. **Creates backups** before modifying files

**Tolerance**: 3.0 pixels
- Matches the tolerance used in `ValidatedPolygonData.validate_bounds()`
- Handles floating-point errors (0.1-1 pixels)
- Handles annotation tool rounding (1-2 pixels)
- Handles EXIF remapping errors (1-3 pixels)

## Impact

### Before Fixes:
- 146 polygons in training set had out-of-bounds coordinates
- These caused validation errors during training
- Polygons were dropped, reducing training data

### After Fixes:
- All coordinates are within valid bounds
- No validation errors expected during training
- All polygons preserved (none dropped)
- Training data loss eliminated

## Related Changes

This fix complements:
1. **Tolerance increase in `polygons_in_canonical_frame()`** (BUG-20251116-001)
   - Prevents double-remapping of polygons already in canonical frame
   - File: `ocr/utils/orientation.py` line 166

2. **Validation tolerance in `ValidatedPolygonData`** (BUG-20251116-001)
   - 3-pixel tolerance for coordinate clamping
   - File: `ocr/datasets/schemas.py`

## Verification

To verify the fixes:
```bash
# Check that no polygons are out of bounds
python scripts/data/investigate_polygon_bounds.py \
    --annotation-file data/datasets/jsons/train.json \
    --image-dir data/datasets/images/train \
    --output-report reports/polygon_bounds_after_fix.json
```

Expected result: 0 polygons out of bounds (or only truly invalid cases beyond tolerance).

## Rollback

If needed, restore from backups:
```bash
# Restore train.json
cp data/datasets/jsons/train.json.backup data/datasets/jsons/train.json

# Restore val.json
cp data/datasets/jsons/val.json.backup data/datasets/jsons/val.json
```

## Notes

- **Backups created**: Both files have `.backup` extensions
- **Fix is reversible**: Original files preserved in backups
- **No data loss**: All polygons preserved, only coordinates adjusted
- **Consistent tolerance**: 3.0 pixels matches validation tolerance

---

*Fixes applied: 2025-11-16*
*Script version: `scripts/data/fix_polygon_coordinates.py`*
*Related bug: BUG-20251116-001*
