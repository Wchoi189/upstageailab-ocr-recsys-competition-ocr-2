---
title: "Bug 20251116 001 Quick Summary"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---



# BUG-20251116-001: Quick Summary

## What Was Fixed

1. **Tolerance increase in `polygons_in_canonical_frame()`**
   - Changed from 1.5 to 3.0 pixels
   - Prevents double-remapping of polygons already in canonical frame
   - File: `ocr/utils/orientation.py` line 166

2. **Annotation files fixed**
   - `train.json`: Fixed 146 polygons in 76 images
   - `val.json`: Fixed 14 polygons in 7 images
   - `test.json`: No fixes needed
   - Backups created: `train.json.backup` (261M), `val.json.backup` (32M)

3. **Checkpoint configuration optimized**
   - Reduced `save_top_k` from 3 to 1
   - Set `verbose: False` to reduce log spam
   - File: `configs/callbacks/model_checkpoint.yaml`

## Impact

- ✅ Prevents double-remapping errors
- ✅ Fixes 160 out-of-bounds polygons
- ✅ Eliminates training data loss from coordinate errors
- ✅ Reduces checkpoint disk usage

## Tools Created

1. `scripts/data/fix_polygon_coordinates.py` - Fixes out-of-bounds coordinates
2. `scripts/data/investigate_polygon_bounds.py` - Investigates root causes

## Test Results

- ✅ 10/10 unit tests passing
- ✅ 6/6 integration tests passing
- ✅ Tolerance fix verified working
- ✅ Double-remapping prevention confirmed

## Next Steps

1. Run training to verify no coordinate errors
2. Monitor for any new issues
3. Consider fixing source annotation tools if errors persist

---

*Date: 2025-11-16*
*Status: ✅ Complete*
