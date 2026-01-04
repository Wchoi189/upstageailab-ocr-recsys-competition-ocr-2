# Walkthrough: BUG_003 Config Precedence Fix

**Date**: 2026-01-04
**Bug ID**: BUG_003
**Status**: ✅ RESOLVED

---

## Problem Summary

When running PARSeq training, the model incorrectly instantiated `FPNDecoder` instead of `PARSeqDecoder`, causing:
```text
ValueError: FPNDecoder requires at least two feature maps from the encoder.
```

## Root Cause Analysis

Used `adt trace-merges` to identify merge order issues in `OCRModel._prepare_component_configs`:

| Priority | Source | Problem |
|----------|--------|---------|
| P3 | `direct_overrides` | PARSeqDecoder (architecture) |
| P4 | `top_level_overrides` | FPNDecoder (legacy) **← Won!** |
| P5 | `cfg.component_overrides` | FPNDecoder (legacy) **← Also won!** |

Legacy configs from [_base/model.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/_base/model.yaml) → `model/architectures: dbnet` leaked through.

## Fix Implementation

### 1. Added [_filter_architecture_conflicts()](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/models/architecture.py#212-268) method
[architecture.py#L204-L259](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/models/architecture.py#L204-L259)

Filters legacy components when they conflict with architecture-defined components:
```python
def _filter_architecture_conflicts(self, top_level_overrides, architecture_overrides):
    # Compare component names, filter if different
    if arch_name and legacy_name and arch_name != legacy_name:
        logger.info(f"BUG_003: Filtering legacy {component} ({legacy_name}) "
                   f"in favor of architecture {component} ({arch_name})")
        del filtered[component]
```

### 2. Reordered merge operations
[architecture.py#L155-L175](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/models/architecture.py#L155-L175)

```diff
# Before (broken):
- merged_config = merge(direct_overrides)     # P3: Architecture
- merged_config = merge(top_level_overrides)  # P4: Legacy (wins!)
- merged_config = merge(cfg.component_overrides)  # P5: Legacy (wins!)

# After (fixed):
+ merged_config = merge(filtered_top_level)   # P3: Legacy (cleaned)
+ merged_config = merge(direct_overrides)     # P4: Architecture (wins!)
+ merged_config = merge(filtered_user_overrides)  # P5: User explicit (cleaned)
```

## Verification

### Debug Toolkit Trace (After Fix)
```bash
uv run adt trace-merges ocr/models/architecture.py
```
```
| Priority | Line | Operation | Winner on Conflict |
|----------|------|-----------|-------------------|
| P3 | 158 | merge | filtered_top_level |
| P4 | 162 | merge | direct_overrides |     ← Architecture wins!
| P5 | 175 | merge | filtered_user_overrides |
```

### Runtime Logs
```
INFO: BUG_003: Filtering legacy decoder (fpn_decoder) in favor of architecture decoder (parseq_decoder)
INFO: BUG_003: Filtering legacy head (db_head) in favor of architecture head (parseq_head)
INFO: BUG_003: Filtering legacy loss (db_loss) in favor of architecture loss (cross_entropy)
```

### Tests
- All 9 architecture tests pass ✅

---

## Remaining Issue (Separate from BUG_003)

PARSeq components are not registered in [ComponentRegistry](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/models/core/registry.py#11-286):
- `parseq_decoder` - Not found
- `parseq_head` - Not found
- `cross_entropy` loss - Not found

This is a component registration issue, not a config precedence issue.

---

## Files Modified

| File | Change |
|------|--------|
| [architecture.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/models/architecture.py) | Added [_filter_architecture_conflicts()](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/models/architecture.py#212-268), fixed merge order |
| [test_architecture.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/tests/unit/test_architecture.py) | Updated test expectations |
| [BUG_003 report](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/artifacts/bug_reports/2026-01-04_1730_BUG_003_config-precedence-leak.md) | Status → resolved |
