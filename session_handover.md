# Session Handover: KIE Training - NaN Loss Resolution

## Issue Resolved
**Root Cause**: The `baseline_train_optimized.parquet` dataset contains **100% empty text strings** in the `texts` column.

When `LayoutLMv3Processor` processes empty strings:
1. No tokens are generated for those words
2. No word-level labels can be aligned
3. All labels become `-100` (ignore index for CrossEntropyLoss)
4. CrossEntropyLoss with all-ignored targets → NaN

## Data Status
| Dataset | Texts | Labels | Status |
|---------|-------|--------|--------|
| `aihub_validation_optimized.parquet` | ✅ Valid (0% empty) | ✅ "text" | Ready for training |
| `baseline_train_optimized.parquet` | ❌ 100% empty | ✅ "text" | **BROKEN** |

## Solution
Created `configs/train_kie_aihub_only.yaml` which uses only AI Hub data:
- Dataset: 5,467 samples with valid word-level text and polygons
- Verified: Forward pass produces loss = 0.656 (NOT NaN)

## Run Command
```bash
# Fixed training command
python runners/train_kie.py --config-name train_kie_aihub_only
```

## Longer-Term Fix Needed
The `baseline_train` pseudo-labeling pipeline needs to be fixed to properly extract text content from the Upstage Document Parse API response. The current pipeline only captured:
- Polygons ✅
- Label type ("text") ✅
- Actual text content ❌ (stored as empty strings)

## Files Modified
- **Created**: `configs/train_kie_aihub_only.yaml` - AI Hub-only training config
