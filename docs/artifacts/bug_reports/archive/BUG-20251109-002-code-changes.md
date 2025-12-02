---
title: "Code Changes for BUG-20251109-002"
author: "ai-agent"
date: "2025-11-09"
timestamp: "2025-11-09 23:04 KST"
type: "bug_fix"
category: "troubleshooting"
status: "in_progress"
version: "1.0"
tags: ['bug', 'cuda', 'loss', 'bce', 'code-changes']
bug_id: "BUG-20251109-002"
severity: "High"
---

# Code Changes for BUG-20251109-002: CUDA Illegal Memory Access in BCE Loss Computation

## Overview
This document tracks all code changes made to fix BUG-20251109-002. All changes are indexed with the bug ID for proper tracking and version control.

## Bug Report
- **Bug ID:** BUG-20251109-002
- **Bug Report:** [docs/bug_reports/BUG-20251109-002-cuda-illegal-memory-access-in-bce-loss-computation.md](../bug_reports/BUG-20251109-002-cuda-illegal-memory-access-in-bce-loss-computation.md)
- **Severity:** High
- **Status:** In Progress (fix applied but error persists - suggests corruption happens earlier)

## Functions Changed (Indexed by Bug ID)

### 1. `BCELoss.forward()` - `ocr/models/loss/bce_loss.py`
**Bug ID:** BUG-20251109-002
**Function:** `BCELoss.forward(self, pred_logits, gt, mask=None)`
**Change Type:** Bug Fix (Partial - error persists)
**Date:** 2025-11-09

#### Changes Made:
1. **Added Input Validation (Lines 40-54)**
   - Added shape validation for `pred_logits`, `gt`, and `mask`
   - Added device validation to ensure all tensors are on the same device
   - **Purpose:** Prevent CUDA illegal memory access from shape/device mismatches
   - **Bug ID Reference:** `# BUG-20251109-002: Validate inputs to prevent CUDA illegal memory access`

2. **Added CUDA Synchronization (Lines 56-64)**
   - Added CUDA synchronization before operations
   - Added error handling for CUDA synchronization failures
   - **Purpose:** Clear any previous CUDA errors before operations
   - **Bug ID Reference:** `# BUG-20251109-002: Check for CUDA errors before operations`

3. **Moved Operations to CPU (Lines 66-90)**
   - Changed to move `gt` and `mask` to CPU first, then do all operations on CPU
   - Changed boolean mask creation to happen on CPU
   - Changed sum operations to happen on CPU
   - **Purpose:** Avoid CUDA operations on potentially corrupted memory
   - **Issue:** Even `.cpu()` fails if CUDA memory is corrupted - suggests corruption happens earlier
   - **Bug ID Reference:** `# BUG-20251109-002: Create boolean masks with error handling`

4. **Enhanced Error Handling (Lines 92-110)**
   - Added detailed error context including shapes and devices
   - Added debugging suggestions (clear cache, reduce batch size, check earlier pipeline)
   - **Purpose:** Provide better debugging information when CUDA errors occur

#### Code References:
- **Before:** Line 31: `positive_count = int(positive.sum().item())`
- **After:** Lines 72-81: Move to CPU first, then sum on CPU

#### Function Docstring:
```python
def forward(self, pred_logits, gt, mask=None):
    """
    Forward pass for BCE loss computation.

    BUG-20251109-002: Fixed CUDA illegal memory access by:
    - Adding input validation (shape/device checks)
    - Adding CUDA synchronization before operations
    - Moving operations to CPU to avoid corrupted memory access
    - Enhanced error handling with debugging context

    See: docs/bug_reports/BUG-20251109-002-code-changes.md
    """
```

#### Status:
- ✅ Changes applied
- ⚠️ Error persists - even `.cpu()` fails, suggesting CUDA memory corruption happens earlier in pipeline
- 🔍 Next step: Investigate data pipeline (collate function, dataset creation) and model forward pass

## Related Changes (wandb import fix)

### 2. `_safe_wandb_finish()` - `runners/train.py`
**Bug ID:** BUG-20251109-002 (Related - wandb import fix)
**Function:** `_safe_wandb_finish()`
**Change Type:** Bug Fix
**Date:** 2025-11-09

#### Changes Made:
- Removed top-level `wandb` import
- Added lazy import helper `_safe_wandb_finish()`
- **Purpose:** Fix wandb import hanging during module import (separate issue, but related to debugging)
- **Bug ID Reference:** Function docstring includes `BUG-20251109-002`

### 3. `_get_wandb()` - `ocr/utils/wandb_utils.py`
**Bug ID:** BUG-20251109-002 (Related - wandb import fix)
**Function:** `_get_wandb()`
**Change Type:** Bug Fix
**Date:** 2025-11-09

#### Changes Made:
- Made wandb import lazy via `_get_wandb()` helper function
- **Purpose:** Fix wandb import hanging during module import
- **Bug ID Reference:** Function docstring includes `BUG-20251109-002`

### 4. `UniqueModelCheckpoint._generate_checkpoint_metadata()` - `ocr/lightning_modules/callbacks/unique_checkpoint.py`
**Bug ID:** BUG-20251109-002 (Related - wandb import fix)
**Function:** `UniqueModelCheckpoint._generate_checkpoint_metadata(self, checkpoint_path: str, metrics: dict)`
**Change Type:** Bug Fix
**Date:** 2025-11-09

#### Changes Made:
- Changed from top-level `import wandb` to lazy import via `_get_wandb()` helper
- **Purpose:** Fix wandb import hanging during module import
- **Bug ID Reference:** Uses `_get_wandb()` helper which is indexed with bug ID

### 5. `WandbCompletionCallback.on_train_end()` - `ocr/lightning_modules/callbacks/wandb_completion.py`
**Bug ID:** BUG-20251109-002 (Related - wandb import fix)
**Function:** `WandbCompletionCallback.on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule)`
**Change Type:** Bug Fix
**Date:** 2025-11-09

#### Changes Made:
- Changed from top-level `import wandb` to lazy import via `_get_wandb()` helper
- **Purpose:** Fix wandb import hanging during module import
- **Bug ID Reference:** Uses `_get_wandb()` helper which is indexed with bug ID

### 6. `WandbCompletionCallback.on_exception()` - `ocr/lightning_modules/callbacks/wandb_completion.py`
**Bug ID:** BUG-20251109-002 (Related - wandb import fix)
**Function:** `WandbCompletionCallback.on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException)`
**Change Type:** Bug Fix
**Date:** 2025-11-09

#### Changes Made:
- Changed from top-level `import wandb` to lazy import via `_get_wandb()` helper
- **Purpose:** Fix wandb import hanging during module import
- **Bug ID Reference:** Uses `_get_wandb()` helper which is indexed with bug ID

## Testing Status
- [x] Code changes applied
- [x] Error persists - even `.cpu()` fails
- [ ] Need to investigate earlier in pipeline
- [ ] Unit tests updated/added
- [ ] Integration/E2E validated
- [ ] Training run verified without CUDA errors

## Next Steps
1. Investigate data pipeline (collate function, dataset creation)
2. Check model forward pass for out-of-bounds tensor access
3. Verify tensor creation in data loading
4. Check for race conditions in multi-threaded data loading
5. Use `CUDA_LAUNCH_BLOCKING=1` to identify exact operation causing corruption

## Notes
- All changes are indexed with bug ID BUG-20251109-002
- The fix in `bce_loss.py` is a workaround - the root cause is likely earlier in the pipeline
- Even moving to CPU fails if CUDA memory is corrupted, suggesting corruption happens during tensor creation or earlier operations
