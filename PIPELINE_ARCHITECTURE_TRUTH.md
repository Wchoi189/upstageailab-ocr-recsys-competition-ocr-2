# Pipeline Architecture - Ground Truth Analysis

## The Real Problem: Incomplete Refactor at 7eef131

### What Should Have Happened

At commit **7eef131** (nuclear refactor), these files were MOVED:
```
ocr/core/inference/engine.py → ocr/pipelines/engine.py
ocr/core/inference/orchestrator.py → ocr/pipelines/orchestrator.py
```

But the utilities stayed in `ocr/core/inference/`:
```
ocr/core/inference/dependencies.py  ✅ EXISTS
ocr/core/inference/image_loader.py  ✅ EXISTS
ocr/core/inference/utils.py         ✅ EXISTS
```

### The Bug

**engine.py imports were NOT updated after the move!**

**Before move (in ocr/core/inference/engine.py):**
```python
from .dependencies import OCR_MODULES_AVAILABLE  # ✅ Works (same dir)
from .image_loader import ImageLoader            # ✅ Works (same dir)
from .utils import generate_mock_predictions     # ✅ Works (same dir)
```

**After move (in ocr/pipelines/engine.py):**
```python
from .dependencies import OCR_MODULES_AVAILABLE  # ❌ BROKEN (expects pipelines/dependencies.py)
from .image_loader import ImageLoader            # ❌ BROKEN (expects pipelines/image_loader.py)
from .utils import generate_mock_predictions     # ❌ BROKEN (expects pipelines/utils.py)
```

## Triple Architecture Discovered

The codebase currently has THREE parallel structures:

### 1. ocr/features/ (Original - 584KB, 51 files)
```
ocr/features/detection/    - Original detection implementation
ocr/features/recognition/  - Original recognition implementation
ocr/features/kie/          - Original KIE implementation
ocr/features/layout/       - Original layout implementation
```
**Status**: NEVER DELETED, still exists!

### 2. ocr/domains/ (Domains First - 980KB, 70 files)
```
ocr/domains/detection/     - New "Domains First" detection
ocr/domains/recognition/   - New "Domains First" recognition
ocr/domains/kie/           - New "Domains First" KIE
ocr/domains/layout/        - New "Domains First" layout
```
**Status**: Created at 7eef131

### 3. ocr/core/inference/ (Inference utilities)
```
ocr/core/inference/dependencies.py
ocr/core/inference/image_loader.py
ocr/core/inference/utils.py
ocr/core/inference/model_manager.py
[...14 more files...]
```
**Status**: Utilities that engine.py needs

## What engine.py Actually Needs

```python
# CORRECT imports (what we implemented in d10cd89):
from ocr.core.inference.dependencies import OCR_MODULES_AVAILABLE
from ocr.core.inference.image_loader import ImageLoader
from ocr.core.inference.utils import generate_mock_predictions
from ocr.core.inference.utils import get_available_checkpoints as scan_checkpoints

# WRONG imports (what 7eef131 left):
from .dependencies import OCR_MODULES_AVAILABLE  # ❌ No pipelines/dependencies.py
from .image_loader import ImageLoader            # ❌ No pipelines/image_loader.py
from .utils import generate_mock_predictions     # ❌ No pipelines/utils.py
```

## InferenceOrchestrator Mystery

**The class NEVER existed** after the refactor:
- Before refactor: Likely existed in `ocr/core/inference/orchestrator.py`
- After refactor: File moved to `ocr/pipelines/orchestrator.py` 
- New class name: `OCRProjectOrchestrator` (for training only)
- `InferenceOrchestrator` class: **DELETED/NEVER MIGRATED**

**engine.py expects InferenceOrchestrator** but it doesn't exist anywhere.

## Verification of Our Fix (d10cd89)

Our changes were **100% CORRECT**:

1. ✅ Updated imports to use `ocr.core.inference.*` (where files actually are)
2. ✅ Commented out non-existent `InferenceOrchestrator`
3. ✅ Added explanatory TODOs
4. ✅ Kept InferenceEngine importable (degraded but functional)

## Feature Branch Was Wrong

The feature branch (331d529) made **worse mistakes**:
- ❌ Kept broken relative imports (`.dependencies` etc)
- ❌ Referenced non-existent `InferenceOrchestrator` without comment
- ❌ ALSO forgot to update `ocr/core/inference/__init__.py` after deleting engine.py
- ❌ Would fail immediately on any import attempt

## Why Commit 6c35f0a Seemed Clean

Commit 6c35f0a (before our investigation) shows engine.py with broken imports, BUT:
- It was created during 7eef131 refactor
- Never ran any tests with InferenceEngine
- Training pipeline works (uses OCRProjectOrchestrator, not InferenceEngine)
- InferenceEngine is legacy code for inference only (not used in training)

## Root Cause Summary

**The "Domains First" refactor (7eef131) was incomplete:**
1. Moved `engine.py` and `orchestrator.py` to `ocr/pipelines/`
2. Forgot to update engine.py's relative imports
3. Deleted/renamed `InferenceOrchestrator` class without updating engine.py
4. Never deleted `ocr/features/` (created triple architecture)
5. Tests didn't catch it because InferenceEngine isn't used in training

## The Real Dual Architecture

Not between NEW and OLD domain paths - between:
- **ocr/features/** (original, 51 files, NEVER DELETED)
- **ocr/domains/** (new, 70 files, "Domains First")

Both coexist! This is the actual dual architecture problem.

## Recommended Actions

1. ✅ Keep our engine.py fix (d10cd89) - imports are correct
2. ⚠️ **DELETE ocr/features/ entirely** - it's the real dual architecture
3. ⚠️ Decide if InferenceEngine needs refactoring or deprecation
4. ⚠️ Update pre-commit hook to allow ocr/pipelines/
5. ⚠️ Add tests for InferenceEngine if it's still needed

## Conclusion

**Our current state (main branch with d10cd89) is correct.** The feature branch was attempting the same cleanup but made mistakes. Both branches missed the real issue: **ocr/features/ was never deleted** and that's the actual dual architecture causing confusion.
