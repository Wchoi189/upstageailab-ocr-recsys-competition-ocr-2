# Quick Start: Debugging Unified OCR App

**For the next developer** - Start here! 👇

---

## TL;DR - What's Broken

🔴 **App starts but UI never loads** - perpetual loading spinner in browser

**Root Cause**: Heavy resources (likely ML models) being loaded during service initialization

**Status**:
- ✅ Lazy imports fixed (moved to module level)
- 🔴 Heavy resource loading not fixed yet
- 🔴 Architecture needs refactoring

---

## Quick Diagnosis Steps

### Step 1: Check if services are loading heavy resources

```bash
# Search for model loading in services
grep -r "torch.load\|load_model\|\.load(" ui/apps/unified_ocr_app/services/

# Check __init__ methods
grep -A 10 "def __init__" ui/apps/unified_ocr_app/services/*.py
```

### Step 2: Add debug to find the blocker

Add to **each service file** at the very top:
```python
print(f"DEBUG: Loading module {__name__}", flush=True)
```

Add to each `__init__` method:
```python
def __init__(self, config):
    print(f"DEBUG: Initializing {self.__class__.__name__}", flush=True)
    # ... rest of code
```

### Step 3: Run and check where it hangs

```bash
# Run streamlit
uv run streamlit run ui/apps/unified_ocr_app/app.py --server.port 8501

# Open in browser: http://localhost:8501
# Check terminal output - last DEBUG line shows where it hung
```

---

## Quick Fix Pattern

### If you find model loading in `__init__`:

**❌ BEFORE (blocks app)**:
```python
class InferenceService:
    def __init__(self, config):
        self.model = torch.load("model.pth")  # BLOCKS HERE!
```

**✅ AFTER (cached, lazy)**:
```python
import streamlit as st

@st.cache_resource
def _load_model(model_path: str):
    """Load model once, cache forever."""
    print(f"Loading model from {model_path}...")
    return torch.load(model_path)

class InferenceService:
    def __init__(self, config):
        model_path = config.get("model_path")
        self.model = _load_model(model_path)  # Fast after first call!
```

### Key Points:
1. Use `@st.cache_resource` for ML models
2. Use `@st.cache_data` for data/results
3. Never load heavy stuff at module level or in `__init__` without caching

---

## Files to Check (Priority Order)

1. **`ui/apps/unified_ocr_app/services/inference_service.py`** (🔴 HIGH)
   - Most likely culprit
   - Check `__init__` and `load_checkpoints()`

2. **`ui/apps/unified_ocr_app/services/preprocessing_service.py`** (🟡 MEDIUM)
   - May load heavy preprocessing pipelines

3. **`ui/apps/unified_ocr_app/services/comparison_service.py`** (🟢 LOW)
   - Calls other services, probably not direct cause

---

## Testing the Fix

### Step 1: Verify imports work
```bash
python3 -c "import ui.apps.unified_ocr_app.app; print('Success!')"
```

### Step 2: Start app
```bash
uv run streamlit run ui/apps/unified_ocr_app/app.py
```

### Step 3: Open browser and check:
- Does UI load within 5 seconds? ✅
- Can you switch between modes? ✅
- Does inference mode work? ✅

---

## Known Issues

### Issue #1: Debug Code Still Present

**File**: `ui/apps/unified_ocr_app/app.py`

**Lines to remove**:
- 15-25: First debug block
- 31-41: Debug file path
- 45-47, 56-58, 62-64, 70-73: Debug writes
- 85-87: More debug writes
- 105-177: All debug code in main()

**Quick cleanup**:
```bash
# Backup first
cp ui/apps/unified_ocr_app/app.py ui/apps/unified_ocr_app/app.py.backup

# Then manually remove debug code or use the clean version in the refactor plan
```

### Issue #2: Monolithic Architecture

**Problem**: 725-line app.py file handles all 3 modes

**Solution**: See [docs/ai_handbook/08_planning/APP_REFACTOR_PLAN.md](docs/ai_handbook/08_planning/APP_REFACTOR_PLAN.md)

**When to refactor**: After fixing the loading issue!

---

## Documentation

### Main Documents (in order to read them):

1. **[SESSION_HANDOVER_APP_REFACTOR.md](SESSION_HANDOVER_APP_REFACTOR.md)** ← START HERE
   - Full context of debugging session
   - What was fixed, what wasn't
   - Next steps

2. **[docs/ai_handbook/08_planning/APP_REFACTOR_PLAN.md](docs/ai_handbook/08_planning/APP_REFACTOR_PLAN.md)**
   - Detailed refactoring plan
   - Multi-page architecture design
   - Implementation steps

3. **[SESSION_COMPLETE_2025-10-21_PHASE7.md](SESSION_COMPLETE_2025-10-21_PHASE7.md)**
   - Phase 7 completion summary
   - Overall project status

### Debug Artifacts:

- **Debug log**: `/tmp/streamlit_debug.log` (if it exists)
- **Streamlit logs**: `logs/ui/unified_app_8501.err`

---

## Quick Commands

```bash
# Start app
uv run streamlit run ui/apps/unified_ocr_app/app.py

# Start on different port
uv run streamlit run ui/apps/unified_ocr_app/app.py --server.port 8502

# Check debug log
cat /tmp/streamlit_debug.log

# Test imports
python3 -c "import ui.apps.unified_ocr_app.app"

# Search for model loading
grep -r "torch.load\|load_model" ui/apps/unified_ocr_app/services/

# Kill all streamlit processes
pkill -9 streamlit
```

---

## Success Criteria

Before moving to refactoring, ensure:

- [ ] App UI loads in browser (< 5 seconds)
- [ ] All 3 modes are accessible
- [ ] No perpetual loading spinner
- [ ] Services use `@st.cache_resource` for heavy operations
- [ ] Debug code removed from app.py

---

## Need Help?

1. Check [SESSION_HANDOVER_APP_REFACTOR.md](SESSION_HANDOVER_APP_REFACTOR.md) for full context
2. Check `/tmp/streamlit_debug.log` to see where execution stopped
3. Add more debug prints to narrow down the issue
4. Search the services for blocking operations

**Key Insight**: The app imports work fine (proven by `python -c "import ..."`), so the issue is something that happens when Streamlit tries to render the UI, likely during service initialization.

---

**Last Updated**: 2025-10-21
**Status**: 🔴 Requires immediate attention
**Priority**: Fix heavy loading, then refactor
