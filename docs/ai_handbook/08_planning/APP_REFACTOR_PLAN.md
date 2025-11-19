# Unified OCR App Refactoring Plan

**Status**: ✅ COMPLETED (2025-10-21)
**Priority**: 🔴 HIGH
**Actual Effort**: 1 session (Big Bang migration)
**Dependencies**: Heavy resource loading already optimized (services use lazy loading & caching)

---

## Table of Contents
1. [Current Problems](#current-problems)
2. [Proposed Solution](#proposed-solution)
3. [Implementation Plan](#implementation-plan)
4. [Migration Strategy](#migration-strategy)
5. [Testing Plan](#testing-plan)
6. [Rollback Plan](#rollback-plan)

---

## Current Problems

### Problem 1: Monolithic Architecture

**File**: [ui/apps/unified_ocr_app/app.py](../../../ui/apps/unified_ocr_app/app.py)
**Lines of Code**: 725 lines
**Functions**: 6 major functions handling 3 different modes

```python
# Current structure
app.py (725 lines)
├── main()                              # 100 lines
├── render_preprocessing_mode()         # 90 lines
├── render_inference_mode()             # 60 lines
├── _render_single_image_inference()    # 80 lines
├── _render_batch_inference()           # 90 lines
└── render_comparison_mode()            # 135 lines
```

**Issues**:
- ❌ All 3 modes loaded even when only 1 is used
- ❌ All imports active regardless of selected mode
- ❌ Hard to maintain (3 unrelated features in 1 file)
- ❌ Merge conflicts when multiple developers work on different modes
- ❌ Difficult to debug (complex control flow)
- ❌ Poor code locality (related code scattered across file)

### Problem 2: Heavy Resource Loading

**Suspected Locations**:
```python
# app.py lines 72-84: All services imported at module level
from ui.apps.unified_ocr_app.services.preprocessing_service import PreprocessingService
from ui.apps.unified_ocr_app.services.inference_service import InferenceService, load_checkpoints
from ui.apps.unified_ocr_app.services.comparison_service import get_comparison_service
```

**Issues**:
- ❌ Services likely load models/checkpoints during `__init__`
- ❌ No `@st.cache_resource` usage
- ❌ Resources loaded even if mode never accessed
- ❌ Blocks entire app startup

### Problem 3: Tight Coupling

```python
# app.py imports EVERYTHING for ALL modes
from ui.apps.unified_ocr_app.components.preprocessing import render_parameter_panel, render_stage_viewer
from ui.apps.unified_ocr_app.components.preprocessing.parameter_panel import render_preset_management
from ui.apps.unified_ocr_app.components.shared import render_image_upload
from ui.apps.unified_ocr_app.services.preprocessing_service import PreprocessingService

from ui.apps.unified_ocr_app.components.inference import render_checkpoint_selector, render_results_viewer
from ui.apps.unified_ocr_app.components.inference.checkpoint_selector import render_hyperparameters, render_mode_selector
from ui.apps.unified_ocr_app.services.inference_service import InferenceService, load_checkpoints

from ui.apps.unified_ocr_app.components.comparison import (
    render_metrics_display,
    render_parameter_sweep,
    render_results_comparison,
)
from ui.apps.unified_ocr_app.components.comparison.results_comparison import render_export_controls
from ui.apps.unified_ocr_app.components.comparison.metrics_display import render_analysis_summary
from ui.apps.unified_ocr_app.services.comparison_service import get_comparison_service
```

**Issues**:
- ❌ 15+ imports loaded regardless of which mode is active
- ❌ Changes to inference affect preprocessing file
- ❌ Cannot deploy modes independently

---

## Proposed Solution

### Solution: Streamlit Multi-Page App

Convert single monolithic `app.py` into multiple page files using Streamlit's built-in multi-page feature.

**Architecture**:
```
ui/apps/unified_ocr_app/
├── app.py                           # Home page (50 lines)
│   └── Config loading, welcome screen
├── pages/                           # Auto-discovered by Streamlit
│   ├── 1_🎨_Preprocessing.py       # Preprocessing mode (200 lines)
│   ├── 2_🤖_Inference.py           # Inference mode (250 lines)
│   └── 3_📊_Comparison.py          # Comparison mode (225 lines)
├── components/                      # Unchanged
│   ├── preprocessing/
│   ├── inference/
│   └── comparison/
├── services/                        # Updated with caching
│   ├── preprocessing_service.py
│   ├── inference_service.py
│   └── comparison_service.py
└── models/                          # Unchanged
    └── app_state.py
```

### Benefits

#### ✅ Performance
- **Lazy Loading**: Only active page's imports are loaded
- **Faster Startup**: Home page loads in <1s
- **Reduced Memory**: Only active mode's resources in RAM
- **Cached Resources**: `@st.cache_resource` per-page

#### ✅ Maintainability
- **Separation of Concerns**: Each mode in own file
- **Code Locality**: Related code together
- **Smaller Files**: ~200 lines each vs 725 lines
- **Clear Boundaries**: No cross-mode dependencies

#### ✅ Developer Experience
- **Parallel Development**: Team can work on different pages simultaneously
- **Easier Debugging**: Smaller, focused files
- **Clearer Testing**: Test pages independently
- **No Merge Conflicts**: Different files for different modes

#### ✅ User Experience
- **Automatic Navigation**: Streamlit builds sidebar menu
- **URL Support**: Each page has unique URL
- **Bookmarkable**: Users can bookmark specific modes
- **Clean UI**: Professional multi-page navigation

---

## Implementation Plan

### Phase 0: Prerequisites (MUST DO FIRST)

#### 0.1: Fix Heavy Resource Loading

**Priority**: 🔴 CRITICAL - Must fix before refactoring

**Action Items**:
1. Investigate service initialization
2. Add `@st.cache_resource` to model loading
3. Ensure services don't load resources at import time
4. Test app loads successfully

**Files to Fix**:
- `ui/apps/unified_ocr_app/services/inference_service.py`
- `ui/apps/unified_ocr_app/services/preprocessing_service.py`
- `ui/apps/unified_ocr_app/services/comparison_service.py`

**Success Criteria**:
- ✅ App UI loads in browser
- ✅ No perpetual loading spinner
- ✅ Startup time < 5 seconds

#### 0.2: Clean Up Debug Code

**Files**:
- `ui/apps/unified_ocr_app/app.py` (remove debug writes, lines 15-180)

**What to Remove**:
- All `/tmp/streamlit_debug.log` writes
- All `print(..., file=sys.stderr)` statements
- Keep only proper logging with `logger.info()`

---

### Phase 1: Create Multi-Page Structure

#### 1.1: Create Pages Directory

```bash
cd ui/apps/unified_ocr_app
mkdir -p pages
```

#### 1.2: Create Shared Utilities Module

**File**: `ui/apps/unified_ocr_app/shared_utils.py`

```python
"""Shared utilities for all pages."""

import streamlit as st
from typing import Dict, Any
from ui.apps.unified_ocr_app.services.config_loader import load_unified_config
from ui.apps.unified_ocr_app.models.app_state import UnifiedAppState


def get_app_config() -> Dict[str, Any]:
    """Get cached app configuration."""
    if "app_config" not in st.session_state:
        st.session_state.app_config = load_unified_config("unified_app")
    return st.session_state.app_config


def get_app_state() -> UnifiedAppState:
    """Get or create app state."""
    return UnifiedAppState.from_session()


def setup_page(title: str, icon: str, layout: str = "wide"):
    """Setup common page configuration."""
    st.set_page_config(
        page_title=f"{title} - OCR Studio",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
```

**Benefits**:
- Reduces code duplication
- Centralizes config management
- Makes pages cleaner

---

### Phase 2: Migrate Preprocessing Mode

#### 2.1: Create `pages/1_🎨_Preprocessing.py`

**Structure**:
```python
"""Preprocessing Studio - Interactive parameter tuning and pipeline visualization.

This page allows users to:
- Upload images and tune preprocessing parameters
- Visualize each stage of the pipeline
- Compare before/after results
- Export configurations as presets
"""

import streamlit as st
import hashlib
import json
from typing import Dict, Any

# Import shared utilities
from ui.apps.unified_ocr_app.shared_utils import get_app_config, get_app_state, setup_page

# Import only what THIS PAGE needs (lazy loading!)
from ui.apps.unified_ocr_app.services.config_loader import load_mode_config
from ui.apps.unified_ocr_app.components.preprocessing import render_parameter_panel, render_stage_viewer
from ui.apps.unified_ocr_app.components.preprocessing.parameter_panel import render_preset_management
from ui.apps.unified_ocr_app.components.shared import render_image_upload
from ui.apps.unified_ocr_app.services.preprocessing_service import PreprocessingService

# Setup page
setup_page("Preprocessing", "🎨")

# Get shared state and config
state = get_app_state()
app_config = get_app_config()
mode_config = load_mode_config("preprocessing", validate=False)

# Page title and description
st.title("🎨 Preprocessing Studio")
if "description" in mode_config:
    st.info(f"ℹ️ {mode_config['description']}")

# === SIDEBAR ===
with st.sidebar:
    st.header("Preprocessing Controls")

    # Image upload
    with st.expander("📤 Image Upload", expanded=True):
        render_image_upload(state, app_config.get("shared", {}))

    # Parameter panel
    current_params = render_parameter_panel(state, mode_config)

    # Preset management
    st.divider()
    render_preset_management(current_params, mode_config)

# === MAIN AREA ===
current_image = state.get_current_image()

if current_image is None:
    st.info("👈 Upload an image from the sidebar to start preprocessing")
    st.markdown("""
        ### 🎨 Preprocessing Studio

        This mode allows you to:
        - Tune preprocessing parameters interactively
        - Visualize each stage of the pipeline
        - Compare before/after results
        - Export configurations as presets

        **Get started by uploading an image!**
    """)
    st.stop()

# Process button
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    process_button = st.button(
        "🚀 Run Pipeline",
        use_container_width=True,
        type="primary",
        help="Execute preprocessing with current parameters",
    )

# Process image if button clicked
if process_button:
    with st.spinner("Processing image through pipeline..."):
        # Create service (cached)
        service = PreprocessingService(mode_config)

        # Generate cache key
        param_str = json.dumps(current_params, sort_keys=True)
        image_hash = hashlib.md5(current_image.tobytes()).hexdigest()[:8]
        cache_key = f"{image_hash}_{hashlib.md5(param_str.encode()).hexdigest()[:8]}"

        # Process
        result = service.process_image(current_image, current_params, cache_key)

        # Store in state
        state.preprocessing_results = result.get("stages", {})
        state.preprocessing_metadata = result.get("metadata", {})
        state.to_session()

        # Show success message
        num_stages = len(state.preprocessing_results) - 1
        total_time = result.get("metadata", {}).get("total_time", 0)
        st.success(f"✅ Processed {num_stages} stages in {total_time:.2f}s")

# Render stage viewer
render_stage_viewer(state, mode_config, state.preprocessing_results)
```

**File Stats**:
- Lines: ~150 (vs 90 in monolithic version)
- Imports: 7 (vs 15 in monolithic)
- Only loaded when user navigates to Preprocessing page

#### 2.2: Test Preprocessing Page

```bash
# Run the app
uv run streamlit run ui/apps/unified_ocr_app/app.py

# Navigate to Preprocessing page in browser
# Test all functionality:
# - Image upload
# - Parameter adjustment
# - Pipeline execution
# - Preset save/load
```

---

### Phase 3: Migrate Inference Mode

#### 3.1: Create `pages/2_🤖_Inference.py`

**Structure** (similar to preprocessing):
```python
"""Inference Studio - Run OCR models on images.

This page allows users to:
- Select trained model checkpoints
- Run single image or batch inference
- Visualize detected text regions
- Export results in multiple formats
"""

import streamlit as st
# ... (similar pattern to preprocessing page)
```

**Key Differences**:
- Has 2 sub-modes (single/batch)
- Loads checkpoints (must be cached!)
- More complex hyperparameter UI

**File Stats**:
- Lines: ~250 (includes both single and batch modes)
- Imports: 8
- Heavy resources: Model loading (must use `@st.cache_resource`)

---

### Phase 4: Migrate Comparison Mode

#### 4.1: Create `pages/3_📊_Comparison.py`

**Structure**:
```python
"""Comparison Studio - A/B test configurations and models.

This page allows users to:
- Compare different preprocessing configurations
- Test multiple inference hyperparameters
- Run end-to-end pipeline comparisons
- Analyze performance metrics
"""

import streamlit as st
# ... (similar pattern)
```

**File Stats**:
- Lines: ~225
- Imports: 10
- Most complex UI (tabs, metrics, charts)

---

### Phase 5: Simplify Main App

#### 5.1: Create New `app.py` (Home Page)

**File**: `ui/apps/unified_ocr_app/app.py` (NEW VERSION)

```python
"""Unified OCR Development Studio - Home Page.

Welcome screen and configuration for the multi-page app.
Select a mode from the sidebar to begin.
"""

import streamlit as st
from ui.apps.unified_ocr_app.shared_utils import get_app_config, get_app_state, setup_page

# Load configuration (cached globally)
config = get_app_config()

# Setup page
st.set_page_config(
    page_title=config["app"]["title"],
    page_icon=config["app"].get("page_icon", "🔍"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize state (shared across all pages)
state = get_app_state()

# === HOME PAGE UI ===
st.title(config["app"]["title"])

if "subtitle" in config["app"]:
    st.markdown(config["app"]["subtitle"])

st.divider()

st.info("👈 Select a mode from the sidebar to begin")

# Mode descriptions
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        ### 🎨 Preprocessing
        Tune image processing pipelines interactively.

        - 7-stage pipeline visualization
        - Real-time parameter adjustment
        - Preset management
        - Before/after comparison
    """)

with col2:
    st.markdown("""
        ### 🤖 Inference
        Run OCR models on images.

        - Single or batch processing
        - Multiple model checkpoints
        - Adjustable hyperparameters
        - Export results (JSON, CSV)
    """)

with col3:
    st.markdown("""
        ### 📊 Comparison
        A/B test configurations and models.

        - Parameter sweep
        - Side-by-side comparison
        - Performance metrics
        - Statistical analysis
    """)

st.divider()

# Quick stats
st.markdown("### 📊 Quick Stats")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Images Processed (Session)", state.get_image_count(), help="Images uploaded this session")

with col_b:
    st.metric("Current Mode", state.current_mode or "None", help="Last selected mode")

with col_c:
    st.metric("Pipeline Runs", len(state.preprocessing_results) if hasattr(state, "preprocessing_results") else 0)

# Footer
st.divider()
st.markdown("""
<sub>
💡 **Tip**: Use the sidebar to navigate between modes. Your session state is preserved across pages.
</sub>
""", unsafe_allow_html=True)
```

**File Stats**:
- Lines: ~100 (vs 725 before!)
- Imports: 2 (vs 15 before!)
- No mode-specific code
- Loads instantly

#### 5.2: Backup Old App

```bash
# Keep old version for reference
mv ui/apps/unified_ocr_app/app.py ui/apps/unified_ocr_app/app_old_monolithic.py

# Create new version
# (paste code from section 5.1)
```

---

### Phase 6: Update Services with Caching

#### 6.1: Fix InferenceService

**File**: `ui/apps/unified_ocr_app/services/inference_service.py`

**Before** (problematic):
```python
class InferenceService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = load_heavy_model()  # ❌ Loads every time!
```

**After** (cached):
```python
import streamlit as st

@st.cache_resource
def _load_inference_model(model_path: str):
    """Load inference model (cached across sessions)."""
    print(f"Loading inference model from {model_path}...")
    model = load_heavy_model(model_path)
    print("Model loaded successfully!")
    return model

class InferenceService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_path = config.get("model_path", "default_model.pth")
        self.model = _load_inference_model(model_path)  # ✅ Cached!
```

#### 6.2: Fix PreprocessingService

**Similar pattern** - cache heavy preprocessing pipelines

#### 6.3: Fix ComparisonService

**Similar pattern** - cache shared resources

---

## Migration Strategy

### Option A: Big Bang Migration (Recommended)

**Timeline**: 1-2 sessions
**Risk**: Medium
**Effort**: High

**Steps**:
1. Complete Phase 0 (fix heavy loading)
2. Create all 3 pages in one session (Phases 2-4)
3. Create new `app.py` (Phase 5)
4. Test thoroughly
5. Deploy all at once

**Pros**:
- ✅ Clean break from old architecture
- ✅ All benefits realized immediately
- ✅ No code duplication
- ✅ Easier to test (one codebase)

**Cons**:
- ❌ Higher risk if issues found
- ❌ More testing required before deploy
- ❌ Harder to rollback

### Option B: Incremental Migration

**Timeline**: 3-4 sessions
**Risk**: Low
**Effort**: Medium

**Steps**:
1. Complete Phase 0
2. Create `pages/` alongside current `app.py`
3. Migrate one mode per session
4. Run both versions in parallel (different ports)
5. Switch when all validated

**Pros**:
- ✅ Lower risk (can test each mode independently)
- ✅ Easy rollback (keep old version)
- ✅ Gradual user migration

**Cons**:
- ❌ Code duplication during transition
- ❌ Maintain two versions
- ❌ Longer migration period

### Recommendation

**Use Option A (Big Bang)** because:
1. You have good test coverage (Phases 0-7 complete)
2. The app is not in production yet
3. Clean architecture is worth the upfront effort
4. Easier to maintain one version

---

## Testing Plan

### Unit Tests

**New Files to Test**:
```python
# tests/ui/test_preprocessing_page.py
def test_preprocessing_page_imports():
    """Test preprocessing page imports work."""
    import ui.apps.unified_ocr_app.pages.1_🎨_Preprocessing as page
    assert page is not None

def test_preprocessing_page_no_heavy_load():
    """Test page doesn't load heavy resources at import."""
    import time
    start = time.time()
    import ui.apps.unified_ocr_app.pages.1_🎨_Preprocessing
    duration = time.time() - start
    assert duration < 2.0, f"Page took {duration}s to import (too slow!)"

# Similar for other pages...
```

### Integration Tests

```python
# tests/ui/test_multipage_navigation.py
def test_navigation_between_pages(streamlit_app):
    """Test users can navigate between pages."""
    # Start at home
    app.home_page()
    assert app.current_page == "home"

    # Navigate to preprocessing
    app.navigate("preprocessing")
    assert app.current_page == "preprocessing"

    # Navigate to inference
    app.navigate("inference")
    assert app.current_page == "inference"

    # State preserved
    assert app.session_state is not None
```

### Manual Testing Checklist

**Home Page**:
- [ ] Loads in < 2 seconds
- [ ] All 3 mode cards displayed
- [ ] Navigation links work
- [ ] Session stats accurate

**Preprocessing Page**:
- [ ] Image upload works
- [ ] Parameters adjustable
- [ ] Pipeline executes
- [ ] Results displayed
- [ ] Presets save/load

**Inference Page**:
- [ ] Checkpoint selection works
- [ ] Single image inference works
- [ ] Batch mode works
- [ ] Results visualization works

**Comparison Page**:
- [ ] Configuration UI works
- [ ] Comparison executes
- [ ] Metrics displayed
- [ ] Export works

**Cross-Page**:
- [ ] Session state preserved across pages
- [ ] No memory leaks
- [ ] Navigation is smooth
- [ ] URLs are bookmarkable

---

## Rollback Plan

### If Migration Fails

**Step 1**: Restore old app.py
```bash
cd ui/apps/unified_ocr_app
mv app.py app_new_failed.py
mv app_old_monolithic.py app.py
```

**Step 2**: Remove pages directory
```bash
rm -rf pages/
```

**Step 3**: Restart Streamlit
```bash
pkill -9 streamlit
uv run streamlit run ui/apps/unified_ocr_app/app.py
```

### If Partial Migration

**Keep both versions**:
```bash
# Old version on port 8501
uv run streamlit run ui/apps/unified_ocr_app/app_old_monolithic.py --server.port 8501

# New version on port 8502
uv run streamlit run ui/apps/unified_ocr_app/app.py --server.port 8502
```

---

## Success Metrics

### Performance
- [ ] Home page load < 2s
- [ ] Page navigation < 1s
- [ ] Memory usage < 2GB per user
- [ ] No resource leaks

### Code Quality
- [ ] Lines per file < 300
- [ ] Imports per file < 10
- [ ] Cyclomatic complexity < 10
- [ ] Test coverage > 80%

### User Experience
- [ ] Mode switching feels instant
- [ ] No broken functionality
- [ ] URLs are bookmarkable
- [ ] Session state preserved

### Developer Experience
- [ ] Easy to find code for each mode
- [ ] Can work on modes in parallel
- [ ] Clear separation of concerns
- [ ] Good documentation

---

## Timeline

| Phase | Task | Effort | Dependencies |
|-------|------|--------|--------------|
| Phase 0 | Fix heavy loading | 0.5 sessions | None |
| Phase 1 | Create structure | 0.25 sessions | Phase 0 |
| Phase 2 | Migrate preprocessing | 0.5 sessions | Phase 1 |
| Phase 3 | Migrate inference | 0.75 sessions | Phase 1 |
| Phase 4 | Migrate comparison | 0.75 sessions | Phase 1 |
| Phase 5 | Simplify main app | 0.25 sessions | Phases 2-4 |
| Phase 6 | Update services | 0.5 sessions | Phase 0 |
| Testing | Manual + automated | 0.5 sessions | All phases |
| **Total** | | **4-5 sessions** | |

**Optimistic**: 2-3 sessions (if no blockers)
**Realistic**: 3-4 sessions
**Pessimistic**: 5-6 sessions (if issues found)

---

## Questions & Decisions

### Decision Required

1. **Which migration strategy?**
   - [ ] Option A: Big Bang (recommended)
   - [ ] Option B: Incremental

2. **When to start?**
   - [ ] Immediately after fixing heavy loading
   - [ ] After Phase 7 documentation complete
   - [ ] After user testing

3. **Who will do the work?**
   - [ ] Original developer (knows codebase)
   - [ ] New developer (fresh perspective)
   - [ ] Pair programming

### Open Questions

1. Will URL structure change break any bookmarks?
2. Should we preserve old app.py as `/legacy`?
3. Do we need a migration guide for users?
4. Should we update docs before or after migration?

---

## References

### Streamlit Documentation
- [Multi-Page Apps](https://docs.streamlit.io/library/get-started/multipage-apps)
- [Session State](https://docs.streamlit.io/library/api-reference/session-state)
- [Caching](https://docs.streamlit.io/library/advanced-features/caching)

### Project Documentation
- [Architecture](UNIFIED_STREAMLIT_APP_ARCHITECTURE.md)
- Session Handover
- Phase 7 Complete

### Related Files
- Current app: [ui/apps/unified_ocr_app/app.py](../../../ui/apps/unified_ocr_app/app.py)
- Components: [ui/apps/unified_ocr_app/components/](../../../ui/apps/unified_ocr_app/components/)
- Services: [ui/apps/unified_ocr_app/services/](../../../ui/apps/unified_ocr_app/services/)

---

**Document Status**: 📋 DRAFT
**Next Review**: After Phase 0 completion
**Owner**: TBD
**Last Updated**: 2025-10-21
