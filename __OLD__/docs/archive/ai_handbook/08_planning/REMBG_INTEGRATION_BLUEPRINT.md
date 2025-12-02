# Living Implementation Blueprint: rembg Background Removal Integration

**Blueprint ID**: `REMBG-INTEGRATION-001`
**Status**: 🟢 READY TO EXECUTE
**Created**: 2025-10-21
**Last Updated**: 2025-10-21
**Owner**: Development Team
**Estimated Duration**: 2-3 days

---

## Blueprint Metadata

```yaml
blueprint:
  type: feature_integration
  complexity: medium
  risk_level: low
  dependencies:
    - rembg==2.0.67 (installed)
    - onnxruntime (installed)
    - existing BackgroundRemoval class
  references:
    - docs/ai_handbook/08_planning/REMBG_INTEGRATION_SUMMARY.md
    - docs/ai_handbook/03_references/guides/background_removal_rembg.md
    - docs/ai_handbook/08_planning/preprocessing_viewer_refactor_plan.md
    - ocr/datasets/preprocessing/background_removal.py
```

---

## 1. Executive Summary

### What We're Building
Integration of rembg-based AI background removal into:
1. **Preprocessing Pipeline** (ui/preprocessing_viewer/pipeline.py)
2. **Streamlit Real-time Inference App** (ui/preprocessing_viewer_app.py)

### Why Now
- ✅ rembg already installed and tested
- ✅ `BackgroundRemoval` class already exists (ocr/datasets/preprocessing/background_removal.py)
- ✅ Solves real problems (cluttered backgrounds, shadows)
- ✅ Portfolio differentiator (AI-powered preprocessing)

### Success Criteria
- [ ] Background removal toggle appears in Streamlit UI
- [ ] Processing completes in <3s per image (CPU)
- [ ] Before/after comparison visible in viewer
- [ ] No performance regression in existing pipeline
- [ ] Clean integration with existing caching

---

## 2. Current State Analysis

### What Exists ✅

1. **BackgroundRemoval Transform** (ocr/datasets/preprocessing/background_removal.py)
   - Albumentations-compatible transform
   - Handles RGBA → RGB conversion
   - White background compositing
   - Multiple model support

2. **Preprocessing Pipeline** (ui/preprocessing_viewer/pipeline.py)
   - Sequential stage processing
   - Intermediate result capture
   - Config validation
   - Telemetry integration

3. **Streamlit App** (ui/preprocessing_viewer_app.py)
   - Parameter controls
   - Side-by-side viewer
   - Config export
   - Preset management

### What's Missing ❌

1. **Pipeline Integration**
   - No `BackgroundRemoval` initialization in `PreprocessingViewerPipeline.__init__`
   - No stage added to `process_with_intermediates`
   - No stage description in metadata

2. **UI Controls**
   - No background removal toggle
   - No model selection dropdown
   - No alpha matting option
   - No mask visualization

3. **Performance Optimization**
   - No lazy loading of rembg models
   - No result caching for background removal
   - No progress indicator

---

## 3. Implementation Plan

### Phase 1: Pipeline Integration (Day 1)

#### Task 1.1: Add BackgroundRemoval to Pipeline
**File**: ui/preprocessing_viewer/pipeline.py

**Changes**:
```python
# IMPORT (add to line 15-26)
from ocr.datasets.preprocessing.background_removal import BackgroundRemoval

# INIT (add to __init__ after line 41)
class PreprocessingViewerPipeline:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

        # NEW: Background removal (lazy init to avoid loading model unnecessarily)
        self._background_removal: BackgroundRemoval | None = None

        # ... existing components ...
```

#### Task 1.2: Add Lazy Property for Background Removal
**Location**: After `__init__` method

**Changes**:
```python
@property
def background_removal(self) -> BackgroundRemoval:
    """Lazy-load background removal to avoid loading model until needed."""
    if self._background_removal is None:
        self.logger.info("Initializing BackgroundRemoval (loading rembg model...)")
        self._background_removal = BackgroundRemoval(
            model="u2net",
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            p=1.0
        )
    return self._background_removal
```

#### Task 1.3: Add Stage to Processing Pipeline
**File**: ui/preprocessing_viewer/pipeline.py
**Location**: In `process_with_intermediates` method, after line 100

**Changes**:
```python
def process_with_intermediates(
    self, image: np.ndarray, config: dict[str, Any], roi: tuple[int, int, int, int] | None = None
) -> dict[str, np.ndarray | str]:
    """Process image through pipeline and capture all intermediate results."""
    self.logger.info(f"Starting preprocessing pipeline - image shape: {image.shape}")
    results: dict[str, np.ndarray | str] = {"original": image.copy()}
    current_image = image.copy()

    # NEW: Stage 1 - Background Removal (FIRST in pipeline)
    if config.get("enable_background_removal", False):
        try:
            self.logger.info("Applying background removal...")
            bg_result = self.background_removal.apply(current_image)
            results["background_removed"] = bg_result
            current_image = bg_result
            self.logger.info(f"Background removal completed - shape: {bg_result.shape}")
        except Exception as e:
            self.logger.error(f"Background removal failed: {e}", exc_info=True)
            results["background_removal_error"] = str(e)

    # Stage 2 - ROI Extraction (existing code continues...)
    if roi is not None:
        # ... existing ROI code ...
```

#### Task 1.4: Add Stage Metadata
**Location**: After `process_with_intermediates` method

**Changes**:
```python
def get_available_stages(self) -> list[str]:
    """Return list of all available pipeline stages."""
    return [
        "original",
        "background_removed",  # NEW
        "roi_extracted",
        "grayscale",
        "color_inverted",
        "document_detected",
        # ... rest of stages ...
    ]

def get_stage_description(self, stage: str) -> str:
    """Get human-readable description for a stage."""
    descriptions = {
        "original": "Original uploaded image",
        "background_removed": "AI background removal (rembg U²-Net)",  # NEW
        "roi_extracted": "Region of interest extracted",
        # ... rest of descriptions ...
    }
    return descriptions.get(stage, stage.replace("_", " ").title())
```

**Verification**:
- [ ] Code compiles without errors
- [ ] Lazy loading works (model not loaded on import)
- [ ] Stage appears in available stages list

---

### Phase 2: UI Controls (Day 1-2)

#### Task 2.1: Add Preset Configuration
**File**: ui/preprocessing_viewer/preset_manager.py
**Location**: `get_default_config` method

**Changes**:
```python
def get_default_config(self) -> dict[str, Any]:
    """Get default preprocessing configuration."""
    return {
        # NEW: Background Removal
        'enable_background_removal': False,  # Disabled by default (expensive)
        'background_removal_model': 'u2net',
        'background_removal_alpha_matting': True,

        # Existing configs...
        'enable_document_detection': True,
        # ... rest ...
    }
```

#### Task 2.2: Add Parameter Controls
**File**: ui/preprocessing_viewer/parameter_controls.py
**Location**: `render_parameter_panels` method

**Changes**:
```python
def render_parameter_panels(self, config: dict[str, Any], on_change: Callable) -> None:
    """Render all parameter control panels."""

    # NEW: Background Removal Panel (add as FIRST panel)
    with st.expander("🎨 Background Removal (AI)", expanded=False):
        st.markdown("**Remove cluttered backgrounds using AI (U²-Net model)**")
        st.info("⚠️ First use downloads ~176MB model. Processing takes 1-3s per image.")

        enable = st.checkbox(
            "Enable Background Removal",
            value=config.get("enable_background_removal", False),
            key="enable_background_removal",
            help="Use AI to remove background (recommended for cluttered photos)"
        )

        if enable:
            model = st.selectbox(
                "Model",
                options=["u2net", "u2netp", "silueta"],
                index=0 if config.get("background_removal_model", "u2net") == "u2net" else 1,
                key="bg_removal_model",
                help="u2net: Best quality (slower), u2netp: Faster (lower quality)"
            )

            alpha_matting = st.checkbox(
                "Alpha Matting (Better Edges)",
                value=config.get("background_removal_alpha_matting", True),
                key="bg_alpha_matting",
                help="Improves edge quality but adds ~0.5s processing time"
            )

            # Update config via callback
            updated_config = config.copy()
            updated_config["enable_background_removal"] = enable
            updated_config["background_removal_model"] = model
            updated_config["background_removal_alpha_matting"] = alpha_matting
            on_change(updated_config)

    # Existing panels continue...
```

#### Task 2.3: Update Main App Display
**File**: ui/preprocessing_viewer_app.py
**Location**: "Full Pipeline" tab rendering

**Changes**:
```python
# In "Full Pipeline" tab (around line 156)
st.header("🎭 Individual Stages")

# NEW: Special handling for background removal comparison
if "background_removed" in pipeline_results:
    st.subheader("Before/After: Background Removal")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original")
        st.image(
            SideBySideViewer.prepare_image_for_display(pipeline_results["original"]),
            use_column_width=True
        )
    with col2:
        st.caption("Background Removed (AI)")
        st.image(
            SideBySideViewer.prepare_image_for_display(pipeline_results["background_removed"]),
            use_column_width=True
        )
    st.markdown("---")

# Existing stage grid continues...
```

**Verification**:
- [ ] Toggle appears in UI
- [ ] Changing toggle triggers reprocessing
- [ ] Model selection works
- [ ] Before/after comparison renders correctly

---

### Phase 3: Performance Optimization (Day 2)

#### Task 3.1: Add Session State Caching
**File**: ui/preprocessing_viewer_app.py
**Location**: After `main()` function start

**Changes**:
```python
def main():
    """Main preprocessing viewer application."""
    st.set_page_config(...)

    # NEW: Initialize background removal session state
    if "bg_removal_model_loaded" not in st.session_state:
        st.session_state.bg_removal_model_loaded = False

    if "bg_removal_cache" not in st.session_state:
        st.session_state.bg_removal_cache = {}  # Cache by image hash

    # ... rest of initialization ...
```

#### Task 3.2: Add Progress Indicator
**File**: ui/preprocessing_viewer/pipeline.py
**Location**: In background removal stage

**Changes**:
```python
# In process_with_intermediates, background removal stage
if config.get("enable_background_removal", False):
    try:
        # NEW: Show progress for first load
        if not st.session_state.get("bg_removal_model_loaded", False):
            with st.spinner("Loading rembg model (first use only, ~176MB)..."):
                bg_result = self.background_removal.apply(current_image)
                st.session_state.bg_removal_model_loaded = True
        else:
            with st.spinner("Removing background..."):
                bg_result = self.background_removal.apply(current_image)

        results["background_removed"] = bg_result
        current_image = bg_result
    except Exception as e:
        # ... error handling ...
```

#### Task 3.3: Add Result Caching
**File**: ui/preprocessing_viewer_app.py
**Location**: Before pipeline processing

**Changes**:
```python
# In "Full Pipeline" tab (around line 138)
with st.spinner("Running preprocessing pipeline..."):
    try:
        # NEW: Check cache for background removal results
        image_hash = hash(image.tobytes())
        config_hash = hash(str(sorted(st.session_state.viewer_config.items())))
        cache_key = f"{image_hash}_{config_hash}"

        if cache_key in st.session_state.bg_removal_cache:
            pipeline_results = st.session_state.bg_removal_cache[cache_key]
            st.info("Using cached results")
        else:
            pipeline_results = pipeline.process_with_intermediates(
                image, st.session_state.viewer_config
            )
            st.session_state.bg_removal_cache[cache_key] = pipeline_results
    except Exception as exc:
        # ... error handling ...
```

**Verification**:
- [ ] First load shows "downloading model" message
- [ ] Subsequent loads use cached model
- [ ] Results cached per (image, config) pair
- [ ] Cache cleared when config changes

---

### Phase 4: Testing & Validation (Day 3)

#### Task 4.1: Create Integration Test
**File**: `tests/ui/test_background_removal_integration.py` (NEW)

**Content**:
```python
"""Integration tests for background removal in preprocessing viewer."""
import numpy as np
import pytest
from ui.preprocessing_viewer.pipeline import PreprocessingViewerPipeline


@pytest.fixture
def test_image():
    """Create a test image with colored regions."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[100:300, 200:400] = [255, 255, 255]  # White foreground
    img[0:100, :] = [100, 100, 100]  # Gray background
    return img


def test_background_removal_disabled_by_default(test_image):
    """Test that background removal is disabled by default."""
    pipeline = PreprocessingViewerPipeline()
    config = {"enable_background_removal": False}

    results = pipeline.process_with_intermediates(test_image, config)

    assert "background_removed" not in results


def test_background_removal_enabled(test_image):
    """Test that background removal works when enabled."""
    pipeline = PreprocessingViewerPipeline()
    config = {"enable_background_removal": True}

    results = pipeline.process_with_intermediates(test_image, config)

    assert "background_removed" in results
    assert results["background_removed"].shape == test_image.shape
    assert results["background_removed"].dtype == np.uint8


def test_background_removal_lazy_loading():
    """Test that BackgroundRemoval is lazy-loaded."""
    pipeline = PreprocessingViewerPipeline()

    # Should not be loaded yet
    assert pipeline._background_removal is None

    # Access property to trigger load
    _ = pipeline.background_removal

    # Now should be loaded
    assert pipeline._background_removal is not None


def test_background_removal_error_handling(test_image):
    """Test error handling when background removal fails."""
    pipeline = PreprocessingViewerPipeline()
    config = {"enable_background_removal": True}

    # Corrupt the image to trigger error
    bad_image = np.zeros((0, 0, 3), dtype=np.uint8)

    results = pipeline.process_with_intermediates(bad_image, config)

    # Should capture error, not crash
    assert "background_removal_error" in results or len(results) > 0
```

**Run Tests**:
```bash
uv run pytest tests/ui/test_background_removal_integration.py -v
```

#### Task 4.2: Manual Testing Checklist

**Test Scenario 1: Enable/Disable Toggle**
- [ ] Upload test image
- [ ] Enable background removal → see processing spinner
- [ ] Verify before/after comparison appears
- [ ] Disable background removal → comparison disappears
- [ ] Re-enable → uses cached model (faster)

**Test Scenario 2: Model Selection**
- [ ] Select "u2net" → slower, higher quality
- [ ] Select "u2netp" → faster, acceptable quality
- [ ] Compare results side-by-side

**Test Scenario 3: Alpha Matting**
- [ ] Enable alpha matting → better edges, slower
- [ ] Disable alpha matting → faster, rougher edges
- [ ] Inspect edge quality on complex image

**Test Scenario 4: Performance**
- [ ] First image: ~3-5s (model download + process)
- [ ] Same image again: <0.1s (cached)
- [ ] Different image: ~1-2s (model loaded, new image)
- [ ] Memory usage: <800MB total

**Test Scenario 5: Error Handling**
- [ ] Upload corrupted image → graceful error message
- [ ] Disconnect internet during first load → retry message
- [ ] Process very large image (>4000px) → warning + resize

#### Task 4.3: Visual Regression Test

**Create golden images**:
```bash
# Save test outputs for comparison
mkdir -p tests/ui/golden_images/background_removal/
uv run python -c "
from ui.preprocessing_viewer.pipeline import PreprocessingViewerPipeline
import cv2

pipeline = PreprocessingViewerPipeline()
image = cv2.imread('path/to/test/receipt.jpg')
config = {'enable_background_removal': True}
results = pipeline.process_with_intermediates(image, config)

cv2.imwrite('tests/ui/golden_images/background_removal/original.png', results['original'])
cv2.imwrite('tests/ui/golden_images/background_removal/removed.png', results['background_removed'])
"
```

**Verification**:
- [ ] Golden images created
- [ ] Visual inspection confirms quality
- [ ] Documented in test README

---

## 4. Risk Assessment & Mitigation

### Risk 1: Model Download Failure (Medium)
**Impact**: Users can't use feature on first run
**Probability**: Low (rembg auto-retries)
**Mitigation**:
- Add explicit error message with troubleshooting steps
- Provide manual model download instructions
- Fallback to non-AI background removal (threshold-based)

**Code**:
```python
try:
    bg_result = self.background_removal.apply(current_image)
except Exception as e:
    if "download" in str(e).lower() or "connection" in str(e).lower():
        self.logger.error("Model download failed. Using fallback method.")
        st.warning("⚠️ AI model download failed. Using simple background removal.")
        # Fallback: Use GrabCut or threshold-based method
        bg_result = self._fallback_background_removal(current_image)
    else:
        raise
```

### Risk 2: Performance Regression (Medium)
**Impact**: UI becomes slow, users frustrated
**Probability**: Medium (AI model is expensive)
**Mitigation**:
- Disabled by default
- Clear performance warnings in UI
- Aggressive caching
- Option to use lighter model (u2netp)

**Monitoring**:
```python
# Add telemetry
import time
start = time.time()
bg_result = self.background_removal.apply(current_image)
duration = time.time() - start
self.logger.info(f"Background removal took {duration:.2f}s")
if duration > 5.0:
    st.warning(f"⚠️ Background removal took {duration:.1f}s. Consider using 'u2netp' model for faster processing.")
```

### Risk 3: Memory Issues (Low)
**Impact**: App crashes on large images
**Probability**: Low (rembg handles resizing)
**Mitigation**:
- Warn users about large images
- Resize input before processing
- Clear cache periodically

**Code**:
```python
# Add size check before processing
if current_image.shape[0] * current_image.shape[1] > 4_000_000:  # 4MP
    st.warning("⚠️ Large image detected. Resizing for background removal...")
    scale = (2000 * 1500) / (current_image.shape[0] * current_image.shape[1])
    scale = scale ** 0.5
    current_image = cv2.resize(current_image, None, fx=scale, fy=scale)
```

### Risk 4: Unexpected rembg Output (Low)
**Impact**: Pipeline crashes on malformed output
**Probability**: Low (rembg is stable)
**Mitigation**:
- Validate output shape and dtype
- Add explicit type checks
- Graceful degradation

**Code**:
```python
bg_result = self.background_removal.apply(current_image)

# Validate output
if bg_result.shape[:2] != current_image.shape[:2]:
    self.logger.warning(f"Background removal changed image size: {current_image.shape} → {bg_result.shape}")
    bg_result = cv2.resize(bg_result, (current_image.shape[1], current_image.shape[0]))

if bg_result.dtype != np.uint8:
    bg_result = bg_result.astype(np.uint8)
```

---

## 5. Success Metrics

### Functional Metrics
- [ ] Background removal toggle functional
- [ ] Model selection works (u2net, u2netp, silueta)
- [ ] Alpha matting toggle works
- [ ] Before/after comparison renders
- [ ] Cache reduces repeat processing time by >90%

### Performance Metrics
- [ ] First load: <10s (includes model download)
- [ ] Subsequent loads: <3s (model cached)
- [ ] Cached results: <0.1s (instant)
- [ ] Memory usage: <1GB total
- [ ] No UI freezing or blocking

### Quality Metrics
- [ ] Background removed on cluttered images (visual inspection)
- [ ] Text regions preserved (no blurring)
- [ ] Edge quality acceptable (with alpha matting)
- [ ] White background consistent (no artifacts)

### User Experience Metrics
- [ ] Clear loading indicators
- [ ] Helpful error messages
- [ ] Performance warnings for slow operations
- [ ] Intuitive UI controls
- [ ] Responsive interface (no hangs)

---

## 6. Rollout Plan

### Phase 1: Development (Day 1-3)
- Complete all implementation tasks
- Run integration tests
- Manual testing on diverse images

### Phase 2: Internal Testing (Day 3)
- Test with 20+ real receipt images
- Document any quality issues
- Performance profiling

### Phase 3: Soft Launch (Day 3)
- Deploy to staging/development environment
- Gather feedback from team
- Monitor performance metrics

### Phase 4: Documentation (Day 3)
- Update background_removal_rembg.md
- Add troubleshooting guide
- Create demo video/GIF

### Phase 5: Production (After validation)
- Merge to main branch
- Update portfolio with demo
- Announce feature

---

## 7. Rollback Plan

### If Performance Unacceptable
1. Disable by default in preset_manager.py
2. Add "Experimental" label to UI
3. Recommend u2netp model as default

### If Critical Bug Found
1. Comment out background removal stage in pipeline.py
2. Hide UI controls in parameter_controls.py
3. Add banner: "Background removal temporarily disabled"

### If Model Download Fails
1. Implement fallback to threshold-based method
2. Add clear error message with workaround
3. Document manual model installation

**Rollback Diff**:
```python
# Quick disable in preset_manager.py
def get_default_config(self) -> dict[str, Any]:
    return {
        'enable_background_removal': False,  # ROLLBACK: Disabled due to [reason]
        # ... rest ...
    }
```

---

## 8. Dependencies & Prerequisites

### Required
- [x] rembg==2.0.67 (installed)
- [x] onnxruntime (installed)
- [x] BackgroundRemoval class (ocr/datasets/preprocessing/background_removal.py)

### Nice to Have
- [ ] GPU available (10x speedup)
- [ ] Fast internet (first-time model download)
- [ ] Test images with cluttered backgrounds

### Blockers
- None identified

---

## 9. Implementation Checklist

### Day 1: Pipeline Integration
- [ ] Add BackgroundRemoval import to pipeline.py
- [ ] Add lazy loading property
- [ ] Add stage to process_with_intermediates
- [ ] Add stage metadata (available_stages, descriptions)
- [ ] Test: verify stage appears in pipeline
- [ ] Test: verify lazy loading works

### Day 1-2: UI Controls
- [ ] Add background removal config to preset_manager.py
- [ ] Add parameter controls panel
- [ ] Add before/after comparison in main app
- [ ] Test: verify toggle enables/disables feature
- [ ] Test: verify model selection works
- [ ] Test: verify alpha matting works

### Day 2: Performance Optimization
- [ ] Add session state caching
- [ ] Add progress indicators
- [ ] Add result caching by (image, config)
- [ ] Test: verify first load shows progress
- [ ] Test: verify cached results are instant
- [ ] Test: verify cache invalidation works

### Day 3: Testing & Validation
- [ ] Create integration tests
- [ ] Run automated tests (pytest)
- [ ] Manual testing: 5 test scenarios
- [ ] Create golden images
- [ ] Performance profiling
- [ ] Memory usage check

### Day 3: Documentation & Deployment
- [ ] Update background_removal_rembg.md
- [ ] Create troubleshooting guide
- [ ] Document performance characteristics
- [ ] Create demo GIF/video
- [ ] Update CHANGELOG.md
- [ ] Merge to main branch

---

## 10. Code Review Checklist

### Architecture
- [ ] Lazy loading prevents unnecessary model initialization
- [ ] Error handling comprehensive (download, processing, validation)
- [ ] Caching strategy sound (by image+config hash)
- [ ] No tight coupling (can disable without breaking)

### Code Quality
- [ ] Type hints present on all new functions
- [ ] Docstrings added for new methods
- [ ] Logging statements appropriate
- [ ] No hardcoded values (use config)

### Testing
- [ ] Integration tests cover happy path
- [ ] Error cases tested (corrupt image, download fail)
- [ ] Performance benchmarks documented
- [ ] Visual regression tests created

### UI/UX
- [ ] Loading indicators present
- [ ] Error messages helpful
- [ ] Performance warnings shown
- [ ] Controls intuitive

### Documentation
- [ ] Implementation guide complete
- [ ] Troubleshooting section added
- [ ] Performance characteristics documented
- [ ] Demo/examples provided

---

## 11. Post-Implementation Tasks

### Week 1
- [ ] Monitor performance metrics (processing time, memory)
- [ ] Collect user feedback
- [ ] Fix any reported bugs
- [ ] Optimize if needed (model selection, caching)

### Week 2
- [ ] Write blog post / portfolio entry
- [ ] Create demo video for portfolio
- [ ] Document lessons learned
- [ ] Plan advanced features (batch processing, custom models)

### Future Enhancements
- [ ] Add mask visualization (show what was removed)
- [ ] Support batch processing (multiple images)
- [ ] Add confidence score display
- [ ] Integrate custom-trained models
- [ ] Add background replacement (not just removal)

---

## 12. References

### Documentation
- REMBG_INTEGRATION_SUMMARY.md
- background_removal_rembg.md
- preprocessing_viewer_refactor_plan.md

### Code
- background_removal.py - BackgroundRemoval class
- pipeline.py - Preprocessing pipeline
- preprocessing_viewer_app.py - Streamlit app

### External
- [rembg GitHub](https://github.com/danielgatis/rembg)
- [U²-Net Paper](https://arxiv.org/abs/2005.09007)
- [rembg PyPI](https://pypi.org/project/rembg/)

---

## 13. Questions & Decisions Log

### Q1: Should background removal be first or after detection?
**Decision**: First (before detection)
**Rationale**: Cleaner input improves detection accuracy
**Source**: REMBG_INTEGRATION_SUMMARY.md

### Q2: Should it be enabled by default?
**Decision**: No (disabled by default)
**Rationale**: Expensive operation, not needed for all images
**Source**: Risk assessment (performance regression)

### Q3: Which model should be default?
**Decision**: u2net (best quality)
**Rationale**: Portfolio piece, quality > speed
**Alternatives**: u2netp for faster processing if needed

### Q4: Should we implement fallback method?
**Decision**: Yes (threshold-based as fallback)
**Rationale**: Handles model download failures gracefully
**Implementation**: Risk mitigation section

---

## 14. Status Updates

### 2025-10-21: Blueprint Created
- Analyzed existing codebase
- Identified integration points
- Created 3-day implementation plan
- Status: 🟢 READY TO EXECUTE

### [Future updates will go here]

---

## 15. Appendix: Quick Start Commands

### Start Development
```bash
# Ensure on correct branch
git checkout 11_refactor/preprocessing
git pull origin 11_refactor/preprocessing

# Verify rembg installed
uv pip list | grep rembg

# Run existing tests
uv run pytest tests/ui/ -v

# Start Streamlit app for testing
uv run streamlit run ui/preprocessing_viewer_app.py
```

### Test Background Removal Standalone
```bash
# Test rembg works
uv run python tests/debug/test_rembg_demo.py path/to/test/image.jpg

# Check results
ls rembg_results/
```

### Run Integration Tests
```bash
# After implementation
uv run pytest tests/ui/test_background_removal_integration.py -v
```

### Profile Performance
```bash
# Time background removal
uv run python -c "
import time
import cv2
from ui.preprocessing_viewer.pipeline import PreprocessingViewerPipeline

pipeline = PreprocessingViewerPipeline()
image = cv2.imread('path/to/test/image.jpg')
config = {'enable_background_removal': True}

start = time.time()
results = pipeline.process_with_intermediates(image, config)
duration = time.time() - start

print(f'Total time: {duration:.2f}s')
print(f'Stages: {list(results.keys())}')
"
```

---

**END OF BLUEPRINT**

**Next Action**: Begin Day 1 implementation (Pipeline Integration)
**Estimated Start**: Immediately
**Estimated Completion**: 3 days from start
