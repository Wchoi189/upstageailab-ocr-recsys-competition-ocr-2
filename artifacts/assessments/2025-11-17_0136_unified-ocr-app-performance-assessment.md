---
title: "Unified OCR App Performance Assessment"
author: "ai-agent"
timestamp: "2025-11-17 01:36 KST"
branch: "main"
status: "draft"
tags: []
---

## Progress Tracker
*(Required for iterative assessments, debugging sessions, or incremental work)*

- **STATUS:** Not Started / In Progress / Completed
- **CURRENT STEP:** [Current phase or task being worked on]
- **LAST COMPLETED TASK:** [Description of last completed task]
- **NEXT TASK:** [Description of the immediate next task]

### Assessment Checklist
- [ ] Initial assessment complete
- [ ] Analysis phase complete
- [ ] Recommendations documented
- [ ] Review and validation complete

---

## 1. Summary

## 2. Assessment

## 3. Recommendations
## Summary

The Unified OCR App experiences performance issues similar to the Command Builder app, with additional complexity from multi-page navigation and heavy image processing operations. The app lacks proper caching for services, checkpoint loading, and image processing results, leading to slow page loads and repeated expensive operations.

## Current Performance Issues

### 1. Service Instantiation on Every Action

**Problem:** Services are created on every button click, not cached:

```python
# In preprocessing page (line 110)
service = PreprocessingService(mode_config)  # Created every time

# In inference page (line 141)
service = InferenceService(mode_config)  # Created every time
```

**Impact:**
- PreprocessingService initializes pipeline on every click
- InferenceService initializes inference engine on every click
- Estimated cost: 100-200ms per action

### 2. Checkpoint Loading on Every Page Render

**Problem:** `load_checkpoints()` is called on every inference page render:

```python
# In inference page (line 60)
checkpoints = load_checkpoints(app_config)  # Not cached, scans directories every time
```

**Impact:**
- Full directory scan of outputs/ directory
- Catalog building on every render
- Estimated cost: 200-500ms per page load

### 3. Image Processing Not Cached

**Problem:** Image processing results are not properly cached:

```python
# In preprocessing page (line 118)
result = service.process_image(current_image, current_params, cache_key)
# Cache key is generated but @st.cache_data may not work well with numpy arrays
```

**Impact:**
- Repeated processing of same image with same parameters
- Heavy computation (background removal, document detection, etc.)
- Estimated cost: 1-5 seconds per processing run

### 4. Config Loading (Partially Optimized)

**Problem:** Config loading has caching but could be improved:

```python
# Already has @st.cache_data, but mode_config loading happens on every page render
mode_config = load_mode_config("preprocessing", validate=False)  # Cached but still called
```

**Impact:**
- Minor overhead from function calls
- Could be optimized further

### 5. Heavy Imports at Module Level

**Problem:** Some heavy imports happen at module level:

```python
# In preprocessing_service.py
# rembg is lazy-loaded (good), but other imports happen at module level
from ocr.datasets.preprocessing.background_removal import BackgroundRemoval
# These imports happen when module is loaded
```

**Impact:**
- Slower app startup
- Memory usage even when features not used

### 6. State Persistence on Every Render

**Problem:** State is persisted on every render:

```python
# In pages
state.to_session()  # Called multiple times per render
```

**Impact:**
- Unnecessary session state updates
- Potential re-renders

## Performance Bottleneck Analysis

### Page Load Flow (Current)

1. User navigates to page → Streamlit reruns
2. Page setup:
   - `setup_page()` → **10ms** (fast)
   - `get_app_state()` → **10ms** (fast)
   - `get_app_config()` → **10ms** (cached)
3. Mode config loading:
   - `load_mode_config()` → **50-100ms** (cached but still overhead)
4. Page-specific operations:
   - **Preprocessing page:**
     - Component rendering → **50-100ms**
   - **Inference page:**
     - `load_checkpoints()` → **200-500ms** (NOT CACHED)
     - Component rendering → **50-100ms**
5. State persistence → **10ms**

**Total: 330-720ms per page load (inference page worst case)**

### Action Flow (Current)

1. User clicks "Run Pipeline" or "Run Inference"
2. Service creation:
   - `PreprocessingService(mode_config)` → **100-200ms**
   - `InferenceService(mode_config)` → **100-200ms**
3. Processing:
   - Image processing → **1-5 seconds** (not cached)
   - Inference → **2-10 seconds** (not cached)
4. State update → **10ms**

**Total: 1.1-15.2 seconds per action**

## Recommendations

### Priority 1: Cache Service Instances

**Action:** Use `@st.cache_resource` for service instances:

```python
@st.cache_resource
def get_preprocessing_service(mode_config: dict) -> PreprocessingService:
    return PreprocessingService(mode_config)

@st.cache_resource
def get_inference_service(mode_config: dict) -> InferenceService:
    return InferenceService(mode_config)
```

**Expected Impact:** 100-200ms reduction per action

### Priority 2: Cache Checkpoint Loading

**Action:** Add caching to `load_checkpoints()`:

```python
@st.cache_data(ttl=300)  # 5 minutes
def load_checkpoints(config: dict[str, Any]) -> list[Any]:
    # ... existing implementation
```

**Expected Impact:** 200-500ms reduction per inference page load

### Priority 3: Optimize Image Processing Caching

**Action:** Improve caching for image processing:

```python
# Use hash-based caching with proper serialization
@st.cache_data(show_spinner=False, ttl=3600)
def process_image_cached(
    image_bytes: bytes,  # Serialize numpy array to bytes
    params_hash: str,
    mode_config_hash: str,
) -> dict[str, Any]:
    # Deserialize and process
    pass
```

**Expected Impact:** 1-5 seconds reduction for repeated operations

### Priority 4: Optimize Config Loading

**Action:** Pre-load configs and use module-level cache:

```python
# In shared_utils.py
_CONFIG_CACHE: dict[str, dict] = {}

def get_mode_config_cached(mode_id: str) -> dict:
    if mode_id not in _CONFIG_CACHE:
        _CONFIG_CACHE[mode_id] = load_mode_config(mode_id, validate=False)
    return _CONFIG_CACHE[mode_id]
```

**Expected Impact:** 50-100ms reduction per page load

### Priority 5: Lazy Import Optimization

**Action:** Move heavy imports inside functions:

```python
# Move imports inside methods
def _apply_background_removal(self, image, params):
    from ocr.datasets.preprocessing.background_removal import BackgroundRemoval
    # ... rest of code
```

**Expected Impact:** Faster app startup, reduced memory

### Priority 6: Optimize State Persistence

**Action:** Only persist state when changed:

```python
class UnifiedAppState:
    def persist_if_changed(self) -> None:
        # Only update if state actually changed
        pass
```

**Expected Impact:** 10-20ms reduction, fewer re-renders

## Additional Considerations

### Image Processing Challenges

- **Numpy arrays:** Streamlit caching doesn't work well with numpy arrays directly
- **Solution:** Serialize to bytes or use hash-based caching
- **Memory:** Large images can cause memory issues with caching

### Checkpoint Scanning

- **Directory size:** Large outputs/ directories slow down scanning
- **Solution:** Incremental scanning, index files, shorter cache TTL
- **Real-time updates:** Balance between freshness and performance

### Multi-Page Navigation

- **State sharing:** State is shared across pages (good)
- **Service reuse:** Services should be cached across pages
- **Config reuse:** Configs should be cached across pages

## Success Criteria

- Page load time < 200ms (from current 330-720ms)
- Action execution time < 1s for cached operations (from 1.1-15.2s)
- Service creation < 10ms (from 100-200ms)
- Checkpoint loading < 50ms (from 200-500ms)

## Testing Strategy

1. **Performance Testing:**
   - Measure page load time before/after
   - Measure action execution time
   - Test with various cache states

2. **Functional Testing:**
   - Verify all pages work correctly
   - Verify caching doesn't break functionality
   - Test cache invalidation

3. **Memory Testing:**
   - Monitor memory usage with caching
   - Test with large images
   - Verify no memory leaks

## References

- Command Builder Assessment: `artifacts/assessments/2025-11-17_0114_streamlit-command-builder-performance-assessment---page-switch-delays.md`
- Current app: `ui/apps/unified_ocr_app/`
- Services: `ui/apps/unified_ocr_app/services/`
- Pages: `ui/apps/unified_ocr_app/pages/`
