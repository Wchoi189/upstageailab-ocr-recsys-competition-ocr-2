---
title: "Unified App Phase 2: Image Processing And Config Optimization"
date: "2025-12-06 18:09 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---







# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Unified App Phase 2: Image Processing and Config Optimization**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Unified App Phase 2: Image Processing and Config Optimization

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Unified App Phase 2, Task 2.1 - Optimize Image Processing Caching
- **LAST COMPLETED TASK:** None (Requires Unified App Phase 1 completion)
- **NEXT TASK:** Optimize image processing caching with hash-based approach

### Implementation Outline (Checklist)

#### **Unified App Phase 2: Image Processing and Config Optimization (Week 2)**
1. [ ] **Task 2.1: Optimize Image Processing Caching**
   - [ ] Implement hash-based caching for image processing
   - [ ] Update preprocessing service to use hash-based cache
   - [ ] Update preprocessing page to generate image hashes
   - [ ] Test image processing caching works correctly

2. [ ] **Task 2.2: Optimize Config Loading**
   - [ ] Add module-level cache for configs in shared_utils
   - [ ] Update pages to use cached config loader
   - [ ] Test config loading performance

3. [ ] **Task 2.3: Optimize State Persistence**
   - [ ] Add `persist_if_changed()` method to UnifiedAppState
   - [ ] Update pages to use optimized persistence
   - [ ] Test state persistence works correctly

4. [ ] **Task 2.4: Add Loading Indicators**
   - [ ] Add spinners for config loading
   - [ ] Add spinners for checkpoint loading
   - [ ] Test loading indicators appear correctly

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Hash-Based Caching: Use image hash for cache keys (numpy arrays challenging)
- [ ] Module-Level Config Cache: Shared cache for configs across pages
- [ ] State Persistence: Only persist when changed
- [ ] Loading Indicators: Show progress for heavy operations

### **Integration Points**
- [ ] Integration with Phase 1 cached services
- [ ] Use existing PreprocessingService and state management
- [ ] Maintain compatibility with existing pages

### **Quality Assurance**
- [ ] Functional Testing: All pages work correctly
- [ ] Performance Testing: Image processing < 1s cached, config loading < 10ms
- [ ] Cache Testing: Verify image processing caching works
- [ ] Regression Testing: No functionality broken

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] All pages work correctly
- [ ] Image processing results are cached correctly
- [ ] Config loading is optimized
- [ ] State only persists when changed

### **Technical Requirements**
- [ ] Image processing cached operations < 1s (from 1-5s)
- [ ] Config loading < 10ms (from 50-100ms)
- [ ] State persistence optimized (only updates when changed)
- [ ] Code is type-hinted and documented

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW
### **Active Mitigation Strategies**:
1. Incremental Development: Implement one optimization at a time
2. Comprehensive Testing: Test each optimization independently
3. Dependency on Phase 1: Requires Unified App Phase 1 completion

### **Fallback Options**:
1. If image caching causes issues: Revert to no caching or use simpler approach
2. If config caching breaks: Revert to Streamlit cache only
3. If state persistence issues: Revert to always persisting

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
- Task completion (move to next task)
- Blocker encountered (document and propose solution)
- Technical discovery (update approach if needed)
- Quality gate failure (address issues before proceeding)

**Update Format:**
1. Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2. Mark completed items with [x]
3. Add any new discoveries or changes to approach
4. Update risk assessment if needed

---

## 🚀 **Immediate Next Action**

**TASK:** Optimize image processing caching with hash-based approach

**OBJECTIVE:** Implement hash-based caching for image processing to avoid repeated processing of the same image with the same parameters, since numpy arrays are challenging to cache directly.

**APPROACH:**
1. Update `process_image()` in preprocessing_service.py to use hash-based caching
2. Generate image hash in preprocessing page before calling service
3. Use hash as cache key instead of numpy array
4. Test that caching works correctly for repeated operations

**SUCCESS CRITERIA:**
- Image processing uses hash-based caching
- Cached image processing < 1s (from 1-5s)
- Same image with same params processes instantly on second run
- All functionality works correctly

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
## Objective

Optimize image processing caching and config loading in the Unified OCR App. This phase focuses on improving action execution time by properly caching image processing results and optimizing configuration loading.

## Context

After Phase 1, services and checkpoints are cached. Phase 2 addresses:
1. Image processing results caching (numpy arrays are challenging)
2. Config loading optimization
3. State persistence optimization

**Current State (after Phase 1):**
- Services are cached
- Checkpoints are cached
- Image processing: 1-5 seconds (not cached)
- Config loading: 50-100ms (cached but could be better)

**Target State:**
- Image processing: < 1s for cached operations
- Config loading: < 10ms
- State persistence: Only updates when changed

## Approach

1. Implement proper image processing caching with serialization
2. Optimize config loading with module-level cache
3. Optimize state persistence
4. Add loading indicators for heavy operations

## Implementation Steps

### Step 1: Optimize Image Processing Caching

**File:** `ui/apps/unified_ocr_app/services/preprocessing_service.py`

The current `@st.cache_data` on `process_image()` may not work well with numpy arrays. Improve it:

```python
import hashlib
import pickle

@st.cache_data(show_spinner=False, ttl=3600)
def process_image(
    _self,
    image_bytes: bytes,  # Changed: Accept bytes instead of numpy array
    parameters: dict[str, Any],
    _image_hash: str,
) -> dict[str, Any]:
    """Process image through preprocessing pipeline with improved caching.

    Args:
        image_bytes: Serialized image (numpy array as bytes)
        parameters: Preprocessing parameters
        _image_hash: Hash for cache busting

    Returns:
        Dict with 'stages' and 'metadata'
    """
    # Deserialize image
    import numpy as np
    image = pickle.loads(image_bytes)

    # ... rest of existing processing logic ...
```

**File:** `ui/apps/unified_ocr_app/pages/1_🎨_Preprocessing.py`

Update to serialize image before calling:

```python
# Around line 118
import pickle

# Serialize image to bytes for caching
image_bytes = pickle.dumps(current_image)

# Process
result = service.process_image(image_bytes, current_params, cache_key)
```

**Alternative Approach (Simpler):** Keep numpy array but use hash-based caching:

```python
# In preprocessing_service.py
@st.cache_data(show_spinner=False, ttl=3600)
def process_image(
    _self,
    image_hash: str,  # Use hash instead of array
    parameters: dict[str, Any],
    mode_config_hash: str,
) -> dict[str, Any]:
    """Process image with hash-based caching."""
    # Image should be passed separately and not cached
    # This is a wrapper that caches based on hash
    pass
```

**Recommendation:** Use the hash-based approach as it's simpler and avoids serialization overhead.

### Step 2: Optimize Config Loading

**File:** `ui/apps/unified_ocr_app/shared_utils.py`

Add module-level cache for configs:

```python
# At top of file
_CONFIG_CACHE: dict[str, dict[str, Any]] = {}

def get_mode_config_cached(mode_id: str) -> dict[str, Any]:
    """Get mode config with module-level caching.

    Args:
        mode_id: Mode identifier

    Returns:
        Mode configuration dictionary
    """
    if mode_id not in _CONFIG_CACHE:
        _CONFIG_CACHE[mode_id] = load_mode_config(mode_id, validate=False)
    return _CONFIG_CACHE[mode_id]
```

**File:** `ui/apps/unified_ocr_app/pages/1_🎨_Preprocessing.py` and `2_🤖_Inference.py`

Update to use cached config loader:

```python
# BEFORE (line 48):
mode_config = load_mode_config("preprocessing", validate=False)

# AFTER:
from ui.apps.unified_ocr_app.shared_utils import get_mode_config_cached

mode_config = get_mode_config_cached("preprocessing")
```

### Step 3: Optimize State Persistence

**File:** `ui/apps/unified_ocr_app/models/app_state.py`

Add method to only persist when changed:

```python
@dataclass
class UnifiedAppState:
    # ... existing fields ...

    def to_session(self) -> None:
        """Save state to Streamlit session_state."""
        st.session_state.unified_app_state = self

    def persist_if_changed(self) -> None:
        """Only persist state if it actually changed."""
        import hashlib
        import pickle

        current_state = st.session_state.get("unified_app_state")
        if current_state != self:
            # State changed, persist it
            self.to_session()
```

**File:** `ui/apps/unified_ocr_app/pages/1_🎨_Preprocessing.py` and `2_🤖_Inference.py`

Update to use optimized persistence:

```python
# BEFORE (line 123):
state.to_session()

# AFTER:
state.persist_if_changed()  # Only updates if changed
```

### Step 4: Add Loading Indicators

**File:** `ui/apps/unified_ocr_app/pages/1_🎨_Preprocessing.py`

Add spinner for config loading:

```python
# Around line 47
with st.spinner("Loading configuration..."):
    mode_config = get_mode_config_cached("preprocessing")
```

**File:** `ui/apps/unified_ocr_app/pages/2_🤖_Inference.py`

Add spinner for checkpoint loading:

```python
# Around line 60
with st.spinner("Loading checkpoints..."):
    checkpoints = load_checkpoints(app_config)
```

## Testing Strategy

### Functional Testing
1. **Verify image processing caching:**
   - Process same image with same params twice
   - Second time should be much faster
   - Verify results are identical

2. **Verify config caching:**
   - Load same page multiple times
   - Config should load instantly after first load
   - Verify config changes are reflected (after cache clear)

3. **Verify state persistence:**
   - Make changes to state
   - Switch pages
   - Return to original page
   - Verify changes persisted

### Performance Testing
1. **Measure image processing time:**
   - First run: 1-5 seconds
   - Cached run: < 1 second

2. **Measure config loading time:**
   - First load: 50-100ms
   - Cached load: < 10ms

3. **Measure state persistence:**
   - Should only persist when changed
   - Verify no unnecessary updates

### Edge Cases
1. **Image processing:**
   - Test with different images
   - Test with different parameters
   - Verify cache keys work correctly

2. **Config changes:**
   - Modify config file
   - Verify cache updates
   - Or manually clear cache

3. **Memory usage:**
   - Monitor memory with image caching
   - Test with large images
   - Verify no memory leaks

## Success Criteria

- ✅ Image processing cached correctly
- ✅ Cached image processing < 1s (from 1-5s)
- ✅ Config loading < 10ms (from 50-100ms)
- ✅ State only persists when changed
- ✅ No regressions in functionality

## Dependencies

- **Requires:** Phase 1 (service caching) must be completed first
- **Blocks:** Phase 3 (advanced optimizations)

## Rollback Plan

If issues occur:
1. Revert image processing caching changes
2. Revert config caching changes
3. Revert state persistence changes
4. Verify app works correctly

## Additional Notes

- **Image Caching:** Numpy arrays are challenging to cache. Hash-based approach is recommended.
- **Config Caching:** Module-level cache is faster than Streamlit cache for frequently accessed configs.
- **State Persistence:** Change detection adds complexity but improves performance.

## References

- Phase 1 Plan: `artifacts/implementation_plans/2025-11-17_XXXX_phase-1:-service-and-checkpoint-caching.md`
- Assessment: `artifacts/assessments/2025-11-17_0136_unified-ocr-app-performance-assessment.md`
