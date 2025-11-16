---
title: "Unified App Phase 2: Image Processing and Config Optimization"
author: "ai-agent"
timestamp: "2025-11-17 01:36 KST"
branch: "main"
type: "implementation_plan"
category: "development"
status: "draft"
tags: []
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

- **STATUS:** Not Started / In Progress / Completed
- **CURRENT STEP:** [Current Phase, Task # - Task Name]
- **LAST COMPLETED TASK:** [Description of last completed task]
- **NEXT TASK:** [Description of the immediate next task]

### Implementation Outline (Checklist)

#### **Phase 1: [Phase 1 Title] (Week [Number])**
1. [ ] **Task 1.1: [Task 1.1 Title]**
   - [ ] [Sub-task 1.1.1 description]
   - [ ] [Sub-task 1.1.2 description]
   - [ ] [Sub-task 1.1.3 description]

2. [ ] **Task 1.2: [Task 1.2 Title]**
   - [ ] [Sub-task 1.2.1 description]
   - [ ] [Sub-task 1.2.2 description]

#### **Phase 2: [Phase 2 Title] (Week [Number])**
3. [ ] **Task 2.1: [Task 2.1 Title]**
   - [ ] [Sub-task 2.1.1 description]
   - [ ] [Sub-task 2.1.2 description]

4. [ ] **Task 2.2: [Task 2.2 Title]**
   - [ ] [Sub-task 2.2.1 description]
   - [ ] [Sub-task 2.2.2 description]

*(Add more Phases and Tasks as needed)*

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] [Architectural Principle 1 (e.g., Modular Design)]
- [ ] [Data Model Requirement (e.g., Pydantic V2 Integration)]
- [ ] [Configuration Method (e.g., YAML-Driven)]
- [ ] [State Management Strategy]

### **Integration Points**
- [ ] [Integration with System X]
- [ ] [API Endpoint Definition]
- [ ] [Use of Existing Utility/Library]

### **Quality Assurance**
- [ ] [Unit Test Coverage Goal (e.g., > 90%)]
- [ ] [Integration Test Requirement]
- [ ] [Performance Test Requirement]
- [ ] [UI/UX Test Requirement]

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] [Key Feature 1 Works as Expected]
- [ ] [Key Feature 2 is Fully Implemented]
- [ ] [Performance Metric is Met (e.g., <X ms latency)]
- [ ] [User-Facing Outcome is Achieved]

### **Technical Requirements**
- [ ] [Code Quality Standard is Met (e.g., Documented, type-hinted)]
- [ ] [Resource Usage is Within Limits (e.g., <X GB memory)]
- [ ] [Compatibility with System Y is Confirmed]
- [ ] [Maintainability Goal is Met]

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW / MEDIUM / HIGH
### **Active Mitigation Strategies**:
1. [Mitigation Strategy 1 (e.g., Incremental Development)]
2. [Mitigation Strategy 2 (e.g., Comprehensive Testing)]
3. [Mitigation Strategy 3 (e.g., Regular Code Quality Checks)]

### **Fallback Options**:
1. [Fallback Option 1 if Risk A occurs (e.g., Simplified version of a feature)]
2. [Fallback Option 2 if Risk B occurs (e.g., CPU-only mode)]
3. [Fallback Option 3 if Risk C occurs (e.g., Phased Rollout)]

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

**TASK:** [Description of the immediate next task]

**OBJECTIVE:** [Clear, concise goal of the task]

**APPROACH:**
1. [Step 1 to execute the task]
2. [Step 2 to execute the task]
3. [Step 3 to execute the task]

**SUCCESS CRITERIA:**
- [Measurable outcome 1 that defines task completion]
- [Measurable outcome 2 that defines task completion]

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

