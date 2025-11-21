---
title: "Unified App Phase 1: Service and Checkpoint Caching"
author: "ai-agent"
timestamp: "2025-11-17 01:36 KST"
branch: "main"
type: "implementation_plan"
category: "development"
status: "draft"
tags: []
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Unified App Phase 1: Service and Checkpoint Caching**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Unified App Phase 1: Service and Checkpoint Caching

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Unified App Phase 1, Task 1.1 - Create Cached Service Factories
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Create cached service factory functions in services/__init__.py

### Implementation Outline (Checklist)

#### **Unified App Phase 1: Service and Checkpoint Caching (Week 1)**
1. [ ] **Task 1.1: Create Cached Service Factories**
   - [ ] Create or update `services/__init__.py`
   - [ ] Add `@st.cache_resource` to `get_preprocessing_service()`
   - [ ] Add `@st.cache_resource` to `get_inference_service()`
   - [ ] Test imports work correctly

2. [ ] **Task 1.2: Update Pages to Use Cached Services**
   - [ ] Update preprocessing page to use cached service
   - [ ] Update inference page to use cached service
   - [ ] Update batch inference to use cached service
   - [ ] Test all pages work correctly

3. [ ] **Task 1.3: Cache Checkpoint Loading**
   - [ ] Add `@st.cache_data` to `load_checkpoints()`
   - [ ] Set appropriate TTL (5 minutes)
   - [ ] Test checkpoint loading performance

4. [ ] **Task 1.4: Add Cache Invalidation Utilities**
   - [ ] Add cache clearing functions to shared_utils
   - [ ] Add optional cache clear button (development)
   - [ ] Test cache invalidation works

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Modular Design: Cached service factories in services module
- [ ] Streamlit Caching: Use `@st.cache_resource` for services, `@st.cache_data` for checkpoints
- [ ] Cache TTL: 5 minutes (300s) for checkpoints, persistent for services
- [ ] State Management: No changes to existing state management

### **Integration Points**
- [ ] Integration with Streamlit caching system
- [ ] Use existing PreprocessingService and InferenceService
- [ ] Maintain compatibility with existing pages

### **Quality Assurance**
- [ ] Functional Testing: All pages work correctly
- [ ] Performance Testing: Page load time < 200ms, service creation < 10ms
- [ ] Cache Testing: Verify caching works and invalidation works
- [ ] Regression Testing: No functionality broken

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] All pages (Preprocessing/Inference/Comparison) work correctly
- [ ] Service instances are cached and reused
- [ ] Checkpoint loading is cached
- [ ] No regressions in existing functionality

### **Technical Requirements**
- [ ] Page load time < 200ms (from 330-720ms)
- [ ] Service creation < 10ms (from 100-200ms)
- [ ] Checkpoint loading < 50ms (from 200-500ms)
- [ ] Code is type-hinted and documented
- [ ] Cache invalidation works correctly

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW
### **Active Mitigation Strategies**:
1. Incremental Development: Implement one service at a time, test after each
2. Comprehensive Testing: Test all pages and functionality after changes
3. Rollback Plan: Easy to revert if issues occur

### **Fallback Options**:
1. If caching causes issues: Revert to direct instantiation (simple rollback)
2. If memory usage too high: Reduce TTL or remove some caches
3. If cache invalidation issues: Use shorter TTL or manual cache clearing

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

**TASK:** Create cached service factory functions in services/__init__.py

**OBJECTIVE:** Create cached factory functions for PreprocessingService and InferenceService to avoid recreating service instances on every action.

**APPROACH:**
1. Create or update `ui/apps/unified_ocr_app/services/__init__.py`
2. Add `get_preprocessing_service()` with `@st.cache_resource`
3. Add `get_inference_service()` with `@st.cache_resource`
4. Test that imports work correctly

**SUCCESS CRITERIA:**
- Factory functions exist with proper decorators
- Imports work without errors
- Functions return correct types
- Services can be instantiated via factories

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
## Objective

Implement Streamlit caching for service instances and checkpoint loading in the Unified OCR App. This is the highest-impact optimization that will reduce page load time by 200-500ms and action execution time by 100-200ms.

## Context

The Unified OCR App creates new service instances on every action and loads checkpoints on every page render. These operations should be cached to improve performance.

**Current Performance:**
- Inference page load: 330-720ms (checkpoint loading: 200-500ms)
- Action execution: 1.1-15.2s (service creation: 100-200ms)

**Target Performance:**
- Page load time: < 200ms
- Service creation: < 10ms
- Checkpoint loading: < 50ms

## Approach

1. Add `@st.cache_resource` decorators to service factory functions
2. Add `@st.cache_data` to checkpoint loading
3. Ensure cache keys are stable and appropriate
4. Add cache invalidation strategy

## Implementation Steps

### Step 1: Create Cached Service Factories

**File:** `ui/apps/unified_ocr_app/services/__init__.py` (create or update)

Create cached factory functions:

```python
from __future__ import annotations

from typing import Any

import streamlit as st

from ui.apps.unified_ocr_app.services.inference_service import InferenceService
from ui.apps.unified_ocr_app.services.preprocessing_service import PreprocessingService

@st.cache_resource
def get_preprocessing_service(mode_config: dict[str, Any]) -> PreprocessingService:
    """Get cached PreprocessingService instance.

    Args:
        mode_config: Mode configuration dictionary

    Returns:
        Cached PreprocessingService instance
    """
    return PreprocessingService(mode_config)

@st.cache_resource
def get_inference_service(mode_config: dict[str, Any]) -> InferenceService:
    """Get cached InferenceService instance.

    Args:
        mode_config: Mode configuration dictionary

    Returns:
        Cached InferenceService instance
    """
    return InferenceService(mode_config)
```

**Note:** The mode_config dict must be hashable for caching to work. Streamlit will automatically hash dict arguments.

### Step 2: Update Pages to Use Cached Services

**File:** `ui/apps/unified_ocr_app/pages/1_🎨_Preprocessing.py`

Replace direct service instantiation:

```python
# BEFORE (line 110):
service = PreprocessingService(mode_config)

# AFTER:
from ui.apps.unified_ocr_app.services import get_preprocessing_service

service = get_preprocessing_service(mode_config)
```

**File:** `ui/apps/unified_ocr_app/pages/2_🤖_Inference.py`

Replace direct service instantiation:

```python
# BEFORE (line 141):
service = InferenceService(mode_config)

# AFTER:
from ui.apps.unified_ocr_app.services import get_inference_service

service = get_inference_service(mode_config)
```

Also update batch inference (line 229):

```python
# BEFORE:
service = InferenceService(mode_config)

# AFTER:
service = get_inference_service(mode_config)
```

### Step 3: Cache Checkpoint Loading

**File:** `ui/apps/unified_ocr_app/services/inference_service.py`

Add caching to `load_checkpoints()` function:

```python
@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
def load_checkpoints(config: dict[str, Any]) -> list[Any]:
    """Load available model checkpoints with caching.

    Args:
        config: App configuration with paths

    Returns:
        List of CheckpointInfo objects

    Note:
        Cached for 5 minutes to balance freshness and performance.
        Clear cache if checkpoints are added/removed.
    """
    # ... existing implementation (lines 226-267)
```

**Important:** The config dict must be stable for caching to work. If config changes frequently, consider using a hash of relevant paths.

### Step 4: Add Cache Invalidation Utilities

**File:** `ui/apps/unified_ocr_app/shared_utils.py`

Add cache clearing utilities:

```python
def clear_service_cache() -> None:
    """Clear all cached service instances."""
    st.cache_resource.clear()

def clear_checkpoint_cache() -> None:
    """Clear checkpoint loading cache."""
    # Clear specific cache by calling with different args or clearing all
    st.cache_data.clear()

def clear_all_caches() -> None:
    """Clear all caches (services and data)."""
    st.cache_resource.clear()
    st.cache_data.clear()
```

**File:** `ui/apps/unified_ocr_app/app.py` (optional, for development)

Add cache clear button in sidebar:

```python
# In main() function, after state initialization
if st.sidebar.button("🔄 Clear Cache", help="Clear all cached data (development only)"):
    from ui.apps.unified_ocr_app.shared_utils import clear_all_caches
    clear_all_caches()
    st.rerun()
```

## Testing Strategy

### Functional Testing
1. **Verify services work:**
   - Run preprocessing pipeline → verify it works
   - Run inference → verify it works
   - Run batch inference → verify it works
   - Switch between pages → verify services still work

2. **Verify caching works:**
   - First action: Should see normal execution time
   - Second action: Should be faster (service reused)
   - Check browser dev tools for cache hits

3. **Verify checkpoint caching:**
   - First page load: Should see normal load time
   - Second page load: Should be faster
   - Add new checkpoint: Should appear after cache expires (5 min) or manual clear

### Performance Testing
1. **Measure service creation time:**
   - Before: 100-200ms
   - After: < 10ms (cached)

2. **Measure checkpoint loading time:**
   - Before: 200-500ms
   - After: < 50ms (cached)

3. **Measure page load time:**
   - Before: 330-720ms
   - After: < 200ms

### Edge Cases
1. **Config changes:**
   - Modify mode_config
   - Verify new service instance is created (different cache key)
   - Verify old instance is not reused incorrectly

2. **Checkpoint changes:**
   - Add new checkpoint
   - Verify appears after cache TTL expires
   - Or manually clear cache to see immediately

3. **Memory usage:**
   - Monitor memory with cached services
   - Verify no memory leaks
   - Test with multiple mode configs

## Success Criteria

- ✅ All services work correctly with caching
- ✅ Service creation time < 10ms (from 100-200ms)
- ✅ Checkpoint loading time < 50ms (from 200-500ms)
- ✅ Page load time < 200ms (from 330-720ms)
- ✅ No regressions in functionality
- ✅ Cache invalidation works correctly

## Rollback Plan

If issues occur:
1. Revert service factory changes
2. Revert page changes (use direct instantiation)
3. Remove caching from load_checkpoints
4. Verify app works without caching

## Additional Notes

- **Cache Keys:** Streamlit automatically creates cache keys from function arguments. Ensure mode_config is stable.
- **TTL Selection:** 5 minutes for checkpoints balances freshness and performance. Adjust based on usage.
- **Memory:** Caching increases memory usage. Monitor and adjust if needed.
- **Development:** Consider shorter TTL or cache clearing button for development.

## References

- Assessment: `artifacts/assessments/2025-11-17_0136_unified-ocr-app-performance-assessment.md`
- Current services: `ui/apps/unified_ocr_app/services/`
- Current pages: `ui/apps/unified_ocr_app/pages/`
