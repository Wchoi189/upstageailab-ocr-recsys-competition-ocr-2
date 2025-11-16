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

