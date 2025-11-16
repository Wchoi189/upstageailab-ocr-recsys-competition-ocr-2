---
title: "Phase 3: Advanced Optimizations"
author: "ai-agent"
timestamp: "2025-11-17 01:30 KST"
branch: "main"
type: "implementation_plan"
category: "development"
status: "draft"
tags: []
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Phase 3: Advanced Optimizations**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Phase 3: Advanced Optimizations

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Phase 3, Task 3.1 - Module-Level Caching for ConfigParser
- **LAST COMPLETED TASK:** None (Requires Phase 1 and Phase 2 completion)
- **NEXT TASK:** Implement module-level caching for ConfigParser

### Implementation Outline (Checklist)

#### **Phase 3: Advanced Optimizations (Week 3)**
1. [ ] **Task 3.1: Module-Level Caching for ConfigParser**
   - [ ] Add module-level cache dictionary
   - [ ] Update `get_available_models()` with module cache
   - [ ] Update other expensive methods similarly
   - [ ] Test cache sharing across instances

2. [ ] **Task 3.2: Optimize Checkpoint Scanning**
   - [ ] Add timestamp-based cache for checkpoints
   - [ ] Implement incremental scanning
   - [ ] Add cache TTL (5 minutes)
   - [ ] Test checkpoint loading performance

3. [ ] **Task 3.3: Background Prefetching**
   - [ ] Add prefetching for likely-needed services
   - [ ] Test prefetching doesn't impact current page
   - [ ] Verify subsequent page loads are instant

4. [ ] **Task 3.4: Additional Micro-Optimizations**
   - [ ] Optimize YAML schema loading
   - [ ] Add performance monitoring (optional)
   - [ ] Test all optimizations together

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Module-Level Caching: Shared cache across ConfigParser instances
- [ ] Incremental Scanning: Optimized checkpoint directory scanning
- [ ] Background Prefetching: Pre-load likely-needed data
- [ ] Performance Monitoring: Optional metrics for development

### **Integration Points**
- [ ] Integration with Phase 1 and Phase 2 optimizations
- [ ] Use existing ConfigParser class
- [ ] Maintain compatibility with existing code

### **Quality Assurance**
- [ ] Functional Testing: All features work correctly
- [ ] Performance Testing: Page switch time < 100ms
- [ ] Memory Testing: Monitor memory usage
- [ ] Regression Testing: No functionality broken

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] All pages load and function correctly
- [ ] Module-level caching works across instances
- [ ] Checkpoint scanning is optimized
- [ ] Prefetching improves perceived performance

### **Technical Requirements**
- [ ] Page switch time < 100ms (from Phase 2's < 150ms)
- [ ] Checkpoint scanning < 50ms (from 200-500ms)
- [ ] Memory usage acceptable (no leaks)
- [ ] Code is type-hinted and documented

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW
### **Active Mitigation Strategies**:
1. Incremental Development: Implement one optimization at a time
2. Comprehensive Testing: Test each optimization independently
3. Dependency Management: Requires Phase 1 and Phase 2 completion

### **Fallback Options**:
1. If module-level cache causes issues: Revert to instance-level cache
2. If checkpoint optimization fails: Revert to full directory scan
3. If prefetching causes problems: Disable prefetching

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

**TASK:** Implement module-level caching for ConfigParser

**OBJECTIVE:** Add module-level cache shared across all ConfigParser instances to reduce redundant file I/O and computation.

**APPROACH:**
1. Add `_MODULE_CACHE` dictionary at module level in `config_parser.py`
2. Update `get_available_models()` to check module cache first
3. Apply same pattern to other expensive methods
4. Test cache sharing works across instances

**SUCCESS CRITERIA:**
- Module-level cache implemented
- Cache shared across ConfigParser instances
- No redundant file I/O operations
- All functionality works correctly

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
## Objective

Implement advanced optimizations including module-level caching, background data loading, and checkpoint scanning optimization. This phase focuses on micro-optimizations and edge cases to achieve the final performance targets.

## Context

After Phases 1 and 2, the app should be significantly faster. Phase 3 addresses:
1. Module-level caching for ConfigParser (shared across instances)
2. Optimized checkpoint scanning (avoid full directory scans)
3. Background prefetching of likely-needed data
4. Additional micro-optimizations

**Current State (after Phase 2):**
- Services are cached and lazy-loaded
- UI renders progressively
- Page switch time: < 150ms

**Target State:**
- Module-level caching reduces redundant work
- Checkpoint scanning is optimized
- Data prefetching improves perceived performance
- Page switch time: < 100ms

## Approach

1. Implement module-level caching for ConfigParser
2. Optimize checkpoint scanning with incremental updates
3. Add background prefetching for likely-needed data
4. Implement additional micro-optimizations

## Implementation Steps

### Step 1: Module-Level Caching for ConfigParser

**File:** `ui/utils/config_parser.py`

Add module-level cache shared across all instances:

```python
# At top of file, after imports
_MODULE_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()  # For thread safety if needed

class ConfigParser:
    """Parser for extracting configuration options from Hydra configs."""

    def __init__(self, config_dir: str | None = None):
        # ... existing __init__ code ...
        # Keep instance cache for backward compatibility
        self._cache: dict[str, Any] = {}

    def get_available_models(self) -> dict[str, list[str]]:
        """Get available model components with module-level caching."""
        cache_key = "models"

        # Check module-level cache first
        if cache_key in _MODULE_CACHE:
            return _MODULE_CACHE[cache_key]

        # Check instance cache
        if cache_key in self._cache:
            # Also store in module cache for future instances
            _MODULE_CACHE[cache_key] = self._cache[cache_key]
            return self._cache[cache_key]

        # Compute (existing logic)
        models: dict[str, list[str]] = {
            "encoders": [],
            "decoders": [],
            # ... rest of existing code ...
        }

        # Store in both caches
        self._cache[cache_key] = models
        _MODULE_CACHE[cache_key] = models
        return models

    # Apply same pattern to other expensive methods:
    # - get_architecture_metadata()
    # - get_optimizer_metadata()
    # - get_preprocessing_profiles()
    # - get_available_datasets()
```

### Step 2: Optimize Checkpoint Scanning

**File:** `ui/utils/config_parser.py`

Optimize `get_available_checkpoints()` to avoid full directory scans:

```python
import os
from pathlib import Path
from typing import Any

# Module-level cache for checkpoints with timestamp
_CHECKPOINT_CACHE: dict[str, tuple[list[str], float]] = {}
_CHECKPOINT_CACHE_TTL = 300  # 5 minutes

def get_available_checkpoints(self) -> list[str]:
    """Get available checkpoint files with optimized scanning."""
    import time

    cache_key = "checkpoints"
    current_time = time.time()

    # Check if cache is still valid
    if cache_key in _CHECKPOINT_CACHE:
        cached_checkpoints, cache_time = _CHECKPOINT_CACHE[cache_key]
        if current_time - cache_time < _CHECKPOINT_CACHE_TTL:
            return cached_checkpoints

    # Only scan if cache expired or missing
    checkpoints = []
    outputs_dir = self.config_dir.parent / "outputs"

    if outputs_dir.exists():
        # Use os.walk for better performance than glob
        for root, dirs, files in os.walk(outputs_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            if 'checkpoints' in root:
                for file in files:
                    if file.endswith('.ckpt'):
                        rel_path = os.path.relpath(
                            os.path.join(root, file),
                            self.config_dir.parent
                        )
                        checkpoints.append(rel_path)

    # Update cache
    _CHECKPOINT_CACHE[cache_key] = (checkpoints, current_time)
    return checkpoints
```

### Step 3: Background Prefetching

**File:** `ui/apps/command_builder/app.py`

Add background prefetching for likely-needed data:

```python
def run() -> None:
    st.set_page_config(page_title="OCR Command Builder", page_icon="🔍", layout="wide")
    st.title("🔍 OCR Training Command Builder")
    st.caption("Build and execute training, testing, and prediction commands with metadata-aware defaults.")

    state = CommandBuilderState.from_session()
    command_type = render_sidebar(state)

    # Prefetch data for other pages in background (non-blocking)
    if command_type == CommandType.TRAIN:
        # Load Train page (current)
        # ... existing code ...

        # Prefetch Test and Predict services in background
        # (Streamlit will cache them, so they're ready when needed)
        if "prefetch_done" not in st.session_state:
            from ui.apps.command_builder.utils import get_command_builder
            # Trigger cache population (non-blocking due to caching)
            _ = get_command_builder()  # Already cached, instant
            st.session_state["prefetch_done"] = True
```

### Step 4: Optimize YAML Schema Loading

**File:** `ui/utils/ui_generator.py`

Ensure schema loading is properly cached and optimized:

```python
@st.cache_data(show_spinner=False, ttl=3600, persist="disk")
def _load_schema(schema_path: str) -> dict[str, Any]:
    """Load schema with disk persistence for faster reloads."""
    with open(schema_path) as f:
        return yaml.safe_load(f) or {}
```

**Note:** `persist="disk"` option may not be available in all Streamlit versions. Check version compatibility.

### Step 5: Additional Micro-Optimizations

**File:** `ui/apps/command_builder/components/training.py`

Optimize recommendation loading:

```python
# Cache recommendations per architecture
@st.cache_data(ttl=3600)
def _get_recommendations_for_arch(architecture: str | None) -> list[UseCaseRecommendation]:
    """Cached wrapper for recommendations."""
    from ui.apps.command_builder.utils import get_recommendation_service
    service = get_recommendation_service()
    return service.for_architecture(architecture)

# In render_training_page:
current_architecture = st.session_state.get(f"{SCHEMA_PREFIX}__architecture") or page.generated.values.get("architecture")
recommendations = _get_recommendations_for_arch(current_architecture)
```

**File:** `ui/apps/command_builder/components/suggestions.py`

Optimize recommendation rendering:

```python
# Only render if recommendations exist and are non-empty
def render_use_case_recommendations(...):
    if not recommendations:
        return  # Early return, skip rendering

    # ... existing rendering code ...
```

### Step 6: Add Performance Monitoring (Optional)

**File:** `ui/apps/command_builder/app.py`

Add optional performance monitoring:

```python
import time

def run() -> None:
    if st.session_state.get("enable_perf_monitoring", False):
        start_time = time.time()

    # ... existing code ...

    if st.session_state.get("enable_perf_monitoring", False):
        elapsed = time.time() - start_time
        st.sidebar.metric("Page Render Time", f"{elapsed*1000:.0f}ms")
```

## Testing Strategy

### Functional Testing
1. **Verify module-level caching:**
   - Create multiple ConfigParser instances
   - Verify they share cache
   - Verify cache persists across instances

2. **Verify checkpoint optimization:**
   - Check checkpoint list loads quickly
   - Verify cache TTL works correctly
   - Verify new checkpoints appear after cache expires

3. **Verify prefetching:**
   - Switch pages rapidly
   - Verify subsequent page loads are instant
   - Verify no errors from prefetching

### Performance Testing
1. **Measure page switch time:**
   - Should be < 100ms after all optimizations
   - UI should appear within 50ms

2. **Measure memory usage:**
   - Monitor memory with module-level caching
   - Verify no memory leaks
   - Verify cache size is reasonable

3. **Profile with cProfile:**
   - Identify any remaining bottlenecks
   - Verify optimizations are effective

### Edge Cases
1. **Cache invalidation:**
   - Modify config files
   - Verify cache updates correctly
   - Verify TTL works as expected

2. **Concurrent access:**
   - Multiple users (if applicable)
   - Verify thread safety
   - Verify no race conditions

## Success Criteria

- ✅ Module-level caching implemented and working
- ✅ Checkpoint scanning optimized (< 50ms)
- ✅ Page switch time < 100ms
- ✅ UI appears within 50ms
- ✅ No memory leaks
- ✅ All functionality preserved

## Dependencies

- **Requires:** Phase 1 and Phase 2 must be completed first
- **Optional:** Performance monitoring can be added independently

## Rollback Plan

If issues occur:
1. Remove module-level caching (revert to instance-level)
2. Revert checkpoint optimization
3. Remove prefetching
4. Verify app works correctly

## Additional Notes

- **Module-Level Cache:** Shared cache improves performance but requires careful management
- **Checkpoint TTL:** 5 minutes is a good default, adjust based on usage patterns
- **Prefetching:** Only prefetch if it doesn't impact current page performance
- **Performance Monitoring:** Useful for development, disable in production

## References

- Phase 1 Plan: `artifacts/implementation_plans/2025-11-17_0126_phase-1:-implement-streamlit-caching-for-command-builder.md`
- Phase 2 Plan: `artifacts/implementation_plans/2025-11-17_XXXX_phase-2:-lazy-loading-and-progressive-rendering.md`
- Assessment: `artifacts/assessments/2025-11-17_0114_streamlit-command-builder-performance-assessment---page-switch-delays.md`

