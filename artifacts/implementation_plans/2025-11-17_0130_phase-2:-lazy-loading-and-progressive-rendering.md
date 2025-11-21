---
title: "Phase 2: Lazy Loading and Progressive Rendering"
author: "ai-agent"
timestamp: "2025-11-17 01:30 KST"
branch: "main"
type: "implementation_plan"
category: "development"
status: "draft"
tags: []
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Phase 2: Lazy Loading and Progressive Rendering**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Phase 2: Lazy Loading and Progressive Rendering

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Phase 2, Task 2.1 - Refactor app.py for Lazy Loading
- **LAST COMPLETED TASK:** None (Requires Phase 1 completion)
- **NEXT TASK:** Refactor app.py to lazy-load services based on active page

### Implementation Outline (Checklist)

#### **Phase 2: Lazy Loading and Progressive Rendering (Week 2)**
1. [ ] **Task 2.1: Refactor app.py for Lazy Loading**
   - [ ] Move service initialization after sidebar render
   - [ ] Add conditional imports based on active page
   - [ ] Test all pages still work correctly

2. [ ] **Task 2.2: Implement Progressive Rendering**
   - [ ] Update training page to render UI first
   - [ ] Add loading spinners for heavy operations
   - [ ] Update test and predict pages similarly

3. [ ] **Task 2.3: Optimize State Persistence**
   - [ ] Add `persist_if_changed()` method to state
   - [ ] Update all pages to use optimized persistence
   - [ ] Test state persistence works correctly

4. [ ] **Task 2.4: Add Loading Indicators**
   - [ ] Add spinners for config loading
   - [ ] Add spinners for schema generation
   - [ ] Test loading indicators appear correctly

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Lazy Loading: Services only loaded for active page
- [ ] Progressive Rendering: UI structure first, then data
- [ ] State Management: Only persist when changed
- [ ] Conditional Imports: Import services only when needed

### **Integration Points**
- [ ] Integration with Phase 1 cached services
- [ ] Use existing page components (training, test, predict)
- [ ] Maintain compatibility with existing state management

### **Quality Assurance**
- [ ] Functional Testing: All pages work correctly
- [ ] Performance Testing: Page switch time < 150ms
- [ ] UI/UX Testing: Loading indicators work correctly
- [ ] Regression Testing: No functionality broken

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] All pages (Train/Test/Predict) load and function correctly
- [ ] Services only loaded for active page
- [ ] UI structure appears within 100ms
- [ ] Loading indicators show for heavy operations

### **Technical Requirements**
- [ ] Page switch time < 150ms (from Phase 1's < 200ms)
- [ ] State only persists when changed
- [ ] Code is type-hinted and documented
- [ ] No regressions in functionality

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW
### **Active Mitigation Strategies**:
1. Incremental Development: Implement one page at a time, test after each
2. Comprehensive Testing: Test all pages and functionality after changes
3. Dependency on Phase 1: Requires Phase 1 completion first

### **Fallback Options**:
1. If lazy loading causes issues: Revert to loading all services upfront
2. If progressive rendering breaks: Revert to synchronous rendering
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

**TASK:** Refactor app.py to lazy-load services based on active page

**OBJECTIVE:** Restructure app.py so that services are only initialized for the active page, reducing initialization overhead for unused pages.

**APPROACH:**
1. Move `render_sidebar()` call before service initialization
2. Add conditional service loading based on `command_type`
3. Move imports inside conditional blocks
4. Test all pages work correctly

**SUCCESS CRITERIA:**
- Services only loaded for active page
- All pages (Train/Test/Predict) work correctly
- Page switch time improved (50-150ms reduction)
- No regressions in functionality

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
## Objective

Implement lazy loading and progressive rendering for the Command Builder app. This phase builds on Phase 1 (caching) to further optimize page switching by only loading services needed for the active page and rendering UI structure before heavy data operations.

## Context

After Phase 1 caching is implemented, we can optimize further by:
1. Only initializing services needed for the active page
2. Rendering UI structure immediately, then loading data
3. Optimizing state persistence to avoid unnecessary updates

**Current State (after Phase 1):**
- Services are cached but still initialized for all pages
- UI waits for all data before rendering
- State is persisted on every render

**Target State:**
- Only active page's services are initialized
- UI structure appears immediately
- State only persists when changed

## Approach

1. Refactor app.py to lazy-load services based on active page
2. Implement progressive rendering in page components
3. Optimize state persistence to only update when changed
4. Add loading indicators for heavy operations

## Implementation Steps

### Step 1: Refactor app.py for Lazy Loading

**File:** `ui/apps/command_builder/app.py`

Restructure to initialize services only when needed:

```python
def run() -> None:
    st.set_page_config(page_title="OCR Command Builder", page_icon="🔍", layout="wide")
    st.title("🔍 OCR Training Command Builder")
    st.caption("Build and execute training, testing, and prediction commands with metadata-aware defaults.")

    state = CommandBuilderState.from_session()

    # Render sidebar FIRST (fast, no heavy operations)
    command_type = render_sidebar(state)

    # Lazy load services only for the active page
    if command_type == CommandType.TRAIN:
        from ui.apps.command_builder.utils import (
            get_command_builder,
            get_config_parser,
            get_recommendation_service
        )
        command_builder = get_command_builder()
        config_parser = get_config_parser()
        recommendation_service = get_recommendation_service()
        render_training_page(state, command_builder, recommendation_service, config_parser)
    elif command_type == CommandType.TEST:
        from ui.apps.command_builder.utils import get_command_builder
        command_builder = get_command_builder()
        render_test_page(state, command_builder)
    else:  # PREDICT
        from ui.apps.command_builder.utils import get_command_builder
        command_builder = get_command_builder()
        render_predict_page(state, command_builder)

    # Only persist state if it changed
    state.persist()
```

### Step 2: Implement Progressive Rendering in Training Page

**File:** `ui/apps/command_builder/components/training.py`

Render UI structure first, then load heavy data:

```python
def render_training_page(
    state: CommandBuilderState,
    command_builder: CommandBuilder,
    recommendation_service: UseCaseRecommendationService,
    config_parser: ConfigParser,
) -> None:
    page: CommandPageData = state.get_page(CommandType.TRAIN)

    # Render UI structure immediately
    st.markdown("### 🚀 Training configuration")

    # Load recommendations in background (with spinner)
    current_architecture = st.session_state.get(f"{SCHEMA_PREFIX}__architecture") or page.generated.values.get("architecture")

    # Show spinner only if recommendations are loading
    with st.spinner("Loading recommendations..."):
        recommendations = recommendation_service.for_architecture(current_architecture)

    # Render recommendations after UI is visible
    render_use_case_recommendations(
        recommendations,
        state,
        schema_prefix=SCHEMA_PREFIX,
        auxiliary_state_keys={"preprocessing_profile": PREPROCESSING_STATE_KEY},
    )

    # Rest of existing code...
    if st.button("Reset form", key="command_builder_train_reset"):
        state.active_use_case = None
        state.reset_command(CommandType.TRAIN)
        _reset_form_state()
        rerun_app()

    # Schema generation (this is cached from Phase 1, but show spinner for first load)
    schema_result = generate_ui_from_schema(str(SCHEMA_PATH))
    # ... rest of existing code ...
```

### Step 3: Optimize State Persistence

**File:** `ui/apps/command_builder/state.py`

Add method to only persist when state actually changes:

```python
@dataclass(slots=True)
class CommandBuilderState:
    # ... existing fields ...

    def persist(self) -> None:
        """Persist state to session. Only updates if state changed."""
        current_state = st.session_state.get(SESSION_KEY)

        # Only update if state actually changed
        if current_state != self:
            st.session_state[SESSION_KEY] = self

    def persist_if_changed(self) -> None:
        """Alternative: Use hash-based change detection for better performance."""
        import hashlib
        import pickle

        current_hash = hashlib.md5(pickle.dumps(self)).hexdigest()
        stored_hash = st.session_state.get(f"{SESSION_KEY}_hash")

        if stored_hash != current_hash:
            st.session_state[SESSION_KEY] = self
            st.session_state[f"{SESSION_KEY}_hash"] = current_hash
```

**File:** `ui/apps/command_builder/app.py`

Update to use optimized persistence:

```python
# At end of run() function:
state.persist()  # This now only updates if changed
```

### Step 4: Add Loading Indicators for Heavy Operations

**File:** `ui/apps/command_builder/components/training.py`

Wrap heavy operations with spinners:

```python
# Around line 58 in training.py
with st.spinner("Loading form schema..."):
    schema_result = generate_ui_from_schema(str(SCHEMA_PATH))
```

**File:** `ui/apps/command_builder/components/test.py` and `predict.py`

Add similar spinners for schema loading:

```python
# In test.py and predict.py, around schema loading
with st.spinner("Loading configuration..."):
    schema_result = generate_ui_from_schema(str(SCHEMA_PATH))
```

### Step 5: Optimize Component Imports

**File:** `ui/apps/command_builder/app.py`

Use conditional imports to avoid loading unused components:

```python
# At top of file, keep only essential imports
from __future__ import annotations
import streamlit as st
from ui.apps.command_builder.state import CommandBuilderState, CommandType

# Lazy import page components (they're imported when needed in run())
# This reduces initial import time
```

## Testing Strategy

### Functional Testing
1. **Verify lazy loading works:**
   - Start on Train page → verify only Train services loaded
   - Switch to Test page → verify Test services load
   - Switch to Predict page → verify Predict services load
   - Verify no errors from missing services

2. **Verify progressive rendering:**
   - UI structure appears immediately
   - Loading spinners show for heavy operations
   - No blank screens during page switch

3. **Verify state persistence:**
   - Make changes to form
   - Switch pages
   - Return to original page
   - Verify changes persisted

### Performance Testing
1. **Measure page switch time:**
   - Should be faster than Phase 1 (50-150ms improvement)
   - UI should appear within 100ms

2. **Measure initial load time:**
   - First page load should be faster
   - Subsequent page switches should be instant

### Edge Cases
1. **Rapid page switching:**
   - Quickly switch between pages
   - Verify no race conditions
   - Verify state is preserved

2. **Service initialization errors:**
   - Verify graceful error handling
   - Verify error messages are clear

## Success Criteria

- ✅ Services only loaded for active page
- ✅ UI structure appears within 100ms
- ✅ Loading indicators show for heavy operations
- ✅ State persistence optimized (only updates when changed)
- ✅ Page switch time < 150ms (from Phase 1's < 200ms)

## Dependencies

- **Requires:** Phase 1 (caching) must be completed first
- **Blocks:** Phase 3 (advanced optimizations)

## Rollback Plan

If issues occur:
1. Revert app.py to load all services upfront
2. Remove progressive rendering changes
3. Revert state persistence optimizations
4. Verify app works correctly

## Additional Notes

- **Import Optimization:** Conditional imports reduce initial load time but may complicate debugging
- **State Persistence:** Hash-based change detection is more efficient but adds complexity
- **Loading Indicators:** Balance between showing progress and cluttering UI

## References

- Phase 1 Plan: `artifacts/implementation_plans/2025-11-17_0126_phase-1:-implement-streamlit-caching-for-command-builder.md`
- Assessment: `artifacts/assessments/2025-11-17_0114_streamlit-command-builder-performance-assessment---page-switch-delays.md`
