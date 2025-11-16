---
title: "Phase 1: Implement Streamlit Caching for Command Builder"
author: "ai-agent"
timestamp: "2025-11-17 01:26 KST"
branch: "main"
type: "implementation_plan"
category: "development"
status: "draft"
tags: []
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Phase 1: Implement Streamlit Caching for Command Builder**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Phase 1: Implement Streamlit Caching for Command Builder

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Phase 1, Task 1.1 - Create Cached Service Factories
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Create `ui/apps/command_builder/utils.py` with cached service factory functions

### Implementation Outline (Checklist)

#### **Phase 1: Implement Streamlit Caching (Week 1)**
1. [ ] **Task 1.1: Create Cached Service Factories**
   - [ ] Create `ui/apps/command_builder/utils.py` file
   - [ ] Add `@st.cache_resource` decorators to service factories
   - [ ] Add `@st.cache_data` decorators to ConfigParser method wrappers
   - [ ] Test imports work correctly

2. [ ] **Task 1.2: Update app.py to Use Cached Services**
   - [ ] Update imports in `app.py`
   - [ ] Replace direct instantiation with cached factories
   - [ ] Test app still works correctly

3. [ ] **Task 1.3: Update UI Generator to Use Cached Methods**
   - [ ] Update `_get_options_from_source()` in `ui_generator.py`
   - [ ] Import cached methods from utils
   - [ ] Test UI generation still works

4. [ ] **Task 1.4: Add Cache Invalidation (Optional)**
   - [ ] Add cache clear button in sidebar (development)
   - [ ] Test cache clearing works
   - [ ] Document for development use

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Modular Design: Separate cached factories in utils module
- [ ] Streamlit Caching: Use `@st.cache_resource` for services, `@st.cache_data` for data
- [ ] Cache TTL: 1 hour (3600s) for config data, persistent for services
- [ ] State Management: No changes to existing state management

### **Integration Points**
- [ ] Integration with Streamlit caching system
- [ ] Use existing ConfigParser, CommandBuilder, UseCaseRecommendationService
- [ ] Maintain compatibility with existing UI generator

### **Quality Assurance**
- [ ] Functional Testing: All pages work correctly
- [ ] Performance Testing: Page switch time < 200ms
- [ ] Cache Testing: Verify caching works and invalidation works
- [ ] Regression Testing: No functionality broken

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] All pages (Train/Test/Predict) load and function correctly
- [ ] Command generation works on all pages
- [ ] Form validation works correctly
- [ ] No regressions in existing functionality

### **Technical Requirements**
- [ ] Page switch time reduced from 470-810ms to < 200ms
- [ ] Cached operations complete in < 10ms
- [ ] Memory usage increase < 100MB
- [ ] Code is type-hinted and documented
- [ ] Cache invalidation works correctly

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW
### **Active Mitigation Strategies**:
1. Incremental Development: Implement one step at a time, test after each
2. Comprehensive Testing: Test all pages and functionality after changes
3. Rollback Plan: Easy to revert if issues occur (just remove utils.py and revert app.py)

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

**TASK:** Create `ui/apps/command_builder/utils.py` with cached service factory functions

**OBJECTIVE:** Create a new utils module that provides cached access to CommandBuilder, ConfigParser, and UseCaseRecommendationService instances, plus cached wrappers for expensive ConfigParser methods.

**APPROACH:**
1. Create new file `ui/apps/command_builder/utils.py`
2. Add imports for streamlit, CommandBuilder, ConfigParser, UseCaseRecommendationService
3. Create `get_command_builder()`, `get_config_parser()`, `get_recommendation_service()` with `@st.cache_resource`
4. Create cached wrappers for `get_architecture_metadata()`, `get_available_models()`, `get_available_architectures()` with `@st.cache_data`
5. Test that imports work correctly

**SUCCESS CRITERIA:**
- File `ui/apps/command_builder/utils.py` exists
- All factory functions are defined with proper decorators
- Imports work without errors
- Functions return correct types

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
## Objective

Implement Streamlit caching for service initialization and expensive operations in the Command Builder app. This is the highest-impact, lowest-risk optimization that will reduce page switch time by 200-400ms.

## Context

The Command Builder app (`ui/apps/command_builder/app.py`) currently creates new instances of `ConfigParser`, `CommandBuilder`, and `UseCaseRecommendationService` on every page render. These services perform expensive file I/O operations that should be cached.

**Current Performance:**
- Page switch time: 470-810ms
- ConfigParser initialization: 200-300ms
- Schema parsing: 100-200ms
- Recommendation service: 50-100ms

**Target Performance:**
- Page switch time: < 200ms
- Cached operations: < 10ms

## Approach

1. Add `@st.cache_resource` decorators to service factory functions
2. Add `@st.cache_data` decorators to expensive ConfigParser methods
3. Ensure cache keys are stable and appropriate
4. Add cache invalidation strategy for development

## Implementation Steps

### Step 1: Create Cached Service Factories

**File:** `ui/apps/command_builder/utils.py` (create new file)

Create cached factory functions:

```python
from __future__ import annotations

import streamlit as st
from ui.utils.command import CommandBuilder
from ui.utils.config_parser import ConfigParser
from ui.apps.command_builder.services.recommendations import UseCaseRecommendationService

@st.cache_resource
def get_command_builder() -> CommandBuilder:
    """Get cached CommandBuilder instance."""
    return CommandBuilder()

@st.cache_resource
def get_config_parser() -> ConfigParser:
    """Get cached ConfigParser instance."""
    return ConfigParser()

@st.cache_resource
def get_recommendation_service() -> UseCaseRecommendationService:
    """Get cached UseCaseRecommendationService instance."""
    return UseCaseRecommendationService(get_config_parser())

# Cached wrapper functions for ConfigParser methods
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_architecture_metadata() -> dict:
    """Get cached architecture metadata."""
    return get_config_parser().get_architecture_metadata()

@st.cache_data(ttl=3600)
def get_available_models() -> dict[str, list[str]]:
    """Get cached available models."""
    return get_config_parser().get_available_models()

@st.cache_data(ttl=3600)
def get_available_architectures() -> list[str]:
    """Get cached available architectures."""
    return get_config_parser().get_available_architectures()
```

### Step 2: Update app.py to Use Cached Services

**File:** `ui/apps/command_builder/app.py`

Replace direct instantiation with cached factories:

```python
# BEFORE (lines 19-21):
command_builder = CommandBuilder()
config_parser = ConfigParser()
recommendation_service = UseCaseRecommendationService(config_parser)

# AFTER:
from ui.apps.command_builder.utils import (
    get_command_builder,
    get_config_parser,
    get_recommendation_service
)

command_builder = get_command_builder()
config_parser = get_config_parser()
recommendation_service = get_recommendation_service()
```

### Step 3: Update UI Generator to Use Cached Methods

**File:** `ui/utils/ui_generator.py`

Update `_get_options_from_source()` function (around line 50) to use cached methods:

```python
@st.cache_data(show_spinner=False, ttl=3600)
def _get_options_from_source(source: str) -> list[str]:
    """Resolve dynamic options list with caching."""
    from ui.apps.command_builder.utils import (
        get_config_parser,
        get_architecture_metadata,
        get_available_models,
        get_available_architectures
    )

    cp = get_config_parser()
    model_source_map = {
        "models.backbones": "backbones",
        "models.encoders": "encoders",
        "models.decoders": "decoders",
        "models.heads": "heads",
        "models.optimizers": "optimizers",
        "models.losses": "losses",
    }
    if source in model_source_map:
        models = get_available_models()
        return models.get(model_source_map[source], [])
    if source == "models.architectures":
        return get_available_architectures()
    if source == "checkpoints":
        return cp.get_available_checkpoints()
    return cp.get_available_datasets() if source == "datasets" else []
```

### Step 4: Add Cache Invalidation for Development (Optional)

**File:** `ui/apps/command_builder/app.py`

Add optional cache clearing button in sidebar (development only):

```python
def run() -> None:
    st.set_page_config(page_title="OCR Command Builder", page_icon="🔍", layout="wide")
    st.title("🔍 OCR Training Command Builder")
    st.caption("Build and execute training, testing, and prediction commands with metadata-aware defaults.")

    state = CommandBuilderState.from_session()

    # Development: Add cache clear button (remove in production)
    if st.sidebar.button("🔄 Clear Cache", help="Clear all cached data (development only)"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    # ... rest of existing code ...
```

## Testing Strategy

### Functional Testing
1. **Verify all pages work:**
   - Navigate to Train page → verify form loads
   - Navigate to Test page → verify form loads
   - Navigate to Predict page → verify form loads
   - Generate commands on each page → verify they work

2. **Verify caching works:**
   - First page load: Should see normal load time
   - Second page load: Should be faster
   - Switch pages multiple times: Should be consistently fast

3. **Verify cache invalidation:**
   - Click "Clear Cache" button
   - Verify cache is cleared
   - Verify app still works after clearing

### Performance Testing
1. **Measure page switch time:**
   - Use browser dev tools Network tab
   - Measure time between page selection and UI update
   - Before: 470-810ms, After: < 200ms (target)

2. **Profile with cProfile (optional):**
   ```bash
   python -m cProfile -o profile.stats -m streamlit run ui/apps/command_builder/app.py
   ```

### Edge Cases
1. **Config file changes:**
   - Modify a config file
   - Verify cache TTL expires after 1 hour
   - Or manually clear cache to see changes

2. **Memory usage:**
   - Monitor memory usage with caching enabled
   - Verify no memory leaks

## Success Criteria

- ✅ All pages load and function correctly
- ✅ Page switch time reduced from 470-810ms to < 200ms
- ✅ No regressions in functionality
- ✅ Cache invalidation works correctly
- ✅ Memory usage is acceptable (< 100MB increase)

## Rollback Plan

If issues occur:
1. Revert changes to `app.py` (use direct instantiation)
2. Remove `utils.py` file
3. Verify app works without caching
4. Investigate issues before re-applying

## Additional Notes

- **TTL Selection:** 1 hour (3600s) is a good default. Adjust based on how often configs change.
- **Cache Keys:** Streamlit automatically creates cache keys from function arguments. Ensure arguments are stable.
- **Memory:** Caching increases memory usage. Monitor and adjust TTL if needed.
- **Development:** Consider shorter TTL (e.g., 60s) for development to see changes faster.

## References

- Assessment: `artifacts/assessments/2025-11-17_0114_streamlit-command-builder-performance-assessment---page-switch-delays.md`
- Streamlit Caching: https://docs.streamlit.io/library/advanced-features/caching
- Current app: `ui/apps/command_builder/app.py`
- ConfigParser: `ui/utils/config_parser.py`

