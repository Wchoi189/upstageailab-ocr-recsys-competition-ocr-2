---
title: "Web Worker Prompt: Streamlit Performance Optimization"
date: "2025-12-06 18:09 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---







# Web Worker Prompt: Streamlit Performance Optimization

## Overview

You are tasked with optimizing Streamlit apps to reduce page switch/load delays. This covers two apps:
1. **Command Builder App** - Reduce page switch delays from 470-810ms to under 100ms
2. **Unified OCR App** - Reduce page load delays from 330-720ms to under 200ms

This is a multi-phase project with clear implementation plans and success criteria for each app.

## Project Context

### Command Builder App
**App Location:** `ui/apps/command_builder/app.py`

**Current Problem:**
- Page switching (Train/Test/Predict) takes 470-810ms
- Heavy initialization happens on every render
- No caching or lazy loading
- UI blocks while loading data

**Target Performance:**
- Page switch time: < 100ms (after all phases)
- UI appears within 50ms
- Smooth, responsive user experience

### Unified OCR App
**App Location:** `ui/apps/unified_ocr_app/app.py`

**Current Problem:**
- Page load time: 330-720ms (inference page worst case)
- Service creation on every action: 100-200ms
- Checkpoint loading on every render: 200-500ms
- Image processing not cached: 1-5 seconds

**Target Performance:**
- Page load time: < 200ms
- Service creation: < 10ms
- Checkpoint loading: < 50ms
- Cached image processing: < 1s

## Implementation Phases

### Command Builder App Phases

### Phase 1: Quick Wins - Caching (Priority: HIGH)
**Estimated Time:** 1-2 hours
**Plan:** `artifacts/implementation_plans/2025-11-17_0126_phase-1:-implement-streamlit-caching-for-command-builder.md`

**Key Tasks:**
1. Create `ui/apps/command_builder/utils.py` with cached service factories
2. Add `@st.cache_resource` decorators to service initialization
3. Add `@st.cache_data` decorators to expensive ConfigParser methods
4. Update `app.py` to use cached services
5. Update `ui_generator.py` to use cached methods

**Expected Impact:** 200-400ms reduction

**Success Criteria:**
- All pages work correctly
- Page switch time < 200ms
- No regressions

### Phase 2: Lazy Loading & Progressive Rendering (Priority: MEDIUM)
**Estimated Time:** 2-3 hours
**Plan:** `artifacts/implementation_plans/2025-11-17_0130_phase-2:-lazy-loading-and-progressive-rendering.md`

**Key Tasks:**
1. Refactor `app.py` to lazy-load services based on active page
2. Implement progressive rendering (UI first, then data)
3. Optimize state persistence (only update when changed)
4. Add loading indicators for heavy operations

**Expected Impact:** 50-150ms additional reduction

**Success Criteria:**
- Services only loaded for active page
- UI appears within 100ms
- Page switch time < 150ms

**Dependencies:** Requires Phase 1 completion

### Phase 3: Advanced Optimizations (Priority: LOW)
**Estimated Time:** 3-4 hours
**Plan:** `artifacts/implementation_plans/2025-11-17_0130_phase-3:-advanced-optimizations.md`

**Key Tasks:**
1. Implement module-level caching for ConfigParser
2. Optimize checkpoint scanning
3. Add background prefetching
4. Additional micro-optimizations

**Expected Impact:** Final 50ms reduction

**Success Criteria:**
- Page switch time < 100ms
- UI appears within 50ms
- No memory leaks

**Dependencies:** Requires Phase 1 and Phase 2 completion

### Unified OCR App Phases

### Unified App Phase 1: Service and Checkpoint Caching (Priority: HIGH)
**Estimated Time:** 1-2 hours
**Plan:** `artifacts/implementation_plans/2025-11-17_0136_unified-app-phase-1:-service-and-checkpoint-caching.md`

**Key Tasks:**
1. Create cached service factories in `services/__init__.py`
2. Add `@st.cache_resource` to service initialization
3. Add `@st.cache_data` to checkpoint loading
4. Update pages to use cached services

**Expected Impact:** 200-500ms reduction in page load, 100-200ms reduction in actions

**Success Criteria:**
- All pages work correctly
- Page load time < 200ms
- Service creation < 10ms
- Checkpoint loading < 50ms

### Unified App Phase 2: Image Processing and Config Optimization (Priority: MEDIUM)
**Estimated Time:** 2-3 hours
**Plan:** `artifacts/implementation_plans/2025-11-17_0136_unified-app-phase-2:-image-processing-and-config-optimization.md`

**Key Tasks:**
1. Implement hash-based caching for image processing
2. Optimize config loading with module-level cache
3. Optimize state persistence
4. Add loading indicators

**Expected Impact:** Image processing < 1s cached, config loading < 10ms

**Success Criteria:**
- Image processing cached correctly
- Config loading < 10ms
- State only persists when changed

**Dependencies:** Requires Unified App Phase 1 completion

## Getting Started

### Prerequisites
1. Read the assessments:
   - Command Builder: `artifacts/assessments/2025-11-17_0114_streamlit-command-builder-performance-assessment---page-switch-delays.md`
   - Unified App: `artifacts/assessments/2025-11-17_0136_unified-ocr-app-performance-assessment.md`
2. Read the Phase 1 implementation plan for the app you're working on
3. Understand the current codebase structure

### Development Setup
```bash
# Navigate to project root
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Run Command Builder app
uv run streamlit run ui/apps/command_builder/app.py

# Run Unified OCR App
uv run streamlit run ui/apps/unified_ocr_app/app.py

# Test page switching/loading and measure current performance
```

### Key Files to Understand

**Command Builder:**
- `ui/apps/command_builder/app.py` - Main app entry point
- `ui/utils/config_parser.py` - ConfigParser class (needs caching)
- `ui/utils/ui_generator.py` - UI generation (needs caching)
- `ui/apps/command_builder/components/` - Page components
- `ui/apps/command_builder/state.py` - State management

**Unified OCR App:**
- `ui/apps/unified_ocr_app/app.py` - Main app entry point
- `ui/apps/unified_ocr_app/pages/` - Page files (Preprocessing, Inference, Comparison)
- `ui/apps/unified_ocr_app/services/` - Service classes (PreprocessingService, InferenceService)
- `ui/apps/unified_ocr_app/shared_utils.py` - Shared utilities
- `ui/apps/unified_ocr_app/models/app_state.py` - State management

## Implementation Guidelines

### Code Quality
- Follow existing code style and patterns
- Add type hints to all new functions
- Add docstrings for new functions
- Test thoroughly before moving to next phase

### Testing Requirements
1. **Functional Testing:**
   - All pages must work correctly
   - All features must function as before
   - No regressions

2. **Performance Testing:**
   - Measure page switch time before/after
   - Use browser dev tools Network tab
   - Verify target performance metrics

3. **Edge Cases:**
   - Test with config file changes
   - Test cache invalidation
   - Test rapid page switching

### Streamlit Best Practices
- Use `@st.cache_resource` for objects (services, parsers)
- Use `@st.cache_data` for computed data (metadata, lists)
- Set appropriate TTL values (3600s = 1 hour default)
- Use `show_spinner=False` for fast operations
- Use `persist="disk"` if available (check version)

### Common Pitfalls to Avoid
1. **Don't cache everything:** Only cache expensive operations
2. **Don't break existing functionality:** Test thoroughly
3. **Don't ignore cache invalidation:** Configs can change
4. **Don't over-optimize:** Focus on high-impact changes first

## Workflow

### Phase 1 Workflow
1. Create `ui/apps/command_builder/utils.py` with cached factories
2. Update `app.py` to use cached services
3. Update `ui_generator.py` to use cached methods
4. Test all pages work correctly
5. Measure performance improvement
6. Document any issues or deviations

### Phase 2 Workflow (After Phase 1)
1. Refactor `app.py` for lazy loading
2. Add progressive rendering to page components
3. Optimize state persistence
4. Test all pages work correctly
5. Measure performance improvement
6. Document any issues or deviations

### Phase 3 Workflow (After Phase 3)
1. Implement module-level caching
2. Optimize checkpoint scanning
3. Add background prefetching
4. Test all pages work correctly
5. Measure performance improvement
6. Document any issues or deviations

## Success Metrics

### Performance Targets
- **Phase 1:** Page switch < 200ms (from 470-810ms)
- **Phase 2:** Page switch < 150ms
- **Phase 3:** Page switch < 100ms

### Quality Targets
- All tests pass
- No regressions
- Code follows project standards
- Documentation updated

## Questions & Support

### If You Get Stuck
1. Review the assessment document for context
2. Check the implementation plan for detailed steps
3. Review existing code patterns in the codebase
4. Test incrementally (don't change everything at once)

### Common Issues
- **Cache not working:** Check decorator placement and arguments
- **Import errors:** Verify import paths are correct
- **State not persisting:** Check state persistence logic
- **Performance not improving:** Profile to find bottlenecks

## Deliverables

For each phase, provide:
1. **Code changes:** All modified files
2. **Test results:** Performance measurements (before/after)
3. **Documentation:** Any deviations from plan or issues encountered
4. **Verification:** Confirmation that success criteria are met

## Additional Resources

- **Command Builder Assessment:** `artifacts/assessments/2025-11-17_0114_streamlit-command-builder-performance-assessment---page-switch-delays.md`
- **Unified App Assessment:** `artifacts/assessments/2025-11-17_0136_unified-ocr-app-performance-assessment.md`
- **Additional Optimizations:** `artifacts/implementation_plans/2025-11-17_0131_additional-performance-optimization-suggestions.md`
- **Streamlit Caching Docs:** https://docs.streamlit.io/library/advanced-features/caching
- **Streamlit Performance:** https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-improve-streamlit-app-performance

## Notes

- **Choose your app:** You can work on Command Builder or Unified App independently
- **Work on phases sequentially:** Don't skip ahead within an app
- **Test thoroughly:** After each phase, test all functionality
- **Document issues:** Any deviations from plan or unexpected behavior
- **Ask questions:** If anything is unclear, refer to the assessment documents

## Quick Reference

**Command Builder Plans:**
- Phase 1: `2025-11-17_0126_phase-1:-implement-streamlit-caching-for-command-builder.md`
- Phase 2: `2025-11-17_0130_phase-2:-lazy-loading-and-progressive-rendering.md`
- Phase 3: `2025-11-17_0130_phase-3:-advanced-optimizations.md`

**Unified App Plans:**
- Phase 1: `2025-11-17_0136_unified-app-phase-1:-service-and-checkpoint-caching.md`
- Phase 2: `2025-11-17_0136_unified-app-phase-2:-image-processing-and-config-optimization.md`

Good luck!
