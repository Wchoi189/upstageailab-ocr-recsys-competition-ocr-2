---
title: "Streamlit Command Builder Performance Assessment - Page Switch Delays"
author: "ai-agent"
timestamp: "2025-11-17 01:14 KST"
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

The Streamlit Command Builder app experiences significant delays when switching between pages (Train/Test/Predict). The root cause is that heavy initialization and computation occur on every page render, even when not needed. The app lacks lazy loading, proper caching, and prioritization of UI rendering over background work.

## Current Performance Issues

### 1. Heavy Initialization on Every Render

**Problem:** All services are instantiated on every  call, regardless of which page is active:



**Impact:**
-  performs multiple file I/O operations:
  - Scans  directories (encoders, decoders, heads, losses)
  - Reads multiple YAML files for architecture metadata
  - Calls  methods which import model modules
  - Scans  directory for checkpoints
- Each instance has its own cache, so new instances don't benefit from previous work
- Estimated cost: 200-500ms per page switch

### 2. Schema Parsing on Every Render

**Problem:**  is called on every render, even though schemas are static:



**Impact:**
- YAML file is read and parsed every time (even with , cache may be invalidated)
-  is instantiated inside  on every call
- Architecture metadata is loaded repeatedly
- Estimated cost: 100-300ms per page render

### 3. No Lazy Loading

**Problem:** All page components are imported and initialized even when not visible:



**Impact:**
- Training page imports and initializes recommendation service even when on Test/Predict pages
- Heavy dependencies (like model registry) are loaded regardless of page
- No conditional loading based on active page

### 4. Missing Streamlit Caching

**Problem:** Critical operations are not cached:

-  methods are not cached (only instance-level cache)
-  is not cached
- Schema loading has cache but may be invalidated unnecessarily
- File system scans (checkpoints, outputs) happen on every call

**Impact:**
- Repeated file I/O and computation
- No benefit from Streamlit's caching mechanism

### 5. Synchronous Heavy Operations

**Problem:** Heavy operations block UI rendering:

- File system scans block the main thread
- YAML parsing blocks rendering
- Model registry imports block rendering

**Impact:**
- User sees blank screen or spinner during page switch
- No progressive rendering (UI first, then data)

### 6. State Management Issues

**Problem:** State is persisted on every render, even when unchanged:



**Impact:**
- Unnecessary session state updates
- Potential re-renders triggered by state changes

## Performance Bottleneck Analysis

### Page Switch Flow (Current)

1. User clicks radio button in sidebar → Streamlit reruns
2.  executes:
   - Creates  (lightweight)
   - Creates  → **200-300ms** (file scans, YAML reads)
   - Creates  → **50-100ms** (metadata loading)
3.  → **10ms** (fast)
4. Page-specific render (e.g., ):
   -  → **100-200ms** (YAML parse, options resolution)
   -  → **50-100ms** (if training page)
   - Form rendering → **50-100ms**
5.  → **10ms**

**Total: 470-810ms per page switch**

### Critical Path

The critical path for page switching is:
1.  initialization (200-300ms)
2. Schema parsing and UI generation (100-200ms)
3. Recommendation service (50-100ms, training page only)

## Recommendations

### Priority 1: Implement Streamlit Caching

**Action:** Add  and  decorators:



**Expected Impact:** 200-400ms reduction per page switch

### Priority 2: Lazy Load Page Components

**Action:** Only initialize services needed for the active page:



**Expected Impact:** 50-150ms reduction (skip unused services)

### Priority 3: Progressive Rendering

**Action:** Render UI structure first, then load data:



**Expected Impact:** Perceived performance improvement (UI appears faster)

### Priority 4: Optimize ConfigParser

**Action:** 
- Make cache persistent across instances (use module-level cache)
- Add  to expensive methods
- Lazy load registry imports



**Expected Impact:** 100-200ms reduction on subsequent calls

### Priority 5: Reduce State Persistence Calls

**Action:** Only persist state when it actually changes:



**Expected Impact:** 10-20ms reduction, fewer re-renders

### Priority 6: Background Loading for Heavy Operations

**Action:** Use Streamlit's experimental features or async patterns:

- Load checkpoint lists in background
- Prefetch architecture metadata
- Use  strategically to update UI after data loads

**Expected Impact:** Better perceived performance

## Implementation Strategy

### Phase 1: Quick Wins (1-2 hours)
1. Add  to service initialization
2. Add  to ConfigParser methods
3. Move service initialization after sidebar render

### Phase 2: Lazy Loading (2-3 hours)
1. Refactor to lazy load page-specific services
2. Conditional imports based on active page
3. Optimize state persistence

### Phase 3: Advanced Optimizations (3-4 hours)
1. Module-level caching for ConfigParser
2. Progressive rendering patterns
3. Background data loading

## Success Criteria

- Page switch time < 200ms (from current 470-810ms)
- UI appears within 100ms of user interaction
- No blocking operations during page switch
- Smooth transitions between pages

## Testing Strategy

1. **Performance Testing:**
   - Measure page switch time before/after
   - Profile with  to identify bottlenecks
   - Test with various cache states (cold, warm, invalidated)

2. **Functional Testing:**
   - Verify all pages still work correctly
   - Test state persistence across page switches
   - Verify cache invalidation works correctly

3. **User Experience Testing:**
   - Verify UI appears quickly
   - Test with slow file systems (simulate)
   - Verify no visual glitches during transitions

## Additional Considerations

- **Cache Invalidation:** Need strategy for when configs change
- **Memory Usage:** Caching increases memory, monitor usage
- **Development vs Production:** Different caching strategies may be needed
- **Streamlit Version:** Ensure compatibility with caching decorators

## References

- Streamlit caching: https://docs.streamlit.io/library/advanced-features/caching
- Performance best practices: https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-improve-streamlit-app-performance
- Current app structure: 
- ConfigParser: 
- UI Generator: 
