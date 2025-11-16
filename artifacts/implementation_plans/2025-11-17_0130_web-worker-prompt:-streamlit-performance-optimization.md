---
title: "Web Worker Prompt: Streamlit Performance Optimization"
author: "ai-agent"
timestamp: "2025-11-17 01:30 KST"
branch: "main"
type: "implementation_plan"
category: "development"
status: "draft"
tags: []
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Web Worker Prompt: Streamlit Performance Optimization**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Web Worker Prompt: Streamlit Performance Optimization

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
# Web Worker Prompt: Streamlit Command Builder Performance Optimization

## Overview

You are tasked with optimizing the Streamlit Command Builder app to reduce page switch delays from 470-810ms to under 100ms. This is a multi-phase project with clear implementation plans and success criteria.

## Project Context

**App Location:** `ui/apps/command_builder/app.py`

**Current Problem:**
- Page switching (Train/Test/Predict) takes 470-810ms
- Heavy initialization happens on every render
- No caching or lazy loading
- UI blocks while loading data

**Target Performance:**
- Page switch time: < 100ms
- UI appears within 50ms
- Smooth, responsive user experience

## Implementation Phases

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
**Plan:** `artifacts/implementation_plans/2025-11-17_XXXX_phase-2:-lazy-loading-and-progressive-rendering.md`

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
**Plan:** `artifacts/implementation_plans/2025-11-17_XXXX_phase-3:-advanced-optimizations.md`

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

## Getting Started

### Prerequisites
1. Read the assessment: `artifacts/assessments/2025-11-17_0114_streamlit-command-builder-performance-assessment---page-switch-delays.md`
2. Read the Phase 1 implementation plan (link above)
3. Understand the current codebase structure

### Development Setup
```bash
# Navigate to project root
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Run the app to understand current behavior
uv run streamlit run ui/apps/command_builder/app.py

# Test page switching and measure current performance
```

### Key Files to Understand
- `ui/apps/command_builder/app.py` - Main app entry point
- `ui/utils/config_parser.py` - ConfigParser class (needs caching)
- `ui/utils/ui_generator.py` - UI generation (needs caching)
- `ui/apps/command_builder/components/` - Page components
- `ui/apps/command_builder/state.py` - State management

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

### Phase 3 Workflow (After Phase 2)
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

- **Assessment:** `artifacts/assessments/2025-11-17_0114_streamlit-command-builder-performance-assessment---page-switch-delays.md`
- **Streamlit Caching Docs:** https://docs.streamlit.io/library/advanced-features/caching
- **Streamlit Performance:** https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-improve-streamlit-app-performance

## Notes

- Work on phases sequentially (don't skip ahead)
- Test thoroughly after each phase
- Document any issues or unexpected behavior
- Ask questions if anything is unclear

Good luck! 🚀

