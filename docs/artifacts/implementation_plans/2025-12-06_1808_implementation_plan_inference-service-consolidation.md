---
title: "2025 11 11 Plan 004 Inference Service Consolidation"
date: "2025-12-06 18:08 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---





# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **PLAN-004: Inference Service Consolidation**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: PLAN-004: Inference Service Consolidation

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Phase 1, Task 1.1 - Analyze Current Inference Engine Implementation
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Review InferenceEngine class structure

### Implementation Outline (Checklist)

#### **Phase 1: Analyze Current Inference Engine**
1. [ ] **Task 1.1: Review InferenceEngine Implementation**
   - [ ] Read `ui/utils/inference/engine.py` and understand class structure
   - [ ] Locate `load_model()` method (around line 96)
   - [ ] Identify checkpoint loading logic
   - [ ] Document current engine initialization pattern

2. [ ] **Task 1.2: Identify Checkpoint Loading Patterns**
   - [ ] Read `ui/apps/inference/services/inference_runner.py`
   - [ ] Read `ui/apps/unified_ocr_app/services/inference_service.py`
   - [ ] Locate checkpoint loading calls
   - [ ] Document duplicate checkpoint loading patterns

3. [ ] **Task 1.3: Identify Tempfile Usage**
   - [ ] Search for `tempfile.NamedTemporaryFile` patterns: `grep -rn "tempfile.NamedTemporaryFile" ui/`
   - [ ] Search for `tempfile` imports: `grep -rn "import tempfile\|from tempfile import" ui/`
   - [ ] Document all tempfile usage locations
   - [ ] Identify which services use tempfiles

#### **Phase 2: Design Checkpoint Caching Strategy**
4. [ ] **Task 2.1: Design Caching API**
   - [ ] Review `ui/apps/inference/services/checkpoint/cache.py` for existing caching patterns
   - [ ] Design checkpoint caching API for InferenceEngine
   - [ ] Define cache key generation strategy (checkpoint path + config hash)
   - [ ] Document cache invalidation strategy

5. [ ] **Task 2.2: Design Singleton Engine Pattern**
   - [ ] Design shared engine instance pattern
   - [ ] Define engine lifecycle management
   - [ ] Document thread-safety considerations
   - [ ] Plan for graceful cleanup

#### **Phase 3: Implement Checkpoint Caching in InferenceEngine**
6. [ ] **Task 3.1: Add Caching to InferenceEngine**
   - [ ] Read `ui/utils/inference/engine.py` and locate `__init__` method
   - [ ] Add `_checkpoint_cache` dictionary to store loaded models
   - [ ] Add `_get_cache_key()` method for checkpoint path hashing
   - [ ] Modify `load_model()` to check cache before loading

7. [ ] **Task 3.2: Implement Cache Lookup Logic**
   - [ ] Add cache lookup in `load_model()` before loading checkpoint
   - [ ] Return cached model if checkpoint path matches
   - [ ] Add cache miss logging
   - [ ] Ensure cache key includes config path for uniqueness

8. [ ] **Task 3.3: Validate InferenceEngine Caching**
   - [ ] Run syntax check: `python -m py_compile ui/utils/inference/engine.py`
   - [ ] Run import check: `python -c "from ui.utils.inference.engine import InferenceEngine"`
   - [ ] Verify caching logic is correct
   - [ ] Check that cache key generation is deterministic

#### **Phase 4: Create Shared Engine Instance**
9. [ ] **Task 4.1: Implement Singleton Engine Factory**
   - [ ] Read `ui/utils/inference/__init__.py` and check current exports
   - [ ] Add `get_inference_engine()` function for shared engine instance
   - [ ] Implement singleton pattern with thread-safety
   - [ ] Add engine cleanup method

10. [ ] **Task 4.2: Export Caching API**
    - [ ] Update `ui/utils/inference/__init__.py` to export `get_inference_engine()`
    - [ ] Export `InferenceEngine` class
    - [ ] Document API usage
    - [ ] Ensure backward compatibility

11. [ ] **Task 4.3: Validate Shared Engine Implementation**
    - [ ] Run syntax check: `python -m py_compile ui/utils/inference/__init__.py`
    - [ ] Run import check: `python -c "from ui.utils.inference import get_inference_engine"`
    - [ ] Verify singleton pattern works correctly
    - [ ] Check that engine can be reused across calls

#### **Phase 5: Migrate Inference Services to Cached Engine**
12. [ ] **Task 5.1: Update InferenceRunner Service**
    - [ ] Read `ui/apps/inference/services/inference_runner.py`
    - [ ] Locate `_perform_inference()` method (around line 105)
    - [ ] Replace engine instantiation with `get_inference_engine()`
    - [ ] Update imports to use shared engine

13. [ ] **Task 5.2: Update Unified OCR App Service**
    - [ ] Read `ui/apps/unified_ocr_app/services/inference_service.py`
    - [ ] Locate `_run_inference_internal()` method (around line 113)
    - [ ] Replace engine instantiation with `get_inference_engine()`
    - [ ] Update imports to use shared engine

14. [ ] **Task 5.3: Validate Service Migrations**
    - [ ] Run syntax check: `python -m py_compile ui/apps/*/services/*.py`
    - [ ] Run import check: `python -c "from ui.apps.inference.services.inference_runner import InferenceService"`
    - [ ] Verify services use shared engine
    - [ ] Check that engine caching works correctly

#### **Phase 6: Eliminate Tempfile Duplication**
15. [ ] **Task 6.1: Identify Tempfile Usage in Services**
    - [ ] Read `ui/apps/inference/services/inference_runner.py` and locate tempfile usage (around line 100)
    - [ ] Read `ui/apps/unified_ocr_app/services/inference_service.py` and locate tempfile usage (around line 144)
    - [ ] Document tempfile usage patterns
    - [ ] Identify which services can use numpy arrays directly

16. [ ] **Task 6.2: Update InferenceEngine to Accept Numpy Arrays**
    - [ ] Read `ui/utils/inference/engine.py` and locate `predict()` or inference method
    - [ ] Add method to accept numpy array input directly
    - [ ] Update method to handle both file paths and numpy arrays
    - [ ] Ensure backward compatibility with file path inputs

17. [ ] **Task 6.3: Update Services to Stream Numpy Arrays**
    - [ ] Update `inference_runner.py` to pass numpy arrays directly
    - [ ] Update `inference_service.py` to pass numpy arrays directly
    - [ ] Remove tempfile creation and cleanup code
    - [ ] Update error handling for numpy array inputs

18. [ ] **Task 6.4: Validate Tempfile Elimination**
    - [ ] Run syntax check: `python -m py_compile ui/apps/*/services/*.py`
    - [ ] Verify no tempfile imports remain: `grep -rn "import tempfile\|from tempfile import" ui/apps/`
    - [ ] Verify no tempfile usage: `grep -rn "tempfile.NamedTemporaryFile\|tempfile.mkstemp" ui/apps/`
    - [ ] Check that numpy array streaming works correctly

#### **Phase 7: Final Validation and Cleanup**
19. [ ] **Task 7.1: Verify Engine Caching Logic**
    - [ ] Verify cache key generation is deterministic
    - [ ] Check that cache invalidation works correctly
    - [ ] Confirm engine reuse across multiple calls
    - [ ] Validate thread-safety of singleton pattern

20. [ ] **Task 7.2: Verify Tempfile Elimination**
    - [ ] Search for remaining tempfile usage: `grep -rn "tempfile" ui/apps/`
    - [ ] Verify numpy array streaming works
    - [ ] Check that file path inputs still work (backward compatibility)
    - [ ] Confirm no memory leaks from cached engines

21. [ ] **Task 7.3: Run Comprehensive Validation**
    - [ ] Syntax check all modified files
    - [ ] Import check all modified files
    - [ ] Verify no circular imports
    - [ ] Check that backward compatibility is maintained

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Checkpoint caching implemented in InferenceEngine
- [ ] Singleton engine pattern implemented
- [ ] Tempfile usage eliminated
- [ ] Numpy array streaming implemented

### **Integration Points**
- [ ] InferenceRunner uses shared engine
- [ ] Unified OCR App uses shared engine
- [ ] Backward compatibility maintained
- [ ] Engine lifecycle managed correctly

### **Quality Assurance**
- [ ] All files pass syntax validation
- [ ] All imports resolve (syntax-wise)
- [ ] No tempfile usage remains
- [ ] Engine caching works correctly
- [ ] Thread-safety verified

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] Checkpoint caching reduces redundant model loading
- [ ] Shared engine instance reused across calls
- [ ] Tempfile usage eliminated
- [ ] Numpy arrays streamed directly to engine

### **Technical Requirements**
- [ ] All Python files compile without syntax errors
- [ ] All imports resolve (syntax-wise)
- [ ] No circular import dependencies
- [ ] Engine caching logic is correct
- [ ] Backward compatibility maintained
- [ ] Inference latency improved

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: VERY HIGH ⚠️⚠️⚠️⚠️
### **Active Mitigation Strategies**:
1. **Feature Flag**: Add config flag to enable/disable caching
2. **Backward Compatibility**: Maintain old path as fallback
3. **Gradual Rollout**: Update one service at a time
4. **Comprehensive Validation**: Syntax and import checks after each change

### **Fallback Options**:
1. **If caching breaks inference**: Disable caching via feature flag, investigate issue
2. **If singleton pattern breaks**: Revert to per-call engine instantiation
3. **If tempfile removal breaks**: Keep tempfile usage, optimize differently
4. **If numpy array streaming fails**: Revert to file path inputs, investigate issue

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

**TASK:** Review InferenceEngine Class Structure

**OBJECTIVE:** Understand current InferenceEngine implementation to design caching strategy

**APPROACH:**
1. Read `ui/utils/inference/engine.py` and understand class structure
2. Locate `load_model()` method (around line 96)
3. Identify checkpoint loading logic
4. Document current engine initialization pattern

**SUCCESS CRITERIA:**
- InferenceEngine structure understood
- Checkpoint loading logic identified
- Ready to design caching strategy

---

## 📝 **Context Management for Web Workers**

### **Recommended Context Scope**
**Focus on ONE file at a time** to avoid context overflow:

1. **Phase 1 (Analysis)**:
   - Context: Read each file individually to understand structure
   - Max files: 1 per task
   - Estimated tokens: ~2000 per file

2. **Phase 2 (Design)**:
   - Context: Review existing caching patterns, design new API
   - Max files: 1 (cache.py)
   - Estimated tokens: ~1500

3. **Phase 3-4 (Engine Implementation)**:
   - Context: `ui/utils/inference/engine.py` and `__init__.py`
   - Max files: 2
   - Estimated tokens: ~3000

4. **Phase 5-6 (Service Migration)**:
   - Context: One service file at a time
   - Max files: 1 per task
   - Estimated tokens: ~2000 per file

5. **Phase 7 (Validation)**:
   - Context: Use grep for pattern searches, read only if needed
   - Max files: 0 (grep only)
   - Estimated tokens: ~500

### **Context Optimization Strategies**
- **Use grep first** to locate patterns before reading files
- **Read only relevant sections** (use line offsets)
- **Skip large docstrings** when not needed
- **Focus on specific methods** rather than entire files

### **Validation Commands (No Runtime Needed)**
```bash
# Syntax check
python -m py_compile <file>

# Import check
python -c "from <module> import <class>"

# Pattern search for tempfile usage
grep -rn "tempfile.NamedTemporaryFile\|tempfile.mkstemp" ui/

# Pattern search for tempfile imports
grep -rn "import tempfile\|from tempfile import" ui/

# Verify engine caching
grep -rn "_checkpoint_cache\|get_inference_engine" ui/
```

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
