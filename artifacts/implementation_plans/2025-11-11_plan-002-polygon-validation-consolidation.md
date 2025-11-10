---
title: "PLAN-002: Polygon Validation Consolidation"
author: "ai-agent"
date: "2025-11-11"
type: "implementation_plan"
category: "architecture"
status: "draft"
version: "0.1"
tags: ["implementation_plan", "architecture", "validation", "consolidation", "high-risk"]
timestamp: "2025-11-11 02:00 KST"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **PLAN-002: Polygon Validation Consolidation**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: PLAN-002: Polygon Validation Consolidation

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Phase 1, Task 1.1 - Identify Duplicate Validation Logic
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Locate all polygon validation implementations

### Implementation Outline (Checklist)

#### **Phase 1: Identify Duplicate Validation Logic**
1. [ ] **Task 1.1: Locate All Polygon Validation Implementations**
   - [ ] Read `ocr/utils/polygon_utils.py` and identify existing validators
   - [ ] Read `ocr/datasets/base.py` and locate polygon validation code
   - [ ] Read `ocr/datasets/db_collate_fn.py` and locate polygon validation code
   - [ ] Read `ocr/lightning_modules/callbacks/wandb_image_logging.py` and locate validation code
   - [ ] Document all validation functions and their signatures

2. [ ] **Task 1.2: Analyze Validation Logic Differences**
   - [ ] Compare validation logic across all locations
   - [ ] Identify common patterns and differences
   - [ ] Document which validations are redundant
   - [ ] Identify which validations are unique to each location

#### **Phase 2: Create Shared Validation Utilities**
3. [ ] **Task 2.1: Design Shared Validator API**
   - [ ] Review `ocr/utils/polygon_utils.py` existing functions
   - [ ] Design unified validator function signatures
   - [ ] Ensure backward compatibility with existing code
   - [ ] Document API contract

4. [ ] **Task 2.2: Implement Shared Validators in polygon_utils.py**
   - [ ] Add `validate_polygon()` function to `ocr/utils/polygon_utils.py`
   - [ ] Add `validate_polygon_batch()` function for batch validation
   - [ ] Ensure functions handle both 2D (N, 2) and 3D (1, N, 2) shapes
   - [ ] Add out-of-bounds checking if image dimensions provided
   - [ ] Add degenerate polygon detection
   - [ ] Include proper docstrings and type hints

5. [ ] **Task 2.3: Validate Shared Validator Implementation**
   - [ ] Run syntax check: `python -m py_compile ocr/utils/polygon_utils.py`
   - [ ] Run import check: `python -c "from ocr.utils.polygon_utils import validate_polygon"`
   - [ ] Verify function signatures match design
   - [ ] Check docstrings are complete

#### **Phase 3: Migrate Dataset Base to Shared Validators**
6. [ ] **Task 3.1: Update ValidatedOCRDataset**
   - [ ] Read `ocr/datasets/base.py` and locate polygon validation calls
   - [ ] Replace inline validation with `validate_polygon()` from polygon_utils
   - [ ] Ensure backward compatibility
   - [ ] Update imports to include polygon_utils

7. [ ] **Task 3.2: Validate Dataset Base Changes**
   - [ ] Run syntax check: `python -m py_compile ocr/datasets/base.py`
   - [ ] Run import check: `python -c "from ocr.datasets.base import ValidatedOCRDataset"`
   - [ ] Verify no duplicate validation logic remains
   - [ ] Check that validation behavior is unchanged

#### **Phase 4: Migrate Collate Function to Shared Validators**
8. [ ] **Task 4.1: Update DBCollateFN**
   - [ ] Read `ocr/datasets/db_collate_fn.py` and locate polygon validation
   - [ ] Replace inline validation with shared validators
   - [ ] Ensure batch validation uses `validate_polygon_batch()`
   - [ ] Update imports

9. [ ] **Task 4.2: Validate Collate Function Changes**
   - [ ] Run syntax check: `python -m py_compile ocr/datasets/db_collate_fn.py`
   - [ ] Run import check: `python -c "from ocr.datasets.db_collate_fn import DBCollateFN"`
   - [ ] Verify validation logic is consolidated
   - [ ] Check function signature unchanged

#### **Phase 5: Migrate WandB Callback to Shared Validators**
10. [ ] **Task 5.1: Update WandB Image Logging Callback**
    - [ ] Read `ocr/lightning_modules/callbacks/wandb_image_logging.py`
    - [ ] Locate `_postprocess_polygons` method (around line 170)
    - [ ] Replace inline validation with shared validators
    - [ ] Update imports

11. [ ] **Task 5.2: Validate WandB Callback Changes**
    - [ ] Run syntax check: `python -m py_compile ocr/lightning_modules/callbacks/wandb_image_logging.py`
    - [ ] Run import check: `python -c "from ocr.lightning_modules.callbacks.wandb_image_logging import WandbImageLoggingCallback"`
    - [ ] Verify validation logic uses shared utilities
    - [ ] Check callback interface unchanged

#### **Phase 6: Final Validation and Cleanup**
12. [ ] **Task 6.1: Verify All Locations Use Shared Validators**
    - [ ] Search for duplicate validation patterns: `grep -r "is_degenerate_polygon\|_is_degenerate_polygon" ocr/`
    - [ ] Search for out-of-bounds checks: `grep -r "is_polygon_out_of_bounds\|out_of_bounds" ocr/`
    - [ ] Verify all three locations import from polygon_utils
    - [ ] Confirm no duplicate validation logic remains

13. [ ] **Task 6.2: Run Comprehensive Validation**
    - [ ] Syntax check all modified files
    - [ ] Import check all modified files
    - [ ] Verify no circular imports
    - [ ] Check that all function signatures match

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Shared validators in `ocr/utils/polygon_utils.py`
- [ ] All three locations use shared utilities
- [ ] Backward compatibility maintained
- [ ] No duplicate validation logic

### **Integration Points**
- [ ] ValidatedOCRDataset uses shared validators
- [ ] DBCollateFN uses shared validators
- [ ] WandB callback uses shared validators
- [ ] All imports resolve correctly

### **Quality Assurance**
- [ ] All files pass syntax validation
- [ ] All imports resolve (syntax-wise)
- [ ] No duplicate validation code
- [ ] Function signatures match across locations

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] All three locations use `validate_polygon()` from polygon_utils
- [ ] No duplicate validation logic remains
- [ ] Validation behavior is consistent across locations
- [ ] Backward compatibility maintained

### **Technical Requirements**
- [ ] All Python files compile without syntax errors
- [ ] All imports resolve (syntax-wise)
- [ ] No circular import dependencies
- [ ] Code is documented with docstrings
- [ ] Type hints are present

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: HIGH ⚠️⚠️
### **Active Mitigation Strategies**:
1. **Gradual Migration**: Update one location at a time
2. **Backward Compatibility**: Maintain existing function signatures
3. **Comprehensive Validation**: Syntax and import checks after each change
4. **Atomic Commits**: One location per commit for easy rollback

### **Fallback Options**:
1. **If shared validators break**: Revert to inline validation, investigate API design
2. **If migration breaks dataset**: Revert specific location, keep others migrated
3. **If validation behavior changes**: Compare old vs new behavior, fix discrepancies
4. **If imports fail**: Check circular dependencies, fix import order

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

**TASK:** Locate All Polygon Validation Implementations

**OBJECTIVE:** Identify all locations where polygon validation is implemented to understand what needs to be consolidated

**APPROACH:**
1. Read `ocr/utils/polygon_utils.py` and identify existing validators (lines 240-289)
2. Read `ocr/datasets/base.py` and search for polygon validation code
3. Read `ocr/datasets/db_collate_fn.py` and search for polygon validation code
4. Read `ocr/lightning_modules/callbacks/wandb_image_logging.py` and locate `_postprocess_polygons` method (around line 170)

**SUCCESS CRITERIA:**
- All validation locations identified
- Validation logic differences documented
- Ready to design shared validator API

---

## 📝 **Context Management for Web Workers**

### **Recommended Context Scope**
**Focus on ONE file at a time** to avoid context overflow:

1. **Phase 1 (Identification)**:
   - Context: Read each file individually to identify validation code
   - Max files: 1 per task
   - Estimated tokens: ~2000 per file

2. **Phase 2 (Shared Validators)**:
   - Context: `ocr/utils/polygon_utils.py` (read relevant sections)
   - Max files: 1
   - Estimated tokens: ~3000

3. **Phase 3-5 (Migration)**:
   - Context: One target file at a time
   - Max files: 1 per phase
   - Estimated tokens: ~2000 per file

4. **Phase 6 (Validation)**:
   - Context: Use grep for pattern searches, read only if needed
   - Max files: 0 (grep only)
   - Estimated tokens: ~500

### **Context Optimization Strategies**
- **Use grep first** to locate validation patterns before reading files
- **Read only relevant sections** (use line offsets)
- **Skip large docstrings** when not needed
- **Focus on specific methods** rather than entire files

### **Validation Commands (No Runtime Needed)**
```bash
# Syntax check
python -m py_compile <file>

# Import check
python -c "from <module> import <class>"

# Pattern search for duplicate validation
grep -rn "is_degenerate_polygon\|_is_degenerate_polygon" ocr/

# Pattern search for out-of-bounds checks
grep -rn "is_polygon_out_of_bounds\|out_of_bounds" ocr/

# Verify imports
grep -rn "from ocr.utils.polygon_utils import" ocr/
```

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*

