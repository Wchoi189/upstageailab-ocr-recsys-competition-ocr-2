---
title: "PLAN-001: Core Training Stabilization"
author: "ai-agent"
date: "2025-11-11"
type: "implementation_plan"
category: "architecture"
status: "draft"
version: "0.1"
tags: ["implementation_plan", "architecture", "training", "cuda", "stability", "critical"]
timestamp: "2025-11-11 02:00 KST"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **PLAN-001: Core Training Stabilization**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: PLAN-001: Core Training Stabilization

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Phase 1, Task 1.1 - Verify Current Implementation
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Verify step function implementation in db_head.py

### Implementation Outline (Checklist)

#### **Phase 1: Step Function Fix (Critical)**
1. [ ] **Task 1.1: Verify Current Step Function Implementation**
   - [ ] Read `ocr/models/head/db_head.py` and locate `_step_function` method
   - [ ] Verify current implementation uses `torch.sigmoid` (already fixed)
   - [ ] Check for any remaining `torch.reciprocal(1 + torch.exp(-k*(x-y)))` patterns
   - [ ] Verify input clamping is present (lines 182-183)

2. [ ] **Task 1.2: Validate Step Function Fix**
   - [ ] Run syntax check: `python -m py_compile ocr/models/head/db_head.py`
   - [ ] Run import check: `python -c "from ocr.models.head.db_head import DBHead"`
   - [ ] Verify no `torch.reciprocal` + `torch.exp` pattern remains
   - [ ] Confirm sigmoid usage is correct (line 188)

#### **Phase 2: Dice Loss Input Clamping**
3. [ ] **Task 2.1: Verify Dice Loss Input Clamping**
   - [ ] Read `ocr/models/loss/dice_loss.py` and locate `forward` method
   - [ ] Verify input clamping is present (line 57: `pred = pred.clamp(0, 1)`)
   - [ ] Check NaN/Inf validation is present (lines 44-53)

4. [ ] **Task 2.2: Validate Dice Loss Changes**
   - [ ] Run syntax check: `python -m py_compile ocr/models/loss/dice_loss.py`
   - [ ] Run import check: `python -c "from ocr.models.loss.dice_loss import DiceLoss"`
   - [ ] Verify clamping occurs before loss computation
   - [ ] Confirm validation checks are in place

#### **Phase 3: Remove Redundant CPU Detaches**
5. [ ] **Task 3.1: Identify CPU Detach Locations**
   - [ ] Search for `.detach().cpu()` in `ocr/lightning_modules/ocr_pl.py`
   - [ ] Locate validation_step method (line 323)
   - [ ] Identify all `.detach().cpu()` calls in forward pass
   - [ ] Document which calls are redundant

6. [ ] **Task 3.2: Remove Redundant Detaches**
   - [ ] Remove `.detach().cpu()` from validation_step (line 323)
   - [ ] Keep tensor on GPU for WandB logging
   - [ ] Update WandB logger to handle GPU tensors
   - [ ] Verify no breaking changes to logging

7. [ ] **Task 3.3: Validate Lightning Module Changes**
   - [ ] Run syntax check: `python -m py_compile ocr/lightning_modules/ocr_pl.py`
   - [ ] Run import check: `python -c "from ocr.lightning_modules.ocr_pl import OCRPLModule"`
   - [ ] Verify validation_step signature unchanged
   - [ ] Check WandB logging still works

#### **Phase 4: Update Hardware Configurations**
8. [ ] **Task 4.1: Review Current Hardware Configs**
   - [ ] Read `configs/hardware/rtx3060_12gb_i5_16core.yaml`
   - [ ] Read `configs/dataloaders/default.yaml`
   - [ ] Read `configs/dataloaders/rtx3060_16core.yaml`
   - [ ] Document current batch size defaults

9. [ ] **Task 4.2: Update Batch Size Defaults**
   - [ ] Update `configs/hardware/rtx3060_12gb_i5_16core.yaml` batch_size to 4 (already set)
   - [ ] Verify dataloader num_workers settings are CUDA-safe
   - [ ] Check pin_memory and persistent_workers settings
   - [ ] Ensure prefetch_factor is appropriate

10. [ ] **Task 4.3: Validate Config Files**
    - [ ] Run YAML syntax check: `python -c "import yaml; yaml.safe_load(open('configs/hardware/rtx3060_12gb_i5_16core.yaml'))"`
    - [ ] Verify all config files are valid YAML
    - [ ] Check for any syntax errors
    - [ ] Confirm config structure is correct

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [x] Step function uses numerically stable sigmoid (already implemented)
- [x] Input clamping prevents extreme values (already implemented)
- [ ] Remove redundant CPU detaches from forward pass
- [ ] Hardware configs optimized for 12GB GPUs

### **Integration Points**
- [ ] Lightning module validation_step updated
- [ ] WandB logging handles GPU tensors
- [ ] Config files maintain backward compatibility

### **Quality Assurance**
- [ ] All files pass syntax validation
- [ ] All imports resolve correctly
- [ ] No breaking changes to API
- [ ] Config files are valid YAML

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] Step function uses `torch.sigmoid` (no reciprocal+exp pattern)
- [ ] Dice loss clamps inputs before computation
- [ ] Validation step keeps tensors on GPU
- [ ] Hardware configs have safe batch size defaults

### **Technical Requirements**
- [ ] All Python files compile without syntax errors
- [ ] All imports resolve (syntax-wise)
- [ ] Config YAML files are valid
- [ ] No duplicate validation logic
- [ ] Code is documented with comments

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: CRITICAL ⚠️⚠️⚠️
### **Active Mitigation Strategies**:
1. **Atomic Commits**: One logical change per commit for easy rollback
2. **Syntax Validation**: Check all files before committing
3. **Backward Compatibility**: Maintain existing API signatures
4. **Incremental Changes**: Test each phase independently

### **Fallback Options**:
1. **If step function fix breaks**: Revert to previous implementation, investigate alternative fix
2. **If CPU detach removal breaks WandB**: Keep detach but optimize differently
3. **If config changes break training**: Revert config files, use command-line overrides
4. **If validation fails**: Rollback specific commit, document issue

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

**TASK:** Verify Current Step Function Implementation

**OBJECTIVE:** Confirm that the step function fix is already implemented and identify any remaining issues

**APPROACH:**
1. Read `ocr/models/head/db_head.py` and locate `_step_function` method (around line 158)
2. Verify it uses `torch.sigmoid(self.k * (x_clamped - y_clamped))` (line 188)
3. Check for any remaining `torch.reciprocal(1 + torch.exp(-k*(x-y)))` patterns
4. Verify input clamping is present (lines 182-183)

**SUCCESS CRITERIA:**
- Step function uses sigmoid (not reciprocal+exp)
- Input clamping is present
- No syntax errors in file
- File can be imported successfully

---

## 📝 **Context Management for Web Workers**

### **Recommended Context Scope**
**Focus on ONE file at a time** to avoid context overflow:

1. **Phase 1-2 (Step Function & Dice Loss)**:
   - Context: `ocr/models/head/db_head.py` + `ocr/models/loss/dice_loss.py`
   - Max files: 2
   - Estimated tokens: ~3000

2. **Phase 3 (Lightning Module)**:
   - Context: `ocr/lightning_modules/ocr_pl.py` (read relevant sections only)
   - Max files: 1
   - Estimated tokens: ~2000

3. **Phase 4 (Config Files)**:
   - Context: Individual config files one at a time
   - Max files: 1 per task
   - Estimated tokens: ~1000 per file

### **Context Optimization Strategies**
- **Read only relevant sections** (use line offsets)
- **Skip large docstrings** when not needed
- **Focus on specific methods** rather than entire files
- **Use grep to locate patterns** before reading full files

### **Validation Commands (No Runtime Needed)**
```bash
# Syntax check
python -m py_compile <file>

# Import check (may fail on missing deps, but should not fail on syntax)
python -c "from <module> import <class>"

# YAML validation
python -c "import yaml; yaml.safe_load(open('<file>'))"

# Pattern search
grep -n "torch.reciprocal.*torch.exp" <file>
```

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*

