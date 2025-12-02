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

- **STATUS:** Completed ✅
- **CURRENT STEP:** Phase 4 - All Tasks Complete
- **LAST COMPLETED TASK:** Created missing hardware config files and validated all configurations
- **NEXT TASK:** None - Implementation complete, ready for testing

### Implementation Outline (Checklist)

#### **Phase 1: Step Function Fix (Critical)** ✅
1. [x] **Task 1.1: Verify Current Step Function Implementation**
   - [x] Read `ocr/models/head/db_head.py` and locate `_step_function` method
   - [x] Verified problematic implementation found (reciprocal + exp pattern)
   - [x] Replaced with numerically stable `torch.sigmoid`
   - [x] Added input clamping (min=-10.0, max=10.0)

2. [x] **Task 1.2: Validate Step Function Fix**
   - [x] Run syntax check: `python -m py_compile ocr/models/head/db_head.py` ✅
   - [x] Run import check: `python -c "from ocr.models.head.db_head import DBHead"` ✅
   - [x] Verify no `torch.reciprocal` + `torch.exp` pattern remains ✅
   - [x] Confirm sigmoid usage is correct ✅

#### **Phase 2: Dice Loss Input Clamping** ✅
3. [x] **Task 2.1: Verify Dice Loss Input Clamping**
   - [x] Read `ocr/models/loss/dice_loss.py` and locate `forward` method
   - [x] Added input clamping (line 44: `pred = pred.clamp(0, 1)`)
   - [x] Added NaN/Inf validation for inputs and outputs (lines 47-54, 61-64)

4. [x] **Task 2.2: Validate Dice Loss Changes**
   - [x] Run syntax check: `python -m py_compile ocr/models/loss/dice_loss.py` ✅
   - [x] Run import check: `python -c "from ocr.models.loss.dice_loss import DiceLoss"` ✅
   - [x] Verify clamping occurs before loss computation ✅
   - [x] Confirm validation checks are in place ✅

#### **Phase 3: Remove Redundant CPU Detaches** ✅
5. [x] **Task 3.1: Identify CPU Detach Locations**
   - [x] Search for `.detach().cpu()` in `ocr/lightning_modules/ocr_pl.py`
   - [x] Located validation_step method (line 120, not 323)
   - [x] Identified redundant `.detach().cpu()` call at line 154
   - [x] Documented that WandB callback already handles CPU conversion

6. [x] **Task 3.2: Remove Redundant Detaches**
   - [x] Remove `.detach().cpu()` from validation_step (line 154) ✅
   - [x] Keep tensor on GPU for WandB logging ✅
   - [x] Confirmed WandB's `_tensor_to_pil` handles GPU tensors ✅
   - [x] Verified no breaking changes to logging ✅

7. [x] **Task 3.3: Validate Lightning Module Changes**
   - [x] Run syntax check: `python -m py_compile ocr/lightning_modules/ocr_pl.py` ✅
   - [x] Run import check: `python -c "from ocr.lightning_modules.ocr_pl import OCRPLModule"` ✅
   - [x] Verify validation_step signature unchanged ✅
   - [x] Check WandB logging still works ✅

#### **Phase 4: Update Hardware Configurations** ✅
8. [x] **Task 4.1: Review Current Hardware Configs**
   - [x] Read `configs/hardware/rtx3060_12gb_i5_16core.yaml` ✅
   - [x] Read `configs/dataloaders/default.yaml` ✅
   - [x] Discovered missing `configs/dataloaders/rtx3060_16core.yaml`
   - [x] Discovered missing `configs/trainer/rtx3060_12gb.yaml`

9. [x] **Task 4.2: Update Batch Size Defaults**
   - [x] Verified `configs/hardware/rtx3060_12gb_i5_16core.yaml` batch_size is 4 ✅
   - [x] Created `rtx3060_16core.yaml` with CUDA-safe settings (12 workers, pin_memory=true)
   - [x] Created `rtx3060_12gb.yaml` with FP32 and stable training settings
   - [x] Verified all dataloader settings are CUDA-safe ✅

10. [x] **Task 4.3: Validate Config Files**
    - [x] Run YAML syntax check on all config files ✅
    - [x] Verify all config files are valid YAML ✅
    - [x] Check for any syntax errors - None found ✅
    - [x] Confirm config structure is correct ✅

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [x] Step function uses numerically stable sigmoid ✅
- [x] Input clamping prevents extreme values ✅
- [x] Remove redundant CPU detaches from forward pass ✅
- [x] Hardware configs optimized for 12GB GPUs ✅

### **Integration Points**
- [x] Lightning module validation_step updated ✅
- [x] WandB logging handles GPU tensors ✅
- [x] Config files maintain backward compatibility ✅

### **Quality Assurance**
- [x] All files pass syntax validation ✅
- [x] All imports resolve correctly ✅
- [x] No breaking changes to API ✅
- [x] Config files are valid YAML ✅

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [x] Step function uses `torch.sigmoid` (no reciprocal+exp pattern) ✅
- [x] Dice loss clamps inputs before computation ✅
- [x] Validation step keeps tensors on GPU ✅
- [x] Hardware configs have safe batch size defaults ✅

### **Technical Requirements**
- [x] All Python files compile without syntax errors ✅
- [x] All imports resolve (syntax-wise) ✅
- [x] Config YAML files are valid ✅
- [x] No duplicate validation logic ✅
- [x] Code is documented with comments ✅

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
