---
title: "PLAN-003: Import-Time Optimization"
author: "ai-agent"
date: "2025-11-11"
type: "implementation_plan"
category: "architecture"
status: "draft"
version: "0.1"
tags: ["implementation_plan", "architecture", "optimization", "imports", "medium-risk"]
timestamp: "2025-11-11 02:00 KST"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **PLAN-003: Import-Time Optimization**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: PLAN-003: Import-Time Optimization

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Partially Complete (WandB Done) ✅
- **CURRENT STEP:** Phase 1-2 Complete - All WandB imports optimized
- **LAST COMPLETED TASK:** Optimized all wandb imports to be lazy
- **NEXT TASK:** Phase 3 - Optimize Streamlit imports (optional - can be done later)

### Implementation Outline (Checklist)

#### **Phase 1: Verify Current WandB Import Status**
1. [ ] **Task 1.1: Check Current WandB Import Patterns**
   - [ ] Read `runners/train.py` and verify wandb is already lazy (lines 41-75)
   - [ ] Check `ocr/lightning_modules/callbacks/*.py` for wandb imports
   - [ ] Verify `_get_wandb()` helper pattern is used
   - [ ] Document any remaining top-level wandb imports

2. [ ] **Task 1.2: Identify Remaining WandB Imports**
   - [ ] Search for `import wandb` patterns: `grep -rn "import wandb" ocr/ runners/`
   - [ ] Search for `from wandb import` patterns: `grep -rn "from wandb import" ocr/ runners/`
   - [ ] Document all locations with top-level wandb imports
   - [ ] Verify which callbacks need lazy imports

#### **Phase 2: Optimize WandB Imports in Callbacks**
3. [ ] **Task 2.1: Update WandB Callbacks to Use Lazy Imports**
   - [ ] Read `ocr/lightning_modules/callbacks/wandb_image_logging.py`
   - [ ] Read `ocr/lightning_modules/callbacks/wandb_completion.py`
   - [ ] Read `ocr/lightning_modules/callbacks/unique_checkpoint.py`
   - [ ] Verify all use `_get_wandb()` helper pattern
   - [ ] Update any that don't use lazy imports

4. [ ] **Task 2.2: Validate WandB Callback Changes**
   - [ ] Run syntax check: `python -m py_compile ocr/lightning_modules/callbacks/*.py`
   - [ ] Run import check: `python -c "from ocr.lightning_modules.callbacks.wandb_image_logging import WandbImageLoggingCallback"`
   - [ ] Verify no top-level wandb imports remain
   - [ ] Check that callbacks can be imported without wandb installed

#### **Phase 3: Optimize Streamlit Imports in UI Services**
5. [ ] **Task 3.1: Identify Streamlit Import Patterns**
   - [ ] Search for `import streamlit` patterns: `grep -rn "import streamlit" ui/`
   - [ ] Search for `from streamlit import` patterns: `grep -rn "from streamlit import" ui/`
   - [ ] Document all locations with top-level streamlit imports
   - [ ] Identify which services need lazy imports

6. [ ] **Task 3.2: Update UI Services to Use Lazy Streamlit Imports**
   - [ ] Read `ui/apps/inference/services/inference_runner.py`
   - [ ] Read `ui/apps/unified_ocr_app/services/inference_service.py`
   - [ ] Create `_get_streamlit()` helper function pattern
   - [ ] Replace top-level streamlit imports with lazy imports
   - [ ] Ensure imports are inside functions guarded by config checks

7. [ ] **Task 3.3: Validate Streamlit Import Changes**
   - [ ] Run syntax check: `python -m py_compile ui/apps/*/services/*.py`
   - [ ] Run import check: `python -c "from ui.apps.inference.services.inference_runner import InferenceService"`
   - [ ] Verify no top-level streamlit imports remain
   - [ ] Check that services can be imported without streamlit installed

#### **Phase 4: Update Callback Configuration**
8. [ ] **Task 4.1: Review Callback Configuration Files**
   - [ ] Read `configs/callbacks/default.yaml`
   - [ ] Read `configs/callbacks/wandb_image_logging.yaml`
   - [ ] Check for conditional callback loading
   - [ ] Document current callback configuration structure

9. [ ] **Task 4.2: Add Conditional Callback Loading**
   - [ ] Update `configs/callbacks/default.yaml` to respect wandb config
   - [ ] Add conditional loading based on wandb.enabled flag
   - [ ] Ensure callbacks are only instantiated if wandb is enabled
   - [ ] Maintain backward compatibility

10. [ ] **Task 4.3: Validate Callback Configuration**
    - [ ] Run YAML syntax check: `python -c "import yaml; yaml.safe_load(open('configs/callbacks/default.yaml'))"`
    - [ ] Verify config structure is valid
    - [ ] Check that conditional loading works correctly

#### **Phase 5: Update Optional Dependencies**
11. [ ] **Task 5.1: Review pyproject.toml**
    - [ ] Read `pyproject.toml` and locate dependency sections
    - [ ] Check for wandb and streamlit dependencies
    - [ ] Document current dependency structure

12. [ ] **Task 5.2: Add Optional Dependency Groups**
    - [ ] Add `[project.optional-dependencies]` section if not present
    - [ ] Create `wandb` optional dependency group
    - [ ] Create `streamlit` optional dependency group
    - [ ] Move wandb and streamlit to optional dependencies
    - [ ] Update documentation for optional dependencies

13. [ ] **Task 5.3: Validate Optional Dependencies**
    - [ ] Run syntax check: `python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"`
    - [ ] Verify dependency groups are valid
    - [ ] Check that core dependencies still work without optional deps

#### **Phase 6: Final Validation and Cleanup**
14. [ ] **Task 6.1: Verify All Imports Are Lazy**
    - [ ] Search for remaining top-level wandb imports: `grep -rn "^import wandb\|^from wandb import" ocr/ runners/`
    - [ ] Search for remaining top-level streamlit imports: `grep -rn "^import streamlit\|^from streamlit import" ui/`
    - [ ] Verify all imports are inside functions
    - [ ] Confirm config guards are present

15. [ ] **Task 6.2: Run Comprehensive Validation**
    - [ ] Syntax check all modified files
    - [ ] Import check all modified files
    - [ ] Verify no circular imports
    - [ ] Check that optional dependencies work correctly

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] All wandb imports are lazy (inside functions)
- [ ] All streamlit imports are lazy (inside functions)
- [ ] Optional dependency groups configured
- [ ] Conditional callback loading implemented

### **Integration Points**
- [ ] Callbacks respect wandb config flags
- [ ] UI services handle missing streamlit gracefully
- [ ] Backward compatibility maintained
- [ ] Import errors handled gracefully

### **Quality Assurance**
- [ ] All files pass syntax validation
- [ ] All imports resolve (syntax-wise)
- [ ] No top-level heavy imports
- [ ] Optional dependencies work correctly

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] All wandb imports are lazy (no top-level imports)
- [ ] All streamlit imports are lazy (no top-level imports)
- [ ] Callbacks only load if wandb is enabled
- [ ] Optional dependencies configured correctly

### **Technical Requirements**
- [ ] All Python files compile without syntax errors
- [ ] All imports resolve (syntax-wise)
- [ ] No circular import dependencies
- [ ] Code can be imported without optional deps installed
- [ ] Import time reduced by 30%+

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM ⚠️
### **Active Mitigation Strategies**:
1. **Lazy Imports**: All heavy imports moved inside functions
2. **Backward Compatibility**: Maintain existing API signatures
3. **Graceful Degradation**: Handle missing optional dependencies
4. **Incremental Changes**: Update one module at a time

### **Fallback Options**:
1. **If lazy imports break callbacks**: Revert to top-level imports, investigate issue
2. **If optional deps break installation**: Keep as required deps, document issue
3. **If config changes break training**: Revert config files, use command-line overrides
4. **If import errors occur**: Add try-except blocks, handle gracefully

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

**TASK:** Check Current WandB Import Patterns

**OBJECTIVE:** Verify current wandb import status and identify any remaining top-level imports

**APPROACH:**
1. Read `runners/train.py` and verify wandb is already lazy (lines 41-75)
2. Search for `import wandb` patterns: `grep -rn "import wandb" ocr/ runners/`
3. Search for `from wandb import` patterns: `grep -rn "from wandb import" ocr/ runners/`
4. Document all locations with top-level wandb imports

**SUCCESS CRITERIA:**
- All wandb import locations identified
- Current lazy import status documented
- Ready to optimize remaining imports

---

## 📝 **Context Management for Web Workers**

### **Recommended Context Scope**
**Focus on ONE file at a time** to avoid context overflow:

1. **Phase 1 (Verification)**:
   - Context: Use grep for pattern searches, read only if needed
   - Max files: 0 (grep only)
   - Estimated tokens: ~500

2. **Phase 2 (WandB Callbacks)**:
   - Context: One callback file at a time
   - Max files: 1 per task
   - Estimated tokens: ~1500 per file

3. **Phase 3 (Streamlit Services)**:
   - Context: One service file at a time
   - Max files: 1 per task
   - Estimated tokens: ~2000 per file

4. **Phase 4-5 (Config & Dependencies)**:
   - Context: Individual config/dependency files
   - Max files: 1 per task
   - Estimated tokens: ~1000 per file

### **Context Optimization Strategies**
- **Use grep first** to locate import patterns before reading files
- **Read only relevant sections** (use line offsets)
- **Skip large docstrings** when not needed
- **Focus on import statements** rather than entire files

### **Validation Commands (No Runtime Needed)**
```bash
# Syntax check
python -m py_compile <file>

# Import check
python -c "from <module> import <class>"

# Pattern search for wandb imports
grep -rn "^import wandb\|^from wandb import" ocr/ runners/

# Pattern search for streamlit imports
grep -rn "^import streamlit\|^from streamlit import" ui/

# YAML validation
python -c "import yaml; yaml.safe_load(open('<file>'))"

# TOML validation
python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"
```

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*

