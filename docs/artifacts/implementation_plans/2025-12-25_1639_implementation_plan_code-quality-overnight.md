---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'development']
title: "Overnight Code Quality Resolution (Ruff & Mypy)"
date: "2025-12-25 16:39 (KST)"
branch: "main"
---

# Overnight Code Quality Resolution (Ruff & Mypy)

This plan outlines the systematic resolution of linting and type errors across the project, optimized for autonomous batch execution with minimal interruptions.

---

# Living Implementation Blueprint: Overnight Code Quality Resolution (Ruff & Mypy)

## Progress Tracker
- **STATUS:** PLANNING
- **CURRENT STEP:** Phase 1, Task 1.1 - Automated Ruff Fixes
- **LAST COMPLETED TASK:** Assessment of current state
- **NEXT TASK:** Execute project-wide `ruff check --fix`

### Implementation Outline (Checklist)

#### **Phase 1: Automated Formatting and Linting (Ruff)**
1. [ ] **Task 1.1: Automated Ruff Fixes**
   - [ ] Run `uv run ruff check --fix .` to resolve common linting errors (unused imports, simple syntax issues).
   - [ ] Run `uv run ruff format .` to ensure consistent formatting across all 473+ files.
   - [ ] Verify formatting with `uv run ruff format --check .`.

2. [ ] **Task 1.2: Specialized Rule Cleanup**
   - [ ] Target specific complex rules (e.g., `B`, `C4`, `UP`) that require more than simple auto-fixing.
   - [ ] Review and fix `isort` (`I`) violations project-wide.

#### **Phase 2: Modular Mypy Resolution**
3. [ ] **Task 2.1: Utility Layer Type Alignment**
   - [ ] Focus on `ocr/utils/` and `AgentQMS/agent_tools/utilities/`.
   - [ ] Resolve type errors in core utility functions to provide a stable base for higher-level modules.
   - [ ] Verify `ocr/utils/` with `uv run mypy --check-untyped-defs ocr/utils/`.

4. [ ] **Task 2.2: Inference and App Layer Resolution**
   - [ ] Resolve errors in `ocr/inference/` and `apps/`.
   - [ ] Handle any complex type issues involving external libraries (PaddleOCR, Torch).

#### **Phase 3: Strictness and Configuration**
5. [ ] **Task 3.1: Strict Rule Activation**
   - [ ] Enable `check_untyped_defs = true` and `disallow_untyped_defs = true` incrementally in `pyproject.toml`.
   - [ ] Resolve the resulting errors (the "hundreds" mentioned by the user).

6. [ ] **Task 3.2: Final Project-Wide Quality Check**
   - [ ] Run `make quality-check` globally.
   - [ ] Ensure all 0 errors reported.

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [x] Use existing `pyproject.toml` and `ruff` configurations.
- [ ] Implement incremental resolution to avoid huge, unreviewable PRs.
- [ ] Minimize "compute burden" by using `mypy` caching and modular checks.

### **Quality Assurance**
- [ ] All code must pass `ruff check .` with 0 errors.
- [ ] All code must pass `mypy .` with at least current strictness levels.
- [ ] Aim for `check_untyped_defs = true` by the end of execution.

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [x] No breaking changes to existing inference logic.
- [ ] Code passes all existing unit tests (`make test`).

### **Technical Requirements**
- [ ] 0 Ruff errors.
- [ ] 0 Mypy errors with `check_untyped_defs = true`.
- [ ] Reduced compute burden for future quality checks due to cleaner type graph.

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM
### **Active Mitigation Strategies**:
1. **Incremental Approach**: Fix one module at a time.
2. **Automated Fixers**: Use `ruff --fix` for low-risk changes.
3. **Commit often**: Small, atomic commits for each fix set.

### **Fallback Options**:
1. **Rollback**: If a fix breaks logic, revert using `git checkout`.
2. **Exclusion**: If a file is too complex to fix overnight, add it to the `exclude` list with a `NOTE` for manual follow-up.

---

## 🚀 **Immediate Next Action**

**TASK**: Execute Phase 1 Task 1.1

**OBJECTIVE**: Resolve all auto-fixable linting errors and format the entire codebase.

**APPROACH**:
1. Run `uv run ruff check --fix .`
2. Run `uv run ruff format .`
3. Commit the changes.

**SUCCESS CRITERIA**:
- `ruff check .` shows significantly fewer errors.
- `ruff format --check .` passes.


---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
