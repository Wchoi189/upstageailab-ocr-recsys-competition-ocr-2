# Web Worker Execution Prompt Template

## Quick Start Prompt

Use this prompt to start implementing any of the PLAN-00X implementation plans with Claude Code web:

---

## 🚀 **Execution Prompt Template**

```
You are an autonomous AI agent executing an implementation plan for an OCR system architecture refactoring.

## Context
I need you to execute [PLAN-00X: Plan Name] from the implementation plans directory. This is part of a larger architecture refactoring effort to stabilize training, consolidate validation logic, optimize imports, consolidate inference services, and clean up legacy code.

## Your Task
Execute the implementation plan located at:
`artifacts/implementation_plans/2025-11-11_plan-00X-[plan-name].md`

## Execution Guidelines

### 1. Read the Plan First
- Read the entire implementation plan document
- Understand the risk level and mitigation strategies
- Review the Progress Tracker section
- Note the current status and next task

### 2. Context Management (CRITICAL)
You are working in a web environment with limited context. Follow these rules:

**For Each Task:**
- Read ONLY the files needed for the current task
- Use grep/search tools FIRST to locate patterns before reading files
- Read only relevant sections (use line offsets when possible)
- Maximum 1-2 files in context at a time
- Estimated token budget: ~2000-3000 tokens per task

**Context Optimization Strategies:**
- Use `grep` to locate patterns before reading full files
- Read only specific methods/sections, not entire files
- Skip large docstrings when not needed
- Use line offsets to read specific sections

### 3. Execution Workflow

**For Each Task:**
1. **Read the task description** in the plan's "Implementation Outline"
2. **Check the "NEXT TASK"** in the Progress Tracker
3. **Execute the task** step by step
4. **Validate immediately** using the provided validation commands
5. **Update the Progress Tracker** after completing the task

### 4. Validation (No Runtime Required)

After each change, run these validation commands:

```bash
# Syntax check
python -m py_compile <file>

# Import check (may fail on missing deps, but should not fail on syntax)
python -c "from <module> import <class>"

# Pattern search (before reading files)
grep -rn "<pattern>" <directory>

# YAML validation
python -c "import yaml; yaml.safe_load(open('<file>'))"
```

**Important:** These commands validate syntax and imports only. They don't require runtime dependencies.

### 5. Progress Tracking

**After each task completion:**
1. Update the Progress Tracker in the plan document:
   - Change STATUS to "In Progress" (if starting) or keep updated
   - Update CURRENT STEP to the current phase/task
   - Update LAST COMPLETED TASK with description
   - Update NEXT TASK to the next task in sequence
   - Mark completed checkboxes with [x]

2. Document any discoveries or blockers in the plan

### 6. Commit Strategy

**After completing each logical change:**
- Make atomic commits (one logical change per commit)
- Use clear commit messages: `[PLAN-00X] <description>`
- Commit frequently to enable easy rollback

### 7. Error Handling

**If you encounter an error:**
1. Document the error in the Progress Tracker
2. Check the "Fallback Options" section in the plan
3. Try the fallback approach if available
4. If blocked, document the blocker and suggest next steps

**If validation fails:**
1. Review the error message
2. Fix the syntax/import issue
3. Re-run validation
4. Document the fix in comments

### 8. Risk Mitigation

**Before making changes:**
- Review the risk level in the plan
- Understand the mitigation strategies
- Check fallback options
- Ensure you understand rollback procedures

**During execution:**
- Make one change at a time
- Validate after each change
- Keep changes atomic and reversible
- Document decisions in code comments

## Starting the Implementation

**First Steps:**
1. Read the implementation plan: `artifacts/implementation_plans/2025-11-11_plan-00X-[plan-name].md`
2. Review the Progress Tracker section
3. Identify the current status and next task
4. Begin with the task listed in "NEXT TASK"

**Example First Task:**
If the plan shows:
- STATUS: Not Started
- NEXT TASK: "Verify step function implementation in db_head.py"

Then:
1. Read `ocr/models/head/db_head.py` (focus on `_step_function` method around line 158)
2. Verify the implementation matches the requirements
3. Run validation commands
4. Update the Progress Tracker
5. Move to the next task

## Important Constraints

1. **No Runtime Testing:** You cannot run the code, only validate syntax and imports
2. **Limited Context:** Work with 1-2 files at a time
3. **Atomic Changes:** One logical change per commit
4. **Validation Required:** Validate after each change
5. **Progress Tracking:** Update the plan document after each task

## Success Criteria

The implementation is complete when:
- All tasks in the Implementation Outline are checked [x]
- All validation commands pass
- Progress Tracker shows STATUS: Completed
- All changes are committed with clear messages
- No syntax or import errors remain

## Questions?

If you need clarification:
1. Check the plan document first (it should have all details)
2. Review the "Risk Mitigation & Fallbacks" section
3. Check the "Context Management" section for file reading guidance
4. Document blockers in the Progress Tracker

---

**Now, please start executing [PLAN-00X: Plan Name]. Begin by reading the plan document and identifying the first task.**
```

---

## 📋 **Plan-Specific Prompts**

### For PLAN-001: Core Training Stabilization

```
Execute PLAN-001: Core Training Stabilization.

This is a CRITICAL risk plan that fixes:
- Step function numerical instability (already fixed, verify)
- Dice loss input clamping (already fixed, verify)
- Remove redundant CPU detaches from validation
- Update hardware configs for 12GB GPUs

Start by reading: artifacts/implementation_plans/2025-11-11_plan-001-core-training-stabilization.md

Focus on validation and verification since most fixes are already implemented.
```

### For PLAN-002: Polygon Validation Consolidation

```
Execute PLAN-002: Polygon Validation Consolidation.

This is a HIGH risk plan that consolidates duplicate polygon validation logic across:
- ocr/datasets/base.py
- ocr/datasets/db_collate_fn.py
- ocr/lightning_modules/callbacks/wandb_image_logging.py

Start by reading: artifacts/implementation_plans/2025-11-11_plan-002-polygon-validation-consolidation.md

Use grep first to locate validation patterns before reading files.
```

### For PLAN-003: Import-Time Optimization

```
Execute PLAN-003: Import-Time Optimization.

This is a MEDIUM risk plan that optimizes import times by:
- Making wandb imports lazy (already partially done, verify)
- Making streamlit imports lazy
- Adding optional dependency groups
- Conditional callback loading

Start by reading: artifacts/implementation_plans/2025-11-11_plan-003-import-time-optimization.md

Use grep to locate import patterns before reading files.
```

### For PLAN-004: Inference Service Consolidation

```
Execute PLAN-004: Inference Service Consolidation.

This is a VERY HIGH risk plan that consolidates inference services by:
- Adding checkpoint caching to InferenceEngine
- Creating shared engine instance
- Eliminating tempfile duplication
- Streaming numpy arrays directly

Start by reading: artifacts/implementation_plans/2025-11-11_plan-004-inference-service-consolidation.md

Work carefully - this affects UI functionality. Make atomic changes and validate frequently.
```

### For PLAN-005: Legacy Cleanup & Config Consolidation

```
Execute PLAN-005: Legacy Cleanup & Config Consolidation.

This is a LOW risk plan that cleans up:
- Archive backup directories
- Remove duplicate scripts
- Consolidate config presets
- Update documentation references

Start by reading: artifacts/implementation_plans/2025-11-11_plan-005-legacy-cleanup-config-consolidation.md

Use terminal commands (ls, mv, rm) and grep for reference searches.
```

---

## 🎯 **Quick Reference Checklist**

Before starting any plan, verify:
- [ ] Plan document is read and understood
- [ ] Risk level is understood
- [ ] Validation commands are available
- [ ] Git branch is correct
- [ ] Context management strategy is clear

During execution:
- [ ] One change at a time
- [ ] Validate after each change
- [ ] Update Progress Tracker after each task
- [ ] Commit frequently with clear messages
- [ ] Document decisions in code comments

After completion:
- [ ] All tasks checked [x]
- [ ] All validation commands pass
- [ ] Progress Tracker shows STATUS: Completed
- [ ] All changes committed
- [ ] No syntax/import errors

---

## 📝 **Example Execution Flow**

1. **Read Plan:** `artifacts/implementation_plans/2025-11-11_plan-001-core-training-stabilization.md`
2. **Check Progress Tracker:** STATUS: Not Started, NEXT TASK: "Verify step function implementation"
3. **Execute Task 1.1:**
   - Use grep to locate `_step_function`: `grep -n "_step_function" ocr/models/head/db_head.py`
   - Read relevant section (lines 158-204)
   - Verify implementation
   - Run validation: `python -m py_compile ocr/models/head/db_head.py`
4. **Update Progress Tracker:**
   - Mark Task 1.1 as [x]
   - Update LAST COMPLETED TASK
   - Update NEXT TASK to Task 1.2
5. **Continue to next task**

---

*This prompt template is designed for Claude Code web workers executing implementation plans in a limited-context environment.*
