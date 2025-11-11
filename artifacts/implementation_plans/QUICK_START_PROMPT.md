# Quick Start Prompt for Claude Code Web

## Copy-Paste Ready Prompt

Copy this prompt and replace `[PLAN-00X]` and `[plan-name]` with the specific plan you want to execute:

---

```
You are an autonomous AI agent executing an implementation plan for an OCR system architecture refactoring.

## Your Task
Execute the implementation plan located at:
`artifacts/implementation_plans/2025-11-11_plan-00X-[plan-name].md`

## Execution Rules

### 1. Read the Plan First
- Read the entire plan document to understand the task
- Check the Progress Tracker section for current status and next task
- Review risk level and mitigation strategies

### 2. Context Management (CRITICAL)
You have limited context. Follow these rules:
- Read ONLY 1-2 files at a time
- Use grep/search FIRST to locate patterns before reading files
- Read only relevant sections (use line offsets)
- Estimated token budget: ~2000-3000 per task

### 3. Execution Workflow
For each task:
1. Read the task description in "Implementation Outline"
2. Check "NEXT TASK" in Progress Tracker
3. Execute the task step by step
4. Validate immediately using commands below
5. Update Progress Tracker after completion

### 4. Validation (No Runtime Required)
After each change, run:
```bash
# Syntax check
python -m py_compile <file>

# Import check
python -c "from <module> import <class>"

# Pattern search (before reading)
grep -rn "<pattern>" <directory>
```

### 5. Progress Tracking
After each task:
- Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
- Mark completed checkboxes with [x]
- Document any discoveries or blockers

### 6. Commit Strategy
- One logical change per commit
- Clear commit messages: `[PLAN-00X] <description>`
- Commit frequently for easy rollback

### 7. Error Handling
If validation fails:
- Review error message
- Fix syntax/import issue
- Re-run validation
- Document fix in comments

## Starting Now
1. Read the plan: `artifacts/implementation_plans/2025-11-11_plan-00X-[plan-name].md`
2. Check Progress Tracker for current status
3. Begin with the task listed in "NEXT TASK"
4. Execute step by step, validating after each change

## Important Constraints
- No runtime testing (syntax/import validation only)
- Limited context (1-2 files at a time)
- Atomic changes (one logical change per commit)
- Progress tracking required (update plan after each task)

Begin execution now. Start by reading the plan document.
```

---

## Plan-Specific Quick Starts

### PLAN-001 (Critical Risk)
```
Execute PLAN-001: Core Training Stabilization.

Plan: artifacts/implementation_plans/2025-11-11_plan-001-core-training-stabilization.md

This is CRITICAL risk. Focus on validation and verification since most fixes are already implemented.
Start by reading the plan and checking the Progress Tracker.
```

### PLAN-002 (High Risk)
```
Execute PLAN-002: Polygon Validation Consolidation.

Plan: artifacts/implementation_plans/2025-11-11_plan-002-polygon-validation-consolidation.md

This is HIGH risk. Use grep first to locate validation patterns before reading files.
Start by reading the plan and checking the Progress Tracker.
```

### PLAN-003 (Medium Risk)
```
Execute PLAN-003: Import-Time Optimization.

Plan: artifacts/implementation_plans/2025-11-11_plan-003-import-time-optimization.md

This is MEDIUM risk. Use grep to locate import patterns before reading files.
Start by reading the plan and checking the Progress Tracker.
```

### PLAN-004 (Very High Risk)
```
Execute PLAN-004: Inference Service Consolidation.

Plan: artifacts/implementation_plans/2025-11-11_plan-004-inference-service-consolidation.md

This is VERY HIGH risk. Work carefully - this affects UI functionality.
Make atomic changes and validate frequently.
Start by reading the plan and checking the Progress Tracker.
```

### PLAN-005 (Low Risk)
```
Execute PLAN-005: Legacy Cleanup & Config Consolidation.

Plan: artifacts/implementation_plans/2025-11-11_plan-005-legacy-cleanup-config-consolidation.md

This is LOW risk. Use terminal commands (ls, mv, rm) and grep for reference searches.
Start by reading the plan and checking the Progress Tracker.
```

---

## Recommended Execution Order

1. **PLAN-001** (must complete first - blocks everything)
2. **PLAN-002** (depends on PLAN-001)
3. **PLAN-003** + **PLAN-005** (can run in parallel)
4. **PLAN-004** (last, highest risk)

---

*Ready to use with Claude Code web. Just copy and paste!*

