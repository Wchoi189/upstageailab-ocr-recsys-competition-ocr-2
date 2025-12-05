---
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['refactoring', 'train', 'logger', 'callbacks']
title: "Phase 4 Refactoring Plan: Reducing Redundancy and Complexity"
date: "2025-12-05 23:07 (KST)"
---

# Master Prompt (use verbatim to start)
"Run Phase 4 refactor: remove redundancy, factor shared helpers, keep imports lean, fail fast. Keep intermediate summaries ultra-short; final summary at end."

# Living Implementation Blueprint: Phase 4 Refactoring Plan

## Progress Tracker
- **STATUS:** Completed
- **CURRENT STEP:** Phase 4 complete
- **LAST COMPLETED TASK:** Phase 4, Task 1.5 - Lean validation & docs touch-up
- **NEXT TASK:** None (ready to close)

### Implementation Outline (Checklist)

#### Phase 4: Redundancy Reduction (single sprint)
1. [x] **Task 1.1: Extract output-dir helper**
   - [x] Add `ensure_output_dirs(paths)` to `ocr/utils/path_utils.py`
   - [x] Replace inline dir creation in `runners/train.py`
   - [x] Keep fail-fast behavior (no broad except)

2. [x] **Task 1.2: Factor callback builder**
   - [x] Create `build_callbacks(config)` (local helper or `ocr/utils/callbacks.py`)
   - [x] Move Hydra instantiation loop out of `train.py`
   - [x] Preserve resolved-config attachment for checkpoint callbacks

3. [x] **Task 1.3: Unify logger usage across runners**
   - [x] Apply `create_logger` in other runners (`runners/test.py`, `runners/predict.py` if present)
   - [x] Remove duplicated logger logic
   - [x] Keep lazy-import discipline

4. [x] **Task 1.4: Import and redundancy audit**
   - [x] Remove unused imports; push heavy/optional imports inside functions
   - [x] Confirm no residual broad `except Exception`
   - [x] Check for repeated logic to inline or centralize
   - [ ] Legacy note: deprecated `PathUtils` helpers remain; schedule removal once callers are migrated

5. [x] **Task 1.5: Lean validation & docs touch-up**
   - [x] Run smoke test `uv run python runners/train.py trainer.max_epochs=1 trainer.limit_train_batches=0.01 trainer.limit_val_batches=0.01 exp_name=phase4_test logger.wandb.enabled=false`
   - [x] Ruff + mypy on touched files
   - [x] Update notes with a final summary only; keep intermediate summaries ultra short

## 📋 Technical Requirements Checklist (keep terse)
- Reuse shared helpers; avoid duplication.
- Maintain lazy imports; no heavy imports at module top unless required.
- Fail fast; no broad `except Exception`.
- Respect Hydra config structure; data contracts live in `architecture/` if needed.

## 🎯 Success Criteria Validation
- Logger, callbacks, and dir setup logic are centralized and reused.
- `runners/train.py` shrinks further (fewer responsibilities, clearer flow).
- No duplicated logger or callback code in other runners.
- Tests, linting, and smoke run succeed.

## 📊 Risk Mitigation & Fallbacks (short)
- Risk: regression from helper extraction → Mitigate with smoke test + minimal diff per step.
- Risk: hidden import side effects → Keep imports lazy; revert specific helper if needed.
- Risk: config differences across runners → Validate logger/callback defaults after reuse.

## 🔄 Blueprint Update Protocol (concise)
- After each task: mark checklist, note blockers, keep intermediate summary to one line.
- Final: one concise summary at end of Phase 4.

## 🚀 Immediate Next Action
- **Task:** Run final validation + smoke for touched areas (Phase 4 Task 1.5).
- **Objective:** Ensure refactor passes lint/type checks and smoke train.
- **Approach:**
  1) Re-run ruff + mypy on touched files.
  2) Run smoke train command in plan.
  3) Capture brief final summary and note legacy removal follow-up (`PathUtils`).
- **Success Criteria:** lint/type checks clean, smoke run completes, notes updated with concise summary.

## Final Summary
Centralized dir setup (`ensure_output_dirs`), callback building (`build_callbacks`), and logger creation (`create_logger`) across runners with lazy imports and narrowed exception handling; legacy `PathUtils` remains marked for removal; lint/mypy and smoke train all pass.

## Follow-Up Work
Created follow-up plan for legacy code removal: `2025-12-05_2353_implementation_plan_pathutils-removal.md`

This plan addresses the critical need to remove deprecated `PathUtils` class and standalone helpers to prevent AI agent confusion and architecture drift (historical issue where agents updated dead code instead of canonical implementations).
