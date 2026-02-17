# Tasks: High-Loss Recognition Image Audit + Logging Stability

**Input**: `/workspaces/specs/001-wandb-config-logging/plan.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`
**Mode**: Phase-by-phase implementation with strict behavior (no shims, no fallback wrappers)

---

## Phase 1: Runtime Stability (Completed)

**Goal**: Restore deterministic WandB recognition image logging path and remove implicit compatibility behavior.

- [x] T001 Verify feature prerequisites and checklist gates
- [x] T002 Identify recognition logging disconnect in runtime gate logic
- [x] T003 Enforce strict WandB-run requirement for recognition image logging
- [x] T004 Remove shim/legacy compatibility files and fallback code paths
- [x] T005 Smoke-test strict recognition image logging path

**Checkpoint**: Recognition image logging succeeds only when a real WandB logger/run is attached.

---

## Phase 2: Render Quality (Completed)

**Goal**: Fix Korean text rendering and image size usability in WandB media panels.

- [x] T006 Implement Korean-capable font discovery and glyph validation
- [x] T007 Enforce strict failure when no Hangul-capable font is available
- [x] T008 Install and verify Nanum font in runtime environment
- [x] T009 Add recognition image max-side cap and wire from config
- [x] T010 Smoke-test caption rendering and media upload after scaling changes

**Checkpoint**: Korean captions render correctly and validation images are not oversized.

---

## Phase 3: High-Loss Audit Core (In Progress)

**Goal**: Implement bounded top-K high-loss sample auditing for recognition validation.

- [x] T011 Add `high_loss_audit` config block to `configs/train/logger/wandb.yaml`
- [x] T012 Compute/collect per-sample validation loss in recognition path
- [x] T013 Maintain epoch-level bounded top-K buffer (`HighLossSample`)
- [x] T014 Log `audit/high_loss_samples` image panel at validation epoch end
- [x] T015 Log optional `audit/high_loss_table` with loss/gt/pred/filename metadata

**Checkpoint**: Each validation epoch emits bounded, interpretable high-loss artifacts.

---

## Phase 4: Validation & Hardening (Not Started)

**Goal**: Validate correctness, limits, and failure semantics under strict mode.

- [x] T016 Unit test: top-K selection and finite-loss filtering
- [x] T017 Unit test: strict failure when `log_recognition_images=true` and no WandB run
- [x] T018 Integration smoke run with `high_loss_audit.enabled=true`
- [x] T019 Verify per-epoch upload count never exceeds configured cap
- [ ] T020 Verify no regression in `val/acc`, `val/cer`, `val_loss` logging

**Checkpoint**: Feature is stable and bounded under real training conditions.

---

## Phase 5: Documentation & Handover (Not Started)

**Goal**: Keep operator documentation aligned with strict runtime behavior.

- [x] T021 Update `quickstart.md` with strict font + image-size settings
- [x] T022 Update troubleshooting with explicit fail-fast error messages
- [x] T023 Record final run evidence and links in report notes

**Checkpoint**: Team can reproduce, configure, and debug behavior quickly.

---

## Execution Rules

1. Complete one phase checkpoint before moving to the next.
2. Mark tasks `[x]` immediately after completion.
3. Keep strict semantics: no silent fallback, no legacy shim files.
4. Run targeted smoke checks after each phase.
