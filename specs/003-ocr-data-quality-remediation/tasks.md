# Tasks: OCR High-Loss Data-Quality Remediation

**Input**: Design documents from `/specs/003-ocr-data-quality-remediation/`
**Prerequisites**: `plan.md` (required), `spec.md` (required), `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Test tasks are intentionally omitted (not explicitly requested in the specification). Validation is performed through phase gates and artifact checks.

**Organization**: Tasks are grouped by user story to enable independent implementation and validation of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm workspace readiness and set canonical planning artifacts.

- [x] T000 Verify experiment workspace exists at `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/`
- [x] T001 Create remediation workspace README in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/README.md`
- [x] T002 Create planning status tracker in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/00-status/2026-02-18_planning-status.md`
- [x] T003 [P] Create remediation config skeleton in `configs/data/quality/remediation.yaml`
- [x] T004 [P] Create artifact index for this feature in `specs/003-ocr-data-quality-remediation/ARTIFACT_INDEX.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build reusable quality modules and run non-mutating baseline diagnostics.

**⚠️ CRITICAL**: No user story execution work should begin until this phase is complete.

- [x] T005 Implement manifest I/O utilities in `scripts/data/quality/manifest_io.py`
- [x] T006 Implement defect rule engine in `scripts/data/quality/defect_rules.py`
- [x] T007 [P] Implement script mismatch heuristic in `scripts/data/quality/defect_rules.py`
- [x] T008 [P] Implement clipping risk heuristic in `scripts/data/quality/defect_rules.py`
- [x] T009 [P] Implement sample quality scoring utilities in `scripts/data/quality/quality_scoring.py`
- [x] T010 Implement gate metric calculator in `scripts/data/quality/gate_metrics.py`
- [x] T011 Implement non-mutating defect distribution analyzer in `scripts/audit/analyze_defect_distribution.py`
- [x] T012 [P] Implement non-mutating loss distribution analyzer in `scripts/audit/compute_loss_distribution.py`
- [x] T013 [P] Implement non-mutating sequence-length analyzer in `scripts/audit/analyze_sequence_lengths.py`
- [x] T014 Generate baseline diagnostics under `data/audit/`

**Checkpoint**: Foundational modules and baseline diagnostics are complete.

---

## Phase 3: User Story 1 - Publish High-Loss Technical Baseline (Priority: P1) 🎯 MVP

**Goal**: Produce a persistent and repeatable high-loss baseline report with complete defect taxonomy coverage and calibrated thresholds.

**Independent Test**: Review generated report artifact and verify all required defect classes, threshold calibration, and evidence summaries are present.

- [x] T015 [P] [US1] Implement high-loss sample export utility in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/export_high_loss_samples.py`
- [x] T016 [P] [US1] Implement defect taxonomy labeling utility in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/label_defect_classes.py`
- [x] T017 [US1] Implement baseline summary generator in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/build_data_quality_baseline_report.py`
- [x] T018 [US1] Add p95 threshold calibration section to report generation in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/build_data_quality_baseline_report.py`
- [x] T019 [US1] Create reusable audit report template in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/templates/ocr-data-quality-audit-template.md`
- [x] T020 [US1] Generate baseline report artifact in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/reports/20260218_1600_report_ocr-high-loss-baseline.md`
- [x] T021 [US1] Register baseline report in experiment metadata `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/manifest.json`

**Checkpoint**: High-loss baseline report is persisted with calibrated thresholds.

---

## Phase 4: User Story 2 - Define Planning-First Remediation Workflow (Priority: P1)

**Goal**: Establish an execution-ready workflow with explicit gates, rollback criteria, and clean holdout protocol.

**Independent Test**: Verify workflow artifacts define phases, entry/exit gates, rollback triggers, and required metrics without mutating training data.

- [x] T022 [P] [US2] Define canonical phase gate matrix in `specs/003-ocr-data-quality-remediation/planning/ocr-data-quality-phase-gates.md`
- [x] T023 [P] [US2] Define rollback triggers in `specs/003-ocr-data-quality-remediation/planning/ocr-data-quality-phase-gates.md`
- [x] T024 [P] [US2] Define metric thresholds and formulas in `specs/003-ocr-data-quality-remediation/planning/ocr-data-quality-metric-criteria.md`
- [x] T025 [US2] Add clean holdout construction protocol in `specs/003-ocr-data-quality-remediation/planning/ocr-clean-holdout-protocol.md`
- [x] T026 [US2] Add annotation quality assurance protocol in `specs/003-ocr-data-quality-remediation/planning/ocr-annotation-qa-protocol.md`
- [x] T027 [US2] Add emergency stop and checkpoint preservation runbook in `specs/003-ocr-data-quality-remediation/EXECUTION_RUNBOOK.md`
- [x] T028 [US2] Create next-session handoff checklist in `specs/003-ocr-data-quality-remediation/SESSION_HANDOVER.md`

**Checkpoint**: Workflow governance is explicit, measurable, and rollback-safe.

---

## Phase 5: User Story 3 - Initialize Controlled Experiment Workspace (Priority: P2)

**Goal**: Ensure experiment workspace is reproducible, self-describing, and linked to planning/report artifacts.

**Independent Test**: Confirm workspace metadata and artifact links can be audited without additional context.

- [x] T029 [P] [US3] Create experiment bootstrap helper in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/experiment/init_ocr_data_quality_experiment.sh`
- [x] T030 [US3] Add experiment operating guide in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/guides/20260218_1900_guide_experiment-operations.md`
- [x] T031 [US3] Add artifact linkage report in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/reports/20260218_1900_report_artifact-linkage-audit.md`
- [x] T032 [US3] Record workflow tasks and linked artifacts in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/manifest.json`

**Checkpoint**: Controlled experiment workspace is fully linked and reproducible.

---

## Phase 6: User Story 4 - Tiered Golden Validation Strategy (Priority: P2)

**Goal**: Define and implement a cost-aware golden validation pipeline using local triage (PaddleOCR + Ollama) and Upstage OCR for ambiguous samples.

**Independent Test**: Run golden-set pipeline on pilot batch and produce categorized outputs (`auto_accept`, `auto_correct`, `manual_review`) with confidence provenance.

- [x] T033 [P] [US4] Implement Upstage OCR client module in `scripts/data/quality/upstage_validator.py`
- [x] T034 [P] [US4] Implement PaddleOCR local validator wrapper in `scripts/data/quality/paddle_validator.py`
- [x] T035 [US4] Implement tiered validation orchestrator in `scripts/data/quality/golden_set_validator.py`
- [x] T036 [US4] Implement golden holdout builder in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/create_golden_holdout_with_upstage.py`
- [x] T037 [US4] Add synthetic data generation target spec in `specs/003-ocr-data-quality-remediation/planning/ocr-synthetic-data-spec.md`
- [x] T038 [US4] Document tiered golden validation policy in `specs/003-ocr-data-quality-remediation/planning/ocr-golden-validation-strategy.md`

**Checkpoint**: Tiered golden validation pipeline is documented and execution-ready.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final consistency, compliance, and handoff quality checks.

- [x] T039 [P] Validate feature artifact naming and structure in `specs/003-ocr-data-quality-remediation/tasks.md`
- [x] T040 [P] Reconcile experiment artifact registry in `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/manifest.json`
- [x] T041 Create final planning summary for execution session in `docs/reports/2026-02-19_ocr-data-quality-remediation-execution-ready.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies, starts immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 completion, blocks all story work
- **Phases 3-6 (User Stories)**: Depend on Phase 2 completion
- **Phase 7 (Polish)**: Depends on completion of all targeted user stories

### User Story Dependencies

- **US1 (P1)**: Starts after Foundational; independent baseline artifact delivery
- **US2 (P1)**: Starts after Foundational; independent workflow governance delivery
- **US3 (P2)**: Depends on US1/US2 outputs for artifact linkage completeness
- **US4 (P2)**: Depends on US2 clean holdout protocol and threshold policy

---

## Parallel Execution Examples

### Foundational Diagnostics
- Run in parallel: `T011`, `T012`, and `T013`
- Then run sequentially: `T014`

### User Story 1
- Run in parallel: `T015` and `T016`
- Then run sequentially: `T017` → `T018` → `T019` → `T020` → `T021`

### User Story 2
- Run in parallel: `T022`, `T023`, and `T024`
- Then run sequentially: `T025` → `T026` → `T027` → `T028`

### User Story 4
- Run in parallel: `T033` and `T034`
- Then run sequentially: `T035` → `T036` → `T037` → `T038`

---

## Implementation Strategy

### MVP First (User Story 1)
1. Complete Phase 1 and Phase 2
2. Complete Phase 3 (US1) and validate independent test criteria
3. Publish baseline artifact and calibrated thresholds before execution runs

### Incremental Delivery
1. Deliver US1 baseline + threshold calibration
2. Deliver US2 phase gates + rollback + clean holdout protocol
3. Deliver US3 experiment reproducibility and lineage
4. Deliver US4 tiered golden validation strategy (Upstage + PaddleOCR)
5. Run Phase 7 polish checks and mark execution-ready state
