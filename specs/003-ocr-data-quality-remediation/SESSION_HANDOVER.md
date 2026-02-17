# Session Handover: OCR Data-Quality Remediation

## Objective for Next Session

Start implementation immediately from Phase 2 (Foundational), using finalized planning artifacts and gate definitions.

## Locked Inputs

- Spec: `specs/003-ocr-data-quality-remediation/spec.md`
- Tasks: `specs/003-ocr-data-quality-remediation/tasks.md`
- Planning index: `specs/003-ocr-data-quality-remediation/planning/INDEX.md`
- Experiment: `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/`

## Start Here (First 30 Minutes)

1. Run baseline diagnostics (see `EXECUTION_RUNBOOK.md`)
2. Commit generated audit artifacts under `data/audit/`
3. Implement `scripts/data/quality/manifest_io.py`
4. Add strict contract checks in `scripts/data/quality/contracts.py`

## Priority Task Sequence

- T005 -> T006 -> T009 -> T010 -> T011/T012/T013 -> T014

## Guardrails

- No hidden fallback behavior in validation/observability paths
- Keep patch-level validation (no full-document substitution)
- Enforce explicit provenance fields for holdout records
- Keep Upstage escalation within cost envelope (target 20%-40%)

## Definition of Ready for Training

- Foundational scripts implemented
- Gate metrics computable from generated artifacts
- Holdout protocol + annotation QA fields validated by contract models
- Experiment manifest updated with report lineage

## Open Technical Risks to Monitor

- Hydra merge anomalies from package directives
- DictConfig serialization leakage across boundaries
- Multiprocessing/CUDA mode drift in training entrypoint
- Schema drift between planning docs and runtime payloads
