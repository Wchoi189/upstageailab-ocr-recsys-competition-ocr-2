# Session Handover: OCR Data-Quality Remediation

**LATEST**: `specs/003-ocr-data-quality-remediation/2026-02-18T16_SESSION_HANDOVER.md`

## Objective for Next Session

Start Phase 3 (US1: Baseline Report) and Phase 4 (US2: Workflow Governance) after calibrating defect thresholds.

## Immediate First Actions

1. Calibrate unreadable_sample threshold (see Critical Calibration below)
2. Implement T015+T016 in parallel (US1 report utilities)
3. Implement T022+T023+T024 in parallel (US2 phase gates)

## Critical Calibration Required

Change `configs/data/quality/remediation.yaml`:
```yaml
defect_rules:
  unreadable_min_len: 0  # was 1 — single-char Korean syllable labels are valid
```
Then re-run:
```
uv run python scripts/audit/analyze_defect_distribution.py \
  --lmdb_path data/processed/recognition/aihub_lmdb_validation \
  --output data/audit/defect_prevalence.json
```

## Gate 0 Status: PASS

| Artifact | Path |
|---|---|
| defect_prevalence | data/audit/defect_prevalence.json |
| loss_percentiles | data/audit/loss_percentiles.json |
| truncation_analysis | data/audit/truncation_analysis.json |

## Priority Task Sequence

T015 → T016 → T017 → T018 → T019 → T020 → T021 (US1)
PARALLEL: T022, T023, T024 (US2)

## Guardrails

- No hidden fallback behavior in validation/observability paths
- Keep patch-level validation (no full-document substitution)
- Enforce explicit provenance fields for holdout records
- Keep Upstage escalation within cost envelope (target 20%-40%)
- unreadable_min_len calibration required before defect_purity gate is reliable

## Definition of Ready for Training

- Foundational scripts implemented ✅
- Gate metrics computable from generated artifacts ✅
- Holdout protocol + annotation QA fields validated by contract models
- Experiment manifest updated with report lineage

## Open Technical Risks to Monitor

- unreadable_sample threshold may over-flag valid single-char Korean samples (18.48% flag rate)
- Labels with len>25 (max=299 found) silently truncated — need investigation
- Loss data is proxy-only until inference run
- Hydra merge anomalies from package directives
- DictConfig serialization leakage across boundaries
- Schema drift between planning docs and runtime payloads

## Locked Inputs

- Spec: `specs/003-ocr-data-quality-remediation/spec.md`
- Tasks: `specs/003-ocr-data-quality-remediation/tasks.md`
- Planning: `specs/003-ocr-data-quality-remediation/planning/INDEX.md`
- Experiment: `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/`
- Audit artifacts: `data/audit/` (all three generated)
