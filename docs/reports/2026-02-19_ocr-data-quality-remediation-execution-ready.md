# OCR Data-Quality Remediation — Execution Ready Summary

**Date**: 2026-02-19
**Feature**: `003-ocr-data-quality-remediation`
**Experiment**: `20260217_154031_ocr_dq_exec_phase1_filtered`
**Status**: PLANNING COMPLETE — Ready for Gate 2 Execution

---

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| Gate 0 — Baseline Diagnostic Readiness | **PASS** | `data/audit/defect_prevalence.json`, `data/audit/loss_percentiles.json`, `data/audit/truncation_analysis.json` |
| Gate 1 — Policy Design Readiness | **PASS** | US2 planning artifacts complete (T022–T028) |
| Gate 2 — Training Run Baseline | PENDING | Requires: clean holdout v1, model eval run |
| Gate 3 — Annotation Quality | PENDING | Requires: annotator kappa ≥ 0.80 |
| Gate 4 — Synthetic Data Generation | PENDING | Blockers: TRDG env ✓, PaddleOCR ✓, fonts ✓ |
| Gate 4.5 — Upstage API Ratio | PENDING | Upstage call ratio target: 20–40% of candidates |
| Gate 5 — Production Readiness | PENDING | Requires Gate 2–4.5 sequential PASS |

---

## Completed User Stories

### US1 — High-Loss Baseline (T015–T021) — COMPLETE
- Audit artifacts generated (immutable): `data/audit/defect_prevalence.json`, `high_loss_samples.json`, `defect_taxonomy.json`
- Loss proxy: `label_len/max_len`, p95=0.32, high_loss=88,257 samples
- Baseline report: `<exp>/.metadata/reports/20260218_1600_report_ocr-high-loss-baseline.md`

### US2 — Remediation Workflow Governance (T022–T028) — COMPLETE
- Phase gate matrix + rollback triggers: `specs/.../planning/ocr-data-quality-phase-gates.md`
- Metric formulas + thresholds: `specs/.../planning/ocr-data-quality-metric-criteria.md`
- Clean holdout protocol: `specs/.../planning/ocr-clean-holdout-protocol.md`
- Annotation QA protocol: `specs/.../planning/ocr-annotation-qa-protocol.md`
- Emergency stop + checkpoint runbook: `specs/.../EXECUTION_RUNBOOK.md`

### US3 — Controlled Experiment Workspace (T029–T032) — COMPLETE
- Bootstrap check: `<exp>/scripts/experiment/init_ocr_data_quality_experiment.sh` — PASS (0 failures)
- Operating guide: `<exp>/.metadata/guides/20260218_1900_guide_experiment-operations.md`
- Linkage audit: `<exp>/.metadata/reports/20260218_1900_report_artifact-linkage-audit.md` — PASS (22/22)
- Manifest: `<exp>/manifest.json` — 28 artifacts, 18 tasks registered

### US4 — Tiered Golden Validation (T033–T038) — COMPLETE
- Upstage API client: `scripts/data/quality/upstage_validator.py`
- PaddleOCR wrapper: `scripts/data/quality/paddle_validator.py`
- Tiered orchestrator: `scripts/data/quality/golden_set_validator.py`
- Holdout builder CLI: `<exp>/scripts/analysis/create_golden_holdout_with_upstage.py`
- Synthetic data spec: `specs/.../planning/ocr-synthetic-data-spec.md`
- Validation strategy policy: `specs/.../planning/ocr-golden-validation-strategy.md`

---

## Environment Readiness

| Dependency | Status | Version / Detail |
|---|---|---|
| Korean fonts (NanumGothic, NanumMyeongjo, Noto) | CLEARED | 54 fonts at `/usr/share/fonts/truetype/` |
| TRDG (text image generation) | CLEARED | v1.8.0 from local source |
| PaddleOCR | CLEARED | v2.10.0 + paddlepaddle v3.1.1 |
| Upstage OCR API | READY (key required) | Set `UPSTAGE_API_KEY` env var |

---

## Locked Configuration

- `unreadable_min_len=0` — LOCKED. Single-char Korean syllables are valid.
- Tokenizer `max_len=25`; 2,247 truncated; 1 outlier at len=299
- 3-tier validation: Tier-1 model score → Tier-2 PaddleOCR CER → Tier-3 Upstage API
- Confidence policy: `≥0.95 + CER≤0.05` → auto_accept; `≥0.98 + CER>0.05` → auto_correct; else → manual_review
- Holdout versioning: `vN` suffix; approved versions immutable
- Gate evidence required: input paths, metrics snapshot, decision, reviewer, timestamp

---

## Open Risks (Pre-Execution)

| ID | Risk | Mitigation |
|---|---|---|
| RISK-01 | CTC loss proxy vs real inference gap | Measure per-sample loss after Gate 2 model eval |
| RISK-02 | `script_mismatch` threshold 0.30 calibration | Manual review of 50 examples pending |
| RISK-03 | len=299 outlier — source unknown | Identify and recommend exclusion |

---

## First Execution Actions (Gate 2 Entry)

1. Construct clean holdout v1:
   ```
   uv run python dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/create_golden_holdout_with_upstage.py \
     --candidates <path> --version 1 --dry_run
   ```
2. Set `UPSTAGE_API_KEY` and remove `--dry_run` for live Gate 4.5 pilot
3. Run training baseline with clean holdout (Gate 2 entry condition)
4. Evaluate model on holdout → per-sample CTC loss → resolve RISK-01

---

## Key Guardrails

- No training data mutation before Gate 2 PASS
- Upstage API calls must remain within 20–40% of candidate pool
- All holdout changes create new versioned artifact; never overwrite
