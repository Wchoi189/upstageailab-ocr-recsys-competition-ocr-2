# Execution Runbook: OCR Data-Quality Remediation

## Purpose

Provide a deterministic, gate-driven run sequence for implementation sessions.

## Session Start Checklist

1. Confirm active experiment: `20260217_154031_ocr_dq_exec_phase1_filtered`
2. Confirm planning bundle reviewed: `specs/003-ocr-data-quality-remediation/planning/INDEX.md`
3. Confirm no pending planning edits before coding begins

## Phase A — Baseline Diagnostics (Read-only)

Run in this order:

```bash
python scripts/audit/analyze_defect_distribution.py \
  --run_id <run_id> \
  --output data/audit/defect_prevalence.json

python scripts/audit/compute_loss_distribution.py \
  --manifest data/processed/recognition/train_manifest.jsonl \
  --output data/audit/loss_percentiles.json

python scripts/audit/analyze_sequence_lengths.py \
  --manifest data/processed/recognition/train_manifest.jsonl \
  --tokenizer_max_len 25 \
  --output data/audit/truncation_analysis.json
```

Exit criteria:
- All three audit outputs generated
- No mutation to source manifests

## Phase B — Foundational Implementation

Implement in order:

1. `scripts/data/quality/manifest_io.py`
2. `scripts/data/quality/defect_rules.py`
3. `scripts/data/quality/quality_scoring.py`
4. `scripts/data/quality/gate_metrics.py`

Validation:
- All modules import cleanly
- Contract validation script passes for sample payloads

## Phase C — Reporting and Governance Linkage

1. Generate baseline report under `docs/reports/`
2. Register artifact path in experiment `manifest.json`
3. Update `.metadata/00-status/` with gate status

## Emergency Procedure

If corrupted GT learning is suspected during active training:

1. Stop run immediately
2. Backup current checkpoint to `outputs/checkpoints/emergency_backup/`
3. Record incident in experiment status note
4. Resume only after gate review

## Rollback Triggers

- Filtered-out ratio > 40%
- Clean-holdout CER degradation > 20%
- Annotation kappa < 0.70
- Upstage call ratio > 40% without approved rationale

## Evidence Required Per Gate

- Command log
- Input/output artifact paths
- Metrics snapshot
- Decision (`pass`, `hold`, `rollback`)
- Reviewer and timestamp
