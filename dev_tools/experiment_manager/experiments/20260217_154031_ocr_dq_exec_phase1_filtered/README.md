# Experiment: ocr_dq_exec_phase1_filtered

**ID**: 20260217_154031_ocr_dq_exec_phase1_filtered
**Spec**: specs/003-ocr-data-quality-remediation/
**Status**: active
**Created**: 2026-02-17T15:40:31

## Intention

Phase 1 execution for OCR data-quality remediation using filtered manifests and gate metrics.
No mutations to source LMDB during this phase.

## Linked Artifacts

- Planning: specs/003-ocr-data-quality-remediation/planning/INDEX.md
- Tasks: specs/003-ocr-data-quality-remediation/tasks.md
- Runbook: specs/003-ocr-data-quality-remediation/EXECUTION_RUNBOOK.md
- Report: dev_tools/experiment_manager/experiments/20260217_150713_ocr_data_quality_remediation/.metadata/reports/20260218_0000_report_ocr-high-loss-data-quality-remediation-plan.md

## Data Source

- LMDB: data/processed/recognition/aihub_lmdb_validation/
- LMDB key schema: num-samples | image-{idx:09d} | label-{idx:09d} (1-indexed)
- state.json: pipeline processing tracker (processed_files, current_index) — not a manifest

## Audit Outputs

Generated under data/audit/ (read-only diagnostics):
- defect_prevalence.json
- loss_percentiles.json (label-length proxy until inference run)
- truncation_analysis.json

## Gate Status

See .metadata/00-status/ for gate snapshots.

## Rollback Triggers

- filtered_out_ratio > 0.40
- clean_holdout_cer_delta > 0.20
- annotation_kappa < 0.70
- upstage_call_ratio > 0.40 without approval
