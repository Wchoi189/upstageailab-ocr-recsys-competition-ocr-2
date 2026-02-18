# Artifact Linkage Audit Report
**Experiment**: `20260217_154031_ocr_dq_exec_phase1_filtered`
**Date**: 2026-02-18T19
**Scope**: All artifacts registered in `manifest.json` (US1 + US2 + US3)
**Purpose**: Verify every registered artifact exists on disk and that path schema is consistent.

---

## 1. Audit Results

### Path Root Convention (Manifest Schema)

The manifest uses **two different root conventions**:

| Path Prefix | Root | Example |
|---|---|---|
| `experiments/...` | `dev_tools/experiment_manager/` | `experiments/20260217.../scripts/analysis/export_high_loss_samples.py` |
| `data/...` | Repo root (`/workspaces`) | `data/audit/defect_prevalence.json` |
| `specs/...` | Repo root (`/workspaces`) | `specs/003-ocr-data-quality-remediation/planning/...` |
| `scripts/...` | Repo root (`/workspaces`) | `scripts/data/quality/defect_rules.py` |
| `configs/...` | Repo root (`/workspaces`) | `configs/data/quality/remediation.yaml` |

**This dual-root convention is a schema inconsistency.** All `experiments/` paths implicitly require knowledge of the `etk_base`. Resolution logic must be applied by all consumers.

`path_roots` field added to `manifest.json` (T032) to make convention explicit.

---

## 2. Artifact Existence Verification

| Status | Type | Path (manifest) | Resolved Path |
|---|---|---|---|
| OK | report | `experiments/20260217_150713_ocr_data_quality_remediation/.metadata/reports/20260218_0000_report_ocr-high-loss-data-quality-remediation-plan.md` | `dev_tools/experiment_manager/…` |
| OK | audit | `data/audit/defect_prevalence.json` | `/workspaces/data/audit/defect_prevalence.json` |
| OK | audit | `data/audit/defect_taxonomy.json` | `/workspaces/data/audit/defect_taxonomy.json` |
| OK | audit | `data/audit/high_loss_samples.json` | `/workspaces/data/audit/high_loss_samples.json` |
| OK | report | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/reports/20260218_1600_report_ocr-high-loss-baseline.md` | `dev_tools/experiment_manager/…` |
| OK | template | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/templates/ocr-data-quality-audit-template.md` | `dev_tools/experiment_manager/…` |
| OK | script | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/export_high_loss_samples.py` | `dev_tools/experiment_manager/…` |
| OK | script | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/label_defect_classes.py` | `dev_tools/experiment_manager/…` |
| OK | script | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/build_data_quality_baseline_report.py` | `dev_tools/experiment_manager/…` |
| OK | handover | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/20260218_1600_SESSION_HANDOVER.md` | `dev_tools/experiment_manager/…` |
| OK | handover | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/20260218_1700_SESSION_HANDOVER.md` | `dev_tools/experiment_manager/…` |
| OK | planning | `specs/003-ocr-data-quality-remediation/planning/ocr-data-quality-phase-gates.md` | `/workspaces/specs/…` |
| OK | planning | `specs/003-ocr-data-quality-remediation/planning/ocr-data-quality-metric-criteria.md` | `/workspaces/specs/…` |
| OK | planning | `specs/003-ocr-data-quality-remediation/planning/ocr-clean-holdout-protocol.md` | `/workspaces/specs/…` |
| OK | planning | `specs/003-ocr-data-quality-remediation/planning/ocr-annotation-qa-protocol.md` | `/workspaces/specs/…` |
| OK | runbook | `specs/003-ocr-data-quality-remediation/EXECUTION_RUNBOOK.md` | `/workspaces/specs/…` |
| OK | handover | `specs/003-ocr-data-quality-remediation/SESSION_HANDOVER.md` | `/workspaces/specs/…` |
| OK | handover | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/20260218_1800_SESSION_HANDOVER.md` | `dev_tools/experiment_manager/…` |
| OK | script | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/experiment/init_ocr_data_quality_experiment.sh` | `dev_tools/experiment_manager/…` |
| OK | guide | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/guides/20260218_1900_guide_experiment-operations.md` | `dev_tools/experiment_manager/…` |
| OK | report | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/reports/20260218_1900_report_artifact-linkage-audit.md` | `dev_tools/experiment_manager/…` (this file) |
| OK | handover | `experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/20260218_1900_SESSION_HANDOVER.md` | `dev_tools/experiment_manager/…` |

**Result: 22/22 artifacts present on disk.**

---

## 3. Schema Finding

**FINDING-01**: Manifest paths use mixed root conventions (`etk_base` vs repo root).

**Remediation applied**: `path_roots` field added to `manifest.json` (T032). Dual-root convention is now explicit and documented.

---

## 4. Audit Summary

| Check | Result |
|---|---|
| Total registered artifacts | 22 |
| Artifacts on disk | 22 (100%) |
| Missing artifacts | 0 |
| Schema anomalies | 1 (dual-root convention — documented, low severity) |

**LINKAGE AUDIT: PASS**
