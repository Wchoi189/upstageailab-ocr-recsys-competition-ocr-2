# Experiment Operations Guide
**Experiment**: `20260217_154031_ocr_dq_exec_phase1_filtered`
**Feature**: `003-ocr-data-quality-remediation`
**Created**: 2026-02-18T19

---

## 1. Session Start Protocol

```bash
# 1. Validate workspace (non-mutating)
bash dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/experiment/init_ocr_data_quality_experiment.sh

# 2. Read latest handover
cat dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/$(
  ls -1t dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/*_SESSION_HANDOVER.md | head -1 | xargs basename
)

# 3. Check planning index
cat specs/003-ocr-data-quality-remediation/planning/INDEX.md
```

---

## 2. Directory Map

```
dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/
├── manifest.json                      # Canonical artifact + gate registry
├── scripts/
│   ├── analysis/                      # US1 analysis scripts (T015-T017)
│   │   ├── export_high_loss_samples.py
│   │   ├── label_defect_classes.py
│   │   └── build_data_quality_baseline_report.py
│   └── experiment/                    # US3 session scripts (T029)
│       └── init_ocr_data_quality_experiment.sh
└── .metadata/
    ├── 00-status/                     # Gate status snapshots
    │   └── 2026-02-18_planning-status.md
    ├── guides/                        # This file — session ops guides
    │   └── 20260218_1900_guide_experiment-operations.md
    ├── reports/                       # Phase reports (timestamped)
    │   ├── 20260218_1600_report_ocr-high-loss-baseline.md
    │   └── 20260218_1900_report_artifact-linkage-audit.md
    ├── templates/                     # Reusable audit templates
    │   └── ocr-data-quality-audit-template.md
    └── *_SESSION_HANDOVER.md         # Timestamped handovers (latest = active)
```

**Shared infrastructure** (project root — reusable across features, NOT session artifacts):
```
scripts/
├── data/quality/
│   ├── manifest_io.py
│   ├── defect_rules.py
│   ├── quality_scoring.py
│   └── gate_metrics.py
└── audit/
    ├── analyze_defect_distribution.py
    ├── compute_loss_distribution.py
    └── analyze_sequence_lengths.py

specs/003-ocr-data-quality-remediation/
├── planning/
│   ├── ocr-data-quality-phase-gates.md       # Gate matrix + rollback triggers
│   ├── ocr-data-quality-metric-criteria.md   # Metric formulas + thresholds
│   ├── ocr-clean-holdout-protocol.md
│   └── ocr-annotation-qa-protocol.md
├── EXECUTION_RUNBOOK.md
└── SESSION_HANDOVER.md                        # Live pointer to latest handover

data/audit/                                    # Generated diagnostics (immutable)
├── defect_prevalence.json
├── loss_percentiles.json
├── truncation_analysis.json
├── defect_taxonomy.json
└── high_loss_samples.json

configs/data/quality/remediation.yaml          # Locked config (unreadable_min_len=0)
```

---

## 3. Gate Reference

| Gate | Status | Entry Condition | Decision |
|---|---|---|---|
| Gate 0 — Baseline Diagnostics | **PASS** | — | 3 audit artifacts generated |
| Gate 1 — Policy Design | **PASS** | Gate 0 | US2 planning docs complete |
| Gate 2 — Filter Stage | pending | Gate 1 | filtered_out_ratio <= 40%; CER delta <= +20% |
| Gate 3 — Correction Queue | pending | Gate 2 | kappa >= 0.70; correction_ratio <= 30% |
| Gate 4 — Synthetic Aug | pending | Gate 3 | discriminator accuracy <= 70% |
| Gate 4.5 — Golden Validation | pending | Gate 4 | Upstage call ratio 20–40%; provenance 100% |
| Gate 5 — Execution Approval | pending | Gates 2–4 | Summary report + manifest lineage + go/no-go |

Gate thresholds: `specs/003-ocr-data-quality-remediation/planning/ocr-data-quality-phase-gates.md`
Metric formulas: `specs/003-ocr-data-quality-remediation/planning/ocr-data-quality-metric-criteria.md`

---

## 4. Running Diagnostics (Non-Mutating)

```bash
# Defect distribution (LMDB direct — no jsonl intermediates)
uv run python scripts/audit/analyze_defect_distribution.py \
  --lmdb_path data/processed/recognition/aihub_lmdb_validation \
  --output data/audit/defect_prevalence.json

# Loss distribution (proxy mode)
uv run python scripts/audit/compute_loss_distribution.py \
  --manifest data/processed/recognition/train_manifest.jsonl \
  --output data/audit/loss_percentiles.json

# Sequence length / truncation analysis
uv run python scripts/audit/analyze_sequence_lengths.py \
  --manifest data/processed/recognition/train_manifest.jsonl \
  --tokenizer_max_len 25 \
  --output data/audit/truncation_analysis.json
```

**Constraint**: Do NOT pass `--mutate` or modify any `data/processed/` paths until Gate 2 approved.

---

## 5. Locked Configuration

| Setting | Value | Rationale |
|---|---|---|
| `unreadable_min_len` | `0` | Single-char Korean syllables are valid data |
| `loss_mode` | `label_length_proxy` | Actual CTC inference pending |
| `high_loss_threshold` | `p95 = 0.32` | Calibrated from label_len/max_len proxy |
| `tokenizer_max_len` | `25` | Frozen — do not change without gate review |

Config file: `configs/data/quality/remediation.yaml`

**Any threshold change** requires: documented rationale + effective date in next phase report.

---

## 6. Emergency Stop Procedure

If active training with suspected corrupted GT:

```bash
# 1. Interrupt training
kill -INT <training_pid>

# 2. Preserve checkpoint
cp -r outputs/checkpoints/latest/ outputs/checkpoints/emergency_backup/$(date +%Y%m%d_%H%M)/

# 3. Record incident
echo "[$(date -Iseconds)] EMERGENCY STOP: <reason>" >> \
  dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/.metadata/00-status/emergency_log.txt
```

Full procedure: `specs/003-ocr-data-quality-remediation/EXECUTION_RUNBOOK.md`

---

## 7. Artifact Naming Convention

| Artifact Type | Pattern | Location |
|---|---|---|
| Session handover | `YYYYMMDD_HHMM_SESSION_HANDOVER.md` | `<experiment_dir>/.metadata/` |
| Phase reports | `YYYYMMDD_HHMM_report_<slug>.md` | `<experiment_dir>/.metadata/reports/` |
| Ops guides | `YYYYMMDD_HHMM_guide_<slug>.md` | `<experiment_dir>/.metadata/guides/` |
| Audit data | `<slug>_v{N}.json(l)` | `data/audit/` (repo root) |
| Session scripts | `<verb>_<noun>.sh/.py` | `<experiment_dir>/scripts/<subdomain>/` |

**Rule**: ALL session artifacts → inside `<experiment_dir>/`. Only shared reusable infrastructure stays in project-root `scripts/`.

---

## 8. Open Risk Register

| ID | Risk | Severity | Status |
|---|---|---|---|
| RISK-01 | CTC loss proxy vs real inference gap | HIGH | OPEN — requires trained model eval |
| RISK-02 | script_mismatch threshold calibration (0.30) | MEDIUM | OPEN — 50-sample manual review pending |
| RISK-03 | len=299 outlier — training exclusion | MEDIUM | OPEN — source unidentified |
