# OCR Data-Quality Phase Gates

## Purpose

Define explicit entry and exit criteria for remediation phases so decisions remain auditable and rollback-safe.

## Scope

- Recognition dataset remediation workflow
- Planning and execution governance
- Gate criteria for filter/correction/synthetic phases

## Phase Gates

### Gate 0 — Baseline Diagnostic Readiness

**Entry**
- High-loss scope defined
- Audit manifests available

**Exit (must satisfy all)**
- `data/audit/defect_prevalence.json` generated
- `data/audit/loss_percentiles.json` generated
- `data/audit/truncation_analysis.json` generated
- Defect taxonomy mapped to required classes

**Decision**
- PASS: proceed to policy design
- FAIL: fill missing diagnostics and rerun

---

### Gate 1 — Policy Design Readiness

**Entry**
- Gate 0 passed

**Exit (must satisfy all)**
- Clean holdout protocol documented
- Metric formulas and thresholds documented
- Rollback triggers documented
- Emergency runbook documented

**Decision**
- PASS: proceed to controlled execution planning
- FAIL: close policy gaps first

---

### Gate 2 — Filter Stage Readiness

**Entry**
- Gate 1 passed
- Initial filter boundary set (default `p95` loss)

**Exit thresholds**
- Filtered-out ratio `<= 40%`
- No critical label class removed entirely
- Clean-holdout CER delta `<= +20%` versus baseline

**Rollback triggers**
- Filtered-out ratio `> 40%`
- CER degradation `> 20%`
- Class collapse detected

---

### Gate 3 — Correction Queue Readiness

**Entry**
- Gate 2 passed
- Review queue prepared

**Exit thresholds**
- Inter-annotator agreement (Cohen's kappa) `>= 0.70`
- Reviewed queue has provenance for all edits
- Correction ratio in reviewed queue `<= 30%`

**Rollback / pause triggers**
- Kappa `< 0.70`
- Correction ratio `> 30%`

---

### Gate 4 — Synthetic Augmentation Readiness

**Entry**
- Gate 3 passed
- Synthetic generation spec approved

**Exit thresholds**
- Initial blend target met (`70% synthetic : 30% verified real` for robustness pilot)
- Synthetic-vs-real discriminator accuracy `<= 70%`
- Synthetic-only validation CER `<= 10%`

**Rollback triggers**
- Discriminator `> 70%`
- Synthetic-only CER `> 10%`

---

### Gate 4.5 — Golden Validation Cost/Quality Gate

**Entry**
- Tiered validator run completed on pilot candidate set

**Exit thresholds**
- Upstage API call ratio within `20%–40%` of candidate pool
- `100%` validator provenance coverage on selected holdout
- Unresolved disagreement queue explicitly documented

**Rollback triggers**
- Upstage API call ratio `> 40%` without quality gain rationale
- Missing provenance fields on any holdout sample

---

### Gate 5 — Execution Candidate Approval

**Entry**
- Gates 2–4 passed on pilot scope

**Exit (must satisfy all)**
- Phase summary report completed
- Manifest lineage complete
- Go/no-go decision recorded

## Gate Evidence Checklist

Every gate decision must record:
- Gate ID and timestamp
- Input artifact paths
- Metrics snapshot
- Tier-level sample counts and Upstage API call ratio
- Decision (`pass`, `hold`, `rollback`)
- Reviewer/owner

## Source-of-Truth Convention

- This file is the canonical gate-definition document for OCR data-quality remediation.
- Any task or runbook referencing gate thresholds must point here first.
