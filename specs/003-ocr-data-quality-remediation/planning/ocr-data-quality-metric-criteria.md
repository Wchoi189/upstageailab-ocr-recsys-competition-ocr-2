# OCR Data-Quality Metric Criteria

## Purpose

Define the formulas, thresholds, and interpretation rules used for remediation gate decisions.

## Mandatory Metrics

## 1) Clean-Holdout CER/WER

- **CER**: character edit distance / total characters
- **WER**: word edit distance / total words
- **Use**: Primary quality metrics for gate decisions

### Formula

- CER = (S + D + I) / N_char
- WER = (S + D + I) / N_word

Where:
- `S`: substitutions
- `D`: deletions
- `I`: insertions
- `N_char`, `N_word`: reference totals

### Criteria

- Filter-stage post-change CER must not worsen by more than 20% from baseline
- WER trend must not show monotonic degradation across two consecutive validation checkpoints

---

## 2) Truncation Rate

- **Definition**: Fraction of samples whose normalized GT length exceeds tokenizer max length (default 25)

### Formula

- truncation_rate = truncation_flagged_samples / total_samples

### Criteria

- Must be explicitly reported each gate cycle
- Any increase > 5 percentage points after data policy changes requires root-cause review

---

## 3) Annotation Consistency

- **Definition**: Agreement quality among annotators for reviewed correction queue samples

### Primary Measure

- Cohen's kappa on double-reviewed subset

### Criteria

- Minimum acceptable kappa: `0.70`
- `0.70-0.80`: acceptable with monitoring
- `>0.80`: high reliability
- `<0.70`: pause correction queue and retrain protocol

---

## 4) Defect Purity in High-Loss Segment

- **Definition**: Fraction of high-loss samples confirmed to belong to defect taxonomy classes

### Formula

- defect_purity = confirmed_defect_samples / reviewed_high_loss_samples

### Criteria

- Target operational baseline: `>= 0.60`
- If below threshold, revisit loss boundary calibration

---

## 5) Validator Provenance Coverage

- **Definition**: Share of holdout samples with validator source, confidence, and disposition recorded

### Formula

- provenance_coverage = samples_with_complete_provenance / total_holdout_samples

### Criteria

- Must be `100%` for gate-eligible clean holdout

---

## 6) Upstage API Call Ratio

- **Definition**: Fraction of candidate samples escalated to Upstage after local triage.

### Formula

- upstage_call_ratio = upstage_called_samples / total_candidate_samples

### Criteria

- Target operating range: `20%–40%`
- If ratio exceeds `40%`, require documented rationale and threshold recalibration proposal

## Reporting Frequency

- At minimum once per gate transition (Gate 0 through Gate 5)
- Store snapshots under `data/audit/` and summarize in phase reports

## Metric Ownership

- Data-quality owner: CER/WER and defect purity
- Annotation lead: kappa and review integrity
- Experiment operator: provenance coverage and manifest lineage

## Notes

- Thresholds are initial governance defaults and can be tuned only via documented gate review.
- Any threshold change requires rationale and effective date in phase report.
