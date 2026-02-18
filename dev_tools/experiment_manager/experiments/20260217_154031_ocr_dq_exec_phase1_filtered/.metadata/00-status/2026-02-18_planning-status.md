# Planning Status Snapshot — 2026-02-18

## Gate: 0 — Baseline Diagnostic Readiness

**Status**: IN_PROGRESS

**Completed prerequisites**:
- [x] Experiment workspace initialized
- [x] Planning bundle reviewed (specs/003-ocr-data-quality-remediation/planning/INDEX.md)
- [x] Contracts defined (scripts/data/quality/contracts.py)
- [x] ARTIFACT_INDEX.md present

**Pending**:
- [ ] data/audit/defect_prevalence.json — requires analyze_defect_distribution.py + run
- [ ] data/audit/loss_percentiles.json — requires compute_loss_distribution.py + run
- [ ] data/audit/truncation_analysis.json — requires analyze_sequence_lengths.py + run

## Phase 2 Implementation Status

| Script | Status |
|---|---|
| scripts/data/quality/manifest_io.py | DONE |
| scripts/data/quality/defect_rules.py | DONE |
| scripts/data/quality/quality_scoring.py | DONE |
| scripts/data/quality/gate_metrics.py | DONE |
| scripts/audit/analyze_defect_distribution.py | DONE |
| scripts/audit/compute_loss_distribution.py | DONE |
| scripts/audit/analyze_sequence_lengths.py | DONE |

## Gate 0 Exit — Baseline Diagnostics COMPLETE (2026-02-18)

| Artifact | Status | Key Metrics |
|---|---|---|
| data/audit/defect_prevalence.json | GENERATED | total=1,339,159 defective=267,197 (19.95%) |
| data/audit/loss_percentiles.json | GENERATED (proxy) | p95=0.32 high_loss_rate=6.59% |
| data/audit/truncation_analysis.json | GENERATED | truncation_rate=0.19% max_label_len=299 |

## Critical Findings from Diagnostics

1. **unreadable_sample dominates**: 247,512 samples (18.48%) flagged — labels with len≤1.
   - Single-char labels may be valid in Korean char-level OCR. Threshold calibration required.
   - Recommend: lower unreadable_min_len to 0 (empty-only) and re-run.

2. **Max label length=299**: Labels up to 299 chars exist vs tokenizer_max_len=25.
   - Severe silent truncation at training time for these samples.
   - at_max_count=2,597 confirmed truncated; additional long-label samples silently affected.

3. **mean label length=3.55**: Dataset is character/syllable-level, not word-level.
   - p50=3, p95=8 confirms short-label distribution — typical for Korean syllable crops.

4. **script_mismatch=29,129 (2.18%)**: High severity, requires sample review.

5. **Loss data**: Proxy only (label-length/max_len ratio). True CTC loss requires inference run.

## Data Source Decision

**Decision**: No .jsonl intermediate files. All audit scripts iterate LMDB directly in-process.
**Rationale**: Eliminates filesystem overhead; LMDB already indexed; state.json is a pipeline tracker, not a manifest.
**LMDB path**: data/processed/recognition/aihub_lmdb_validation/
**Loss data**: Label-length distribution used as proxy until a dedicated inference pass is run.

## Open Risks

- **Threshold calibration**: unreadable_min_len=1 may over-flag valid single-char Korean samples
- **Long labels**: max_label_len=299 suggests potential dataset contamination with non-crop text
- LMDB contains aihub_validation data — train LMDB may differ
- state.json current_index=1339160 with 11397 processed_files — gap indicates incomplete ingestion or multi-pass writes
- Per-sample loss requires inference run; no training loss logs found
