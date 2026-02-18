# OCR Data Quality Baseline Report

**Generated**: 2026-02-18T09:17:31Z
**Spec**: `specs/003-ocr-data-quality-remediation/spec.md`
**Feature**: 003-ocr-data-quality-remediation
**Experiment**: `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/`

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| LMDB path | `data/processed/recognition/aihub_lmdb_validation` |
| Total samples | 1,339,159 |
| Clean samples | 1,305,568 (97.4916%) |
| Defective samples | 33,591 (2.5084%) |
| Mean quality score | 0.982075 |
| Provenance coverage | 100.0% |
| Tokenizer max len | 25 |

## 2. Defect Prevalence

| Defect Class | Count | Rate | Severity |
|---|---|---|---|
| unreadable_sample | 0 | 0.0000% | CRITICAL |
| truncation_misalignment | 2,597 | 0.1939% | HIGH |
| script_mismatch | 29,129 | 2.1752% | HIGH |
| hallucinated_gt_chars | 94 | 0.0070% | MEDIUM |
| missing_char_due_to_clipping | 1,786 | 0.1334% | LOW |

> **Note**: `unreadable_sample` threshold recalibrated from `len≤1` to `len==0` in this session.
> 247,512 single-char Korean syllables previously counted as defective are now correctly classified as clean.

## 3. Severity Distribution

| Severity | Count | Rate |
|---|---|---|
| CRITICAL | 0 | 0.0000% |
| HIGH | 31,726 | 2.3691% |
| MEDIUM | 79 | 0.0059% |
| LOW | 1,786 | 0.1334% |

## 4. Sequence Length Analysis

| Stat | Value |
|---|---|
| min | 1 |
| mean | 3.5507 |
| p50 | 3 |
| p75 | 4 |
| p90 | 6 |
| p95 | 8 |
| p99 | 14 |
| max | 299 |

| Truncation Metric | Value |
|---|---|
| at_max_count (len==25) | 2,597 (0.1939%) |
| near_max_count (len 0.001334 ) | 1,786 (0.1334%) |
| samples with len>25 | 2,247 (investigated this session) |
| max observed len | 299 (multi-line stamp/address labels) |

> **Risk**: Labels with len>25 are silently truncated at training time.
> Nature: multi-line document stamps containing phone/fax/address chains.

## 5. Loss Distribution & P95 Threshold Calibration

> **Mode**: `label_length_proxy` — Per-sample CTC loss not available. Label-length/max_len ratio used as proxy. Run inference for true loss distribution.

### 5.1 Proxy Loss Percentiles

| Percentile | Proxy Loss (label_len / max_len) |
|---|---|
| p50 | 0.1200 |
| p75 | 0.1600 |
| p90 | 0.2400 |
| p95 | 0.3200 |
| p99 | 0.5600 |

### 5.2 P95 Threshold Calibration

| Parameter | Value | Basis |
|---|---|---|
| `high_loss_threshold_p95` | 0.3200 | label_len/max_len at p95 |
| high-loss count | 88,257 | samples above p95 threshold |
| high-loss rate | 6.5905% | of total dataset |
| p95 label length proxy | 8 chars | = threshold × 25 |
| exported high-loss samples | 88,257 | `data/audit/high_loss_samples.json` |

> **Calibration Note**: Proxy threshold is conservative. Actual CTC loss inference
> will shift the distribution. P95 recalibration required after inference run (Priority 3).

## 6. Defect Taxonomy Examples

### truncation_misalignment (2,597 samples, HIGH)

- `idx=445` label='대전광역시장(신탄진동장)(주민학습문화센터장),' flags={'label_len': 25}
- `idx=1236` label='(0525)30-1351/전송(0525)36-1978/담당배병갑' flags={'label_len': 35}
- `idx=3561` label='61-1/전화(0525)30-1351/전송(0525)36-1978/담당배병갑' flags={'label_len': 42}

### script_mismatch (29,129 samples, HIGH)

- `idx=115` label='/' flags={'non_korean_ratio': 1.0}
- `idx=207` label='말*' flags={'non_korean_ratio': 0.5}
- `idx=208` label='*' flags={'non_korean_ratio': 1.0}

### hallucinated_gt_chars (94 samples, MEDIUM)

- `idx=109784` label='X20명=1,000,0000원' flags={'max_consecutive_repeat': 4}
- `idx=136218` label='/전송30-3333/담당' flags={'max_consecutive_repeat': 4}
- `idx=137320` label='/전송30-3333/담당' flags={'max_consecutive_repeat': 4}

### missing_char_due_to_clipping (1,786 samples, LOW)

- `idx=373` label='군포시시설관리공단이사장(군포문화센터장),' flags={'label_len': 22, 'clipping_risk_min_len': 22}
- `idx=386` label='앙도서관장(대구지역 평생교육정보센터장),' flags={'label_len': 22, 'clipping_risk_min_len': 22}
- `idx=397` label='대전광역시장(덕암동장)(주민학습문화센터장),' flags={'label_len': 24, 'clipping_risk_min_len': 22}

## 7. Phase Gate Readiness

| Gate | Criterion | Value | Status |
|---|---|---|---|
| Gate 0 | defect_prevalence.json present | ✓ | PASS |
| Gate 0 | loss_percentiles.json present | ✓ (proxy) | PASS |
| Gate 0 | truncation_analysis.json present | ✓ | PASS |
| Gate 2 | max_filtered_out_ratio ≤ 0.40 | 2.5084% defective | PASS |
| Gate 3 | min_defect_purity ≥ 0.80 | pending real inference | PENDING |
| Gate 4 | min_provenance_coverage ≥ 0.95 | 100.0% | PASS |

## 8. Open Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Loss proxy ≠ actual CTC loss | HIGH | Run inference-mode compute_loss_distribution.py |
| 2,247 labels with len>25 silently truncated | HIGH | Add to high-risk filter set; review GT source |
| script_mismatch threshold (0.30) not validated | MEDIUM | Sample manual review of 50 mismatch examples |
| No train LMDB confirmed locally | MEDIUM | Audit output reflects validation split only |

## 9. Artifact Registry

| Artifact | Path | Status |
|---|---|---|
| Defect prevalence | `data/audit/defect_prevalence.json` | GENERATED |
| Loss percentiles | `data/audit/loss_percentiles.json` | GENERATED (proxy) |
| Truncation analysis | `data/audit/truncation_analysis.json` | GENERATED |
| High-loss samples | `data/audit/high_loss_samples.json` | GENERATED |
| Defect taxonomy | `data/audit/defect_taxonomy.json` | GENERATED |
| This report | `docs/reports/2026-02-18_ocr-high-loss-baseline.md` | GENERATED |

---
*Auto-generated by `scripts/analysis/build_data_quality_baseline_report.py`*