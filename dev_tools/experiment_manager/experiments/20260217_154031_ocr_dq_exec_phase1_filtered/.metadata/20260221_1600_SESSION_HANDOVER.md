# SESSION_CONTEXT: 2026-02-21T16:00
**Status:** CONCLUDED
**Current Gate:** Gate 2 PASS | Gate 4.5 PASS | Manual Review COMPLETE | RISK-01 RESOLVED

## 1. STATE_INVENTORY (Artifacts — changes from 20260221_1500)

- `[A35]` `data/audit/risk01_ctc_loss_surrogate.json` : NEW — 422 per-sample CTC loss surrogate records (`-log(confidence)`, PaddleOCR inference)
- `[A36]` `data/audit/risk01_loss_percentiles.json` : NEW — distribution analysis (inference mode)
- `[A37]` `dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/risk01_ctc_loss_inference.py` : NEW — RISK-01 inference script

All previous artifacts (A01–A34) unchanged.

## 2. COMPUTE_FINDINGS (Metrics & Telemetry)

### RISK-01: CTC Loss Surrogate Distribution

| Metric | Surrogate (PaddleOCR, actual inference) | Proxy (label_length, Gate 0) | Delta |
|---|---|---|---|
| `p50` | **0.003135** | 0.120 | -96% |
| `p75` | 0.032361 | 0.160 | -80% |
| `p90` | 0.077159 | 0.240 | -68% |
| `p95` | **0.119158** | 0.320 | -63% |
| `p99` | 0.294348 | 0.560 | -47% |
| `mean` | 0.026714 | — | — |
| `high_loss_rate` | 5.2% (22/422) | 5.0% (by p95 def.) | — |

### RISK-01 Key Findings

1. **Label-length proxy IS NOT a reliable CTC loss proxy**
   - Pearson r(label_length, surrogate_loss) = **0.0529** (near-zero, no correlation)
   - The Gate 0 proxy assumption (`longer label → higher CTC loss`) is **INVALID for this corpus**
   - Implication: Gate 0 proxy outputs must NOT be used as CTC loss estimates for Gate 4 discriminator

2. **CER and surrogate loss are independent signals**
   - Pearson r(validation_CER, surrogate_loss) = **0.0017** (near-zero)
   - Upstage CER and PaddleOCR surrogate loss measure different failure modes

3. **True high-loss drivers (not label length)**
   - Short date/numeric strings: `95.12.21` → `%7` (loss=0.502), `2003.10.13` → `20S1013` (loss=0.158)
   - Latin/URL patterns: `@gimhae.go.kr` → `Ooimhrae cro K` (loss=0.309)
   - Mixed Korean-Latin: `(D=300m/m)` → `D=30Om/n` (loss=0.177)
   - Character type (not length) determines OCR difficulty

4. **Holdout is PaddleOCR-easy overall**
   - 94.8% of samples below p95 threshold (loss < 0.119)
   - p50 confidence ≈ e^-0.003 = **99.7%** — very high median confidence
   - Holdout quality confirmed: clean Korean text is easy for a standard OCR model

5. **8 high-loss samples with short labels (≤10 chars)**
   - Proxy would have predicted these as EASY (short = low proxy loss)
   - Actual inference reveals them as HARD (non-Korean scripts, dates, numbers)

### Inference Performance
- 422 samples / 51.1s = **8.5 samples/sec** (PaddleOCR recognition, CPU)
- 0 errors (all images accessible locally)

## 3. LOGIC_DECISIONS (Architectural Lock-in)

- **DECISION_27**: Label-length proxy RETIRED for loss estimation purposes. Replace with surrogate (`-log(confidence)`) from PaddleOCR inference where loss distribution is needed.
- **DECISION_28**: Gate 4 discriminator calibration → use `risk01_ctc_loss_surrogate.json` (p95=0.119) as the high-loss threshold, NOT the proxy p95=0.320.
- **DECISION_29**: RQ-06 resolution — CTC loss surrogate is viable as discriminator input; binary classifier escalation NOT needed (high_loss_rate=5.2% << 70% threshold in spec).

## 4. EXECUTION_QUEUE (Pending Tasks)

### Immediate Next Steps
1. **Gate 4**: Write `scripts/data/quality/trdg_to_jsonl.py` (TRDG → JSONL bridge) — PRIMARY
2. **RISK-02**: Manual review of 50 `script_mismatch` examples — calibrate 0.30 threshold
3. **RISK-03**: len=299 outlier (idx 578326) — source identification + exclusion rec

### Future Work
- v3 holdout: activate Tier-1 triage via `model_confidence` field using RISK-01 p95=0.119 threshold
- v3 holdout: `--upstage_budget_ratio 0.30` for tighter cost envelope

## 5. HANDOVER_TOKEN (Next Session Start Point)

> **RISK-01 RESOLVED. Label-length proxy invalidated.**
>
> CTC loss surrogate computed for all 422 holdout samples.
> `data/audit/risk01_ctc_loss_surrogate.json` (422 records, `-log(conf)` from PaddleOCR).
> `data/audit/risk01_loss_percentiles.json` (p95=0.119, mode=paddle_ocr_ctc_surrogate).
>
> Key finding: label_length r=0.05 → NOT correlated with actual loss.
> High-loss drivers: dates, numbers, Latin/URL patterns (not label length).
>
> **Next priority: Gate 4 — write `scripts/data/quality/trdg_to_jsonl.py`.**
> Canonical holdout: `holdout_clean_v2.jsonl` (422 samples). Ollama at http://host.docker.internal:11434.
