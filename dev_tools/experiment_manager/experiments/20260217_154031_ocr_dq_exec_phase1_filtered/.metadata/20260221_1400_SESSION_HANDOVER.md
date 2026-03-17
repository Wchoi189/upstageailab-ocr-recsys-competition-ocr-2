# SESSION_CONTEXT: 2026-02-21T14:00
**Status:** CONCLUDED
**Current Gate:** Gate 2 — PASS (v1 + v2) | Gate 4.5 — PASS (v1 + v2)

## 1. STATE_INVENTORY (Artifacts — changes from 20260221_1200)

- `[A31]` `data/processed/recognition/holdout_clean_v2.jsonl` : NEW — 396 clean samples (Tier-2.5 Ollama active)
- `[A32]` `data/processed/recognition/holdout_review_queue_v2.jsonl` : NEW — 104 samples pending manual review
- `[A33]` `data/audit/holdout_construction_summary_v2.json` : NEW — Gate 4.5 summary v2

All previous artifacts (A01–A30) unchanged.

## 2. COMPUTE_FINDINGS (Metrics & Telemetry)

### Gate 2 Comparative Results

| Metric | v1 (no Ollama) | v2 (Ollama active) | Delta |
|---|---|---|---|
| `tier2_count` (PaddleOCR) | 500 | 500 | — |
| `tier2_5_count` (Ollama) | 0 | **244** | +244 |
| `tier3_count` (Upstage) | 200 (cap) | **124** | -38% |
| `manual_count` (fallback) | 46 | **0** | -100% |
| `auto_accept` | 127 | **223** | +76% |
| `auto_correct` | 177 | 173 | -2% |
| `manual_review` | 196 | **104** | -47% |
| **clean holdout** | 304 | **396** | **+30%** |
| `upstage_call_ratio` | 0.40 (cap) | **0.248** | -38% |
| `ollama_call_ratio` | 0.0 | **0.488** | — |
| **Gate 4.5** | **PASS** | **PASS** | |

### Ollama Impact Analysis
- Absorbed 244 of ~246 Tier-2 escalations (effectively all)
- Resolved 140 locally (auto_accept/auto_correct) at zero API cost
- Escalated 124 to Upstage (vs 200 without Ollama)
- Eliminated all 46 manual fallbacks (budget no longer exhausted)
- Ollama model: `richardyoung/olmocr2:7b-q8` (8.0 GB VRAM active during run)

### Environment Note
- Ollama was unreachable at 20260221_1200 run (v1 used `--no_ollama`)
- Ollama came online mid-session; v2 run without `--no_ollama` flag

## 3. LOGIC_DECISIONS (Architectural Lock-in)

- **DECISION_21**: v2 (Ollama active) is the **canonical holdout** for downstream use.
  `holdout_clean_v2.jsonl` (396) supersedes v1 (304).
- **DECISION_22**: Ollama Tier-2.5 is mandatory for production runs when VRAM ≥ 9GB.
  Fallback `--no_ollama` only if Ollama unreachable; accept Gate 4.5 at cap (0.40).
- **DECISION_23**: `upstage_call_ratio=0.248` in v2 — target range satisfied with headroom.
  For v3+ runs, `--upstage_budget_ratio 0.30` can tighten cost envelope if needed.

## 4. EXECUTION_QUEUE (Pending Tasks)

### Immediate Next Steps (Priority Order)
1. **Manual review** of `holdout_review_queue_v2.jsonl` (104 samples) — update `review_status`
2. **RISK-01**: Per-sample CTC loss inference — requires trained model in eval mode
3. **RISK-02**: Manual review of 50 `script_mismatch` examples — calibrate 0.30 threshold
4. **RISK-03**: len=299 outlier (idx 578326) — source identification + exclusion rec

### Gate 4 (Synthetic Data Generation)
- 396 clean samples in holdout — may be sufficient for first TRDG synthetic batch
- Write `scripts/data/quality/trdg_to_jsonl.py` (TRDG → JSONL bridge)

### Future Runs
- v3 holdout with `model_confidence` field to activate Tier-1 triage
- v3 holdout with `--upstage_budget_ratio 0.30` for tighter cost envelope

## 5. HANDOVER_TOKEN (Next Session Start Point)

> **GATE 2 COMPLETE (v1 + v2). Canonical holdout: v2.**
>
> `holdout_clean_v2.jsonl`: 396 samples (auto_accept + auto_correct, Ollama-assisted).
> `holdout_review_queue_v2.jsonl`: 104 samples awaiting manual review.
>
> Ollama (`richardyoung/olmocr2:7b-q8`) is ACTIVE at http://host.docker.internal:11434.
>
> Next session options:
> A) Manual review of 104 review-queue samples → expand clean holdout past 396
> B) RISK-01: CTC loss inference on clean holdout
> C) Gate 4: write trdg_to_jsonl.py
