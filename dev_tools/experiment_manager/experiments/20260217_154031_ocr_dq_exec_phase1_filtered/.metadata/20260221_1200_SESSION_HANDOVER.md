# SESSION_CONTEXT: 2026-02-21T12:00
**Status:** CONCLUDED
**Current Gate:** Gate 2 — PASS | Gate 4.5 — PASS

## 1. STATE_INVENTORY (Artifacts — changes from 20260219_1130)

- `[A25]` `<exp>/scripts/experiment/build_candidates_jsonl.py` : NEW — extracts LMDB images + builds candidates JSONL for holdout builder input
- `[A26]` `data/processed/recognition/candidates.jsonl` : NEW — 2000 high-loss candidates (uniform sample, mode=uniform, pool from high_loss_samples.json); absolute image_path per record
- `[A27]` `data/processed/recognition/candidates_images/` : NEW — 2000 extracted JPEG images from LMDB (idx-named: `{idx:09d}.jpg`)
- `[A28]` `data/processed/recognition/holdout_clean_v1.jsonl` : NEW — 304 auto-accepted/auto-corrected samples
- `[A29]` `data/processed/recognition/holdout_review_queue_v1.jsonl` : NEW — 196 samples requiring manual review
- `[A30]` `data/audit/holdout_construction_summary_v1.json` : NEW — Gate 4.5 summary + full tier stats
- `[A16]` `scripts/data/quality/upstage_validator.py` : UPDATED — `_call_api` now retries on 429 with exponential backoff (max 4 retries, 2/4/8/16s); Retry-After header respected

All previous artifacts (A01–A24) unchanged.

## 2. COMPUTE_FINDINGS (Metrics & Telemetry)

### Gate 2 Live Run Results (v1, seed=42, sample_size=500)
| Metric | Value |
|---|---|
| `pool_size` | 2000 (uniform from 88,257 high-loss) |
| `sampled` | 500 |
| `tier1_count` | 0 (no model_confidence in candidates) |
| `tier2_count` | 500 (all processed by PaddleOCR) |
| `tier2_5_count` | 0 (Ollama unreachable at run time) |
| `tier3_count` | 200 (Upstage budget at cap) |
| `manual_count` | 46 (Upstage budget exhausted) |
| `auto_accept` | 127 |
| `auto_correct` | 177 |
| `manual_review` | 196 |
| `upstage_call_ratio` | **0.40** |
| `ollama_call_ratio` | 0.0 |
| **Gate 4.5** | **PASS** (0.40 within [0.20, 0.40]) |

### Output Artifacts
| Artifact | Count |
|---|---|
| `holdout_clean_v1.jsonl` | 304 samples (auto_accept + auto_correct) |
| `holdout_review_queue_v1.jsonl` | 196 samples |

## 3. LOGIC_DECISIONS (Architectural Lock-in)

- **DECISION_17**: Candidates JSONL built from `high_loss_samples.json` (T015 export) via LMDB image extraction. `sample_mode=uniform` distributes across the full 88,257 sorted list for label-length diversity. Pool capped at 2000 for practical image extraction time.
- **DECISION_18**: `build_candidates_jsonl.py` writes absolute image paths; holdout builder resolves them correctly.
- **DECISION_19**: 429 retry-backoff added to `UpstageOCRClient._call_api` (max 4 retries, exponential 2→16s). This prevents transient rate-limit bursts from crashing the batch. Retry-After header respected if present.
- **DECISION_20**: `--no_ollama` used for this run (Ollama unreachable). Result: all Paddle `manual_review` escalated directly to Upstage → budget capped at 200 calls (40%). The Ollama tier would reduce Upstage consumption in future runs.

## 4. EXECUTION_QUEUE (Pending Tasks)

### Immediate Next Steps
1. **Manual review** of `holdout_review_queue_v1.jsonl` (196 samples) — update `review_status` from `pending`
2. **RISK-01**: Per-sample CTC loss inference — requires trained model in eval mode
3. **RISK-02**: Manual review of 50 `script_mismatch` examples — calibrate 0.30 threshold
4. **RISK-03**: len=299 outlier (idx 578326, "창구즉결...") — identify source + exclusion recommendation

### Gate 4 (Synthetic Data Generation)
- Activate when holdout clean set ≥ 500 confirmed samples (currently 304 clean + pending reviews)
- `scripts/data/quality/trdg_to_jsonl.py` — TRDG → JSONL bridge (not yet written)

### Architecture Improvements (Optional)
- Activate Ollama (olmocr2:7b-q8) to reduce Upstage call ratio below 40%
- Add `model_confidence` field to candidates JSONL from inference to enable Tier-1 triage
- `ocr-golden-validation-strategy.md` — document 4-tier architecture (deferred)

### Open Risks (unchanged from 20260219_1130)
- [ ] RISK-01: Per-sample CTC loss inference
- [ ] RISK-02: Manual review calibration for script_mismatch threshold
- [ ] RISK-03: len=299 outlier source identification

## 5. HANDOVER_TOKEN (Next Session Start Point)

> **GATE 2 COMPLETE — Gate 4.5 PASS — Holdout v1 built.**
>
> `holdout_clean_v1.jsonl`: 304 samples (auto_accept + auto_correct).
> `holdout_review_queue_v1.jsonl`: 196 samples awaiting manual review.
>
> Next session focus options:
> A) Manual review of 196 review-queue samples → expand clean holdout
> B) CTC loss inference on clean holdout (RISK-01)
> C) Gate 4: synthetic data generation (TRDG bridge)
>
> If running holdout builder again (v2): enable Ollama to absorb Tier-2 escalations,
> or add model_confidence to candidates for Tier-1 triage.
