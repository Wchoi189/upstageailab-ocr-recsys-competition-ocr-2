# SESSION_CONTEXT: 2026-02-19T11
**Status:** ACTIVE
**Current Gate:** Gate 1 — PASS | Validation Throughput Enhancement — COMPLETE

## 1. STATE_INVENTORY (Artifacts)

- `[A01]` `configs/data/quality/remediation.yaml` : unreadable_min_len=0 (locked)
- `[A02]` `scripts/data/quality/defect_rules.py` : unreadable_min_len=0 default (locked)
- `[A03]` `data/audit/defect_prevalence.json` : defective=33,591 (2.51%), immutable
- `[A04]` `data/audit/high_loss_samples.json` : 88,257 samples, p95=0.32, immutable
- `[A05]` `data/audit/defect_taxonomy.json` : per-class taxonomy, immutable
- `[A06]` `specs/.../planning/ocr-data-quality-phase-gates.md` : Gates 0–5+4.5 + rollback triggers
- `[A07]` `specs/.../planning/ocr-data-quality-metric-criteria.md` : 6 metric formulas + thresholds
- `[A08]` `specs/.../planning/ocr-clean-holdout-protocol.md` : stratified sampling + verification pipeline
- `[A09]` `specs/.../planning/ocr-annotation-qa-protocol.md` : reviewer workflow + kappa gate
- `[A10]` `specs/.../EXECUTION_RUNBOOK.md` : emergency stop + rollback triggers + gate evidence
- `[A11]` `specs/.../SESSION_HANDOVER.md` : live handover pointer
- `[A12]` `<exp>/scripts/experiment/init_ocr_data_quality_experiment.sh` : bootstrap checker (VERIFIED PASS)
- `[A13]` `<exp>/.metadata/guides/20260218_1900_guide_experiment-operations.md` : ops guide
- `[A14]` `<exp>/.metadata/reports/20260218_1900_report_artifact-linkage-audit.md` : linkage audit PASS (22/22)
- `[A15]` `<exp>/manifest.json` : 28 artifacts, 18 tasks, paths absolute from repo root
- `[A16]` `scripts/data/quality/upstage_validator.py` : **UPDATED** — added `_RateLimiter`, `UpstageKeyPool` (3+1=4 rps dual-key pool)
- `[A17]` `scripts/data/quality/paddle_validator.py` : PaddleOCR local validator (Tier-2)
- `[A18]` `scripts/data/quality/golden_set_validator.py` : **UPDATED** — Tier-2.5 Ollama added; `OrchestratorStats.tier2_5_count`, `ollama_call_ratio`
- `[A19]` `<exp>/scripts/analysis/create_golden_holdout_with_upstage.py` : Holdout builder CLI (needs update to use UpstageKeyPool)
- `[A20]` `specs/.../planning/ocr-synthetic-data-spec.md` : RQ-01 through RQ-06 resolved
- `[A21]` `specs/.../planning/ocr-golden-validation-strategy.md` : 3-tier strategy (now effectively 4-tier with Ollama)
- `[A22]` `specs/003-ocr-data-quality-remediation/tasks.md` : T039 COMPLETE — all tasks [x]; paths corrected
- `[A23]` `docs/reports/2026-02-19_ocr-data-quality-remediation-execution-ready.md` : T041 COMPLETE — execution-ready summary
- `[A24]` `scripts/data/quality/ollama_validator.py` : **NEW** — OllamaOCRClient (olmocr2:7b-q8 primary, qwen2.5vl:7b fallback); Tier-2.5

## 2. COMPUTE_FINDINGS (Metrics & Telemetry)

### Validation Pipeline Enhancement — Smoke Tests PASS
| Component | Status | Detail |
|---|---|---|
| `UpstageKeyPool` import + schedule | PASS | Schedule: [0,0,0,1] — 3:1 weighted round-robin |
| `OllamaOCRClient` import | PASS | `is_available()=True` — olmocr2:7b-q8 confirmed |
| JSON parse (full) | PASS | `{"text":"가나다","confidence":0.92}` → ('가나다', 0.92) |
| JSON parse (fence) | PASS | markdown code fence stripping works |
| JSON parse (fallback) | PASS | raw text → (text, 0.70) |
| `TieredGoldenValidator` new params | PASS | `ollama_client=` accepted |
| `OrchestratorStats.tier2_5_count` | PASS | tracked + in `as_dict()` |

### Available Ollama Models (at http://host.docker.internal:11434)
- `richardyoung/olmocr2:7b-q8` — primary OCR model (VRAM: 9–12 GB)
- `qwen2.5vl:7b` — VL fallback (VRAM: 6 GB)
- `qwen3:4b-instruct`, `qwen3:1.7b`, `qwen3-coder:30b`

### API Key Pool
- `UPSTAGE_API_KEY` — Tier1: 3 rps
- `UPSTAGE_API_KEY2` — Tier0: 1 rps
- Combined: 4 rps via `UpstageKeyPool`

## 3. LOGIC_DECISIONS (Architectural Lock-in)

- **DECISION_01–10**: Unchanged (see 20260219_1000_SESSION_HANDOVER.md)
- **DECISION_11**: Tier-2.5 (Ollama local VLM) inserted between Tier-2 (PaddleOCR) and Tier-3 (Upstage) to absorb `manual_review` escalations at zero API cost. Fallback on Ollama error: skip to Tier-3, not fail.
- **DECISION_12**: `UpstageKeyPool` is the canonical Tier-3 client. `UpstageOCRClient` remains available for single-key usage. `UpstageKeyPool` reads `UPSTAGE_API_KEY` (3 rps) + `UPSTAGE_API_KEY2` (1 rps). Falls back to single-key if `UPSTAGE_API_KEY2` absent.
- **DECISION_13**: `_RateLimiter` is thread-safe (uses `threading.Lock`). Weighted round-robin counter is also lock-protected. Safe for concurrent batch processing.
- **DECISION_14**: Ollama confidence policy: model self-reports confidence in JSON. If unparseable, default=0.70. If no text returned, confidence=0.0 → manual_review.

## 4. EXECUTION_QUEUE (Pending Tasks)

### Immediate (Gate 2 Entry)
- [ ] Update `create_golden_holdout_with_upstage.py` to use `UpstageKeyPool` instead of `UpstageOCRClient` + add `OllamaOCRClient` to orchestrator construction
- [ ] Gate 2: Construct clean holdout v1 (set `UPSTAGE_API_KEY` + `UPSTAGE_API_KEY2`, run with `--dry_run`)

### Critical Open Risks (unchanged)
- [ ] RISK-01: Per-sample CTC loss inference | Requires trained model in eval mode
- [ ] RISK-02: Manual review of 50 `script_mismatch` examples | Calibrate 0.30 threshold
- [ ] RISK-03: len=299 outlier — identify source + exclusion recommendation

### Future (Post-Gate 2)
- [ ] `scripts/data/quality/trdg_to_jsonl.py` — TRDG → JSONL bridge (required before Gate 4)
- [ ] Char frequency analysis (RQ-04, deferred to Gate 3)
- [ ] Surya-OCR Python lib install for high-throughput batch path (alternative to Ollama)
- [ ] `ocr-golden-validation-strategy.md` update: document 4-tier architecture (Paddle→Ollama→Upstage)

## 5. HANDOVER_TOKEN (Next Session Start Point)

> **THROUGHPUT ENHANCEMENT COMPLETE.**
>
> New 4-tier pipeline: T1 model → T2 PaddleOCR → T2.5 Ollama (olmocr2) → T3 Upstage (4 rps pool).
>
> First action: Run bootstrap check →
> `bash dev_tools/experiment_manager/experiments/20260217_154031_ocr_dq_exec_phase1_filtered/scripts/experiment/init_ocr_data_quality_experiment.sh`
>
> **Next: Update holdout builder CLI to wire up UpstageKeyPool + OllamaOCRClient.**
>
> Then: Gate 2 — construct clean holdout v1 with `--dry_run` first.
>
> Key env vars needed: `UPSTAGE_API_KEY` (3 rps) + `UPSTAGE_API_KEY2` (1 rps)
