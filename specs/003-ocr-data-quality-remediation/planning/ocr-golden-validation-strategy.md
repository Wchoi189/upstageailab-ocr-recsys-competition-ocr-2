# OCR Golden Validation Strategy (Tiered)

## Purpose

Define a cost-aware and quality-controlled method to build a clean holdout set for OCR remediation decisions.

## Scope

- Applies to patch-level OCR validation (not full-document parsing)
- Supports Korean + mixed-script text patches
- Produces immutable, versioned clean-holdout manifests

## Why Patch-Level Validation

- Ground-truth labels are patch-specific, so validation must stay patch-specific.
- Full-document OCR output cannot be reliably mapped back to each patch label.
- Patch-level confidence/provenance is required for correction and manual-review routing.

## Tiered Validation Workflow (Implemented: US4)

| Tier | Component | Implementation | Cost |
|---|---|---|---|
| Tier 1 | Model confidence triage | In-memory, zero cost | Free |
| Tier 2 | PaddleOCR local validator | `paddle_validator.py` | Low (local) |
| Tier 3 | Upstage OCR API | `upstage_validator.py` | High (per-call) |
| Fallback | Manual review queue | Human annotation | Human cost |

Routing: `TieredGoldenValidator` in `golden_set_validator.py`.

**Note**: Ollama VLM (`qwen2.5vl:7b`) triage was considered but deferred. May be inserted
between Tier 2 and Tier 3 in a future version to reduce Upstage call ratio.

## Confidence Policy (Locked)

Applied identically at Tier 2 and Tier 3:

| Condition | Disposition |
|---|---|
| confidence >= 0.95 AND CER(ocr, gt) <= 0.05 | `auto_accept` |
| confidence >= 0.98 AND CER(ocr, gt) > 0.05 | `auto_correct` candidate |
| Any other | `manual_review` |

- CER: `editdistance(normalize(ocr), normalize(gt)) / len(normalize(gt))`
- Normalization: NFKC + strip

Tier 1 routing: `model_confidence >= 0.95` → `auto_accept` without CER check.

Thresholds are locked for this version. Recalibration requires Gate 4.5 PASS evidence.

## API Efficiency Policy

- Do not send all patches to paid API.
- Target Upstage call ratio: `20%–40%` of candidate pool after local triage.
- Prioritization order for Upstage:
  1. high-confidence model/GT disagreement
  2. high-loss samples
  3. unresolved Paddle/Ollama disagreement

## Holdout Construction Policy

- Target pilot range: 200-500 samples
- Minimum gate-eligible size: 200 verified samples
- Sampling strategy: stratified by length and script class
- Verification provenance required for every sample:
  - validator source
  - confidence
  - decision (`auto_accept`, `auto_correct`, `manual_review`)

## Cost Control

- Avoid sending all samples to paid API
- Prioritize API calls to high-loss and disagreement subsets
- Track and report actual Upstage call ratio per batch
- Keep API-validation summary report with call counts and acceptance outcomes

## Model Suitability Notes

- `qwen2.5vl:7b` is suitable for local triage and readability checks.
- General VLMs are not the source of truth for final character-level OCR validation.
- PaddleOCR/Upstage outputs should be preferred for final GT validation decisions.
- For 32×128-like small patches, apply deterministic upscaling before local OCR/VLM inference.

## Quality Controls

- Double-blind review on 20% of manual queue
- Track inter-annotator agreement (Cohen’s kappa)
- Escalate low agreement to protocol review

## Implementation References

| Component | Path |
|---|---|
| Tier-2 validator | `scripts/data/quality/paddle_validator.py` |
| Tier-3 client | `scripts/data/quality/upstage_validator.py` |
| Orchestrator | `scripts/data/quality/golden_set_validator.py` |
| Holdout builder CLI | `<experiment_dir>/scripts/analysis/create_golden_holdout_with_upstage.py` |
| Contract types | `scripts/data/quality/contracts.py` (`HoldoutRecord`) |

## Artifacts

- `data/audit/loss_percentiles.json`
- `data/audit/defect_prevalence.json`
- `data/audit/truncation_analysis.json`
- `data/processed/recognition/holdout_clean_v{N}.jsonl`
- `data/processed/recognition/holdout_review_queue_v{N}.jsonl`
- `data/audit/holdout_construction_summary_v{N}.json`
