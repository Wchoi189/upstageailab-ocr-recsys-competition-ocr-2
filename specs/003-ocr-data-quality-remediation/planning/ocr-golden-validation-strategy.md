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

## Tiered Validation Workflow

1. **Tier 1 — Model Confidence Triage**
   - Auto-accept obvious clean samples where model prediction matches GT and confidence is high.
2. **Tier 2 — Local OCR Validator (PaddleOCR)**
   - Run low-cost local OCR as the primary secondary check.
3. **Tier 3 — Local VLM Triage (Ollama `qwen2.5vl:7b`)**
   - Use VLM only for triage/ambiguity classification, not final character-level judgment.
4. **Tier 4 — Upstage OCR Verification**
   - Use Upstage OCR API as high-accuracy judge for ambiguous/high-risk samples only.
5. **Manual Review**
   - Route unresolved or low-confidence disagreements to human review.

## Suggested Thresholds

- Tier 1:
  - `model_conf >= 0.95` and `pred == gt`: `auto_accept`
  - `model_conf >= 0.90` and `pred != gt`: high-priority API candidate
  - `model_conf < 0.70`: ambiguous candidate
- Tier 2 (PaddleOCR):
  - strong match with confidence `>= 0.95`: `auto_accept`
- Tier 3 (Ollama triage):
  - agreement with GT across model + Paddle + Ollama: `auto_accept_candidate`
  - disagreement or low confidence: escalate to Upstage
- Tier 4 (Upstage):
  - `upstage_conf >= 0.98` and `text == gt`: `auto_accept`
  - `upstage_conf >= 0.98` and `text != gt`: `auto_correct_candidate`
  - otherwise: `manual_review`

Thresholds are initial defaults and must be recalibrated with observed distributions.

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

## Artifacts

- `data/audit/loss_percentiles.json`
- `data/audit/defect_prevalence.json`
- `data/audit/truncation_analysis.json`
- `data/processed/recognition/holdout_clean_v*.jsonl`
- `data/processed/recognition/holdout_review_queue_v*.jsonl`
- `data/audit/upstage_api_usage_summary_v*.json`
