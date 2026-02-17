# OCR Annotation QA Protocol

## Purpose

Ensure correction queue quality is measurable, repeatable, and safe for training-data updates.

## Scope

- Applies to manual review and correction queue stages
- Covers reviewer assignment, adjudication, and agreement tracking

## Review Policy

- Single-pass review for low-risk samples
- Double-blind review for 20% random sample of reviewed items
- Mandatory escalation for unresolved disagreements

## Reviewer Workflow

1. Pull assigned queue batch
2. Validate readability and alignment
3. Compare GT vs model vs validator outputs
4. Select disposition (`accept_gt`, `correct_gt`, `reject_sample`)
5. Submit with rationale and confidence

## Required Record Fields

- `sample_id`
- `reviewer_id`
- `decision`
- `corrected_text` (if applicable)
- `confidence`
- `rationale`
- `timestamp`

## Agreement & Quality Metrics

- Primary: Cohen's kappa on double-reviewed subset
- Secondary:
  - disagreement rate
  - correction ratio
  - unresolved escalation rate

### Thresholds

- kappa `>= 0.70`: continue
- kappa `< 0.70`: pause queue and retrain reviewers
- correction ratio `> 30%`: trigger root-cause review

## Escalation Rules

Escalate samples when:
- script mismatch uncertainty remains
- clipping/segmentation ambiguity prevents reliable correction
- validator disagreement persists after second review

Escalated samples move to `manual_review` backlog and are excluded from clean holdout until resolved.

## Audit Outputs

- `data/audit/annotation_qa_metrics_v{N}.json`
- `data/audit/annotation_disagreement_samples_v{N}.jsonl`
- Summary section in phase report with protocol compliance status

## Change Control

- Protocol changes require:
  - documented rationale
  - effective date
  - updated threshold statement
  - acknowledgement in next gate report