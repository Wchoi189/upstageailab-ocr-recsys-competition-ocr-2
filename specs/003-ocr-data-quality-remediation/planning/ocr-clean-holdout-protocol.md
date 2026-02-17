# OCR Clean Holdout Construction Protocol

## Purpose

Construct an immutable, high-trust holdout split for gate evaluation of remediation changes.

## Minimum Requirements

- Minimum holdout size: 200 samples
- Versioned manifest required
- Verification provenance required for every sample

## Inputs

- Candidate pool manifest (`jsonl`)
- Defect diagnostics from `data/audit/`
- Tiered validator outputs (local + Upstage + manual)

## Sampling Policy

- Use stratified sampling across:
  - text length buckets
  - script classes (Korean/mixed/ASCII-heavy)
  - source domains (if available)
- Exclude unresolved unreadable samples from clean holdout
- Maintain exclusion list separately for auditability

## Verification Pipeline

1. Tier-1 confidence triage
2. Tier-2 local validator check
3. Tier-3 Upstage check for ambiguous samples
4. Manual review for unresolved disagreement

## Per-Sample Required Fields

- `sample_id`
- `image_path`
- `gt_text`
- `verification_source` (`model`, `paddle`, `upstage`, `manual`)
- `verification_confidence`
- `verification_disposition` (`auto_accept`, `auto_correct`, `manual_review`)
- `review_status`

## Output Artifacts

- `data/processed/recognition/holdout_clean_v{N}.jsonl`
- `data/processed/recognition/holdout_review_queue_v{N}.jsonl`
- `data/audit/holdout_construction_summary_v{N}.json`

## Immutability Rules

- Once a holdout version is approved for gate use, it is immutable
- Any corrections create a new version (`vN+1`) with changelog
- Gate reports must reference exact holdout version used

## Acceptance Criteria

- `>= 200` verified samples
- `100%` provenance coverage
- No unresolved samples in clean split
- Manifest schema validated

## Failure Handling

- If size < 200, expand candidate pool and repeat stratified sampling
- If provenance coverage < 100%, reject version and regenerate metadata
- If disagreement backlog exceeds capacity, freeze gate decision to `hold`