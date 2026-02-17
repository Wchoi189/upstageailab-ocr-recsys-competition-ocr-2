# OCR Synthetic Data Specification

## Purpose

Define initial synthetic data targets and quality constraints for robust OCR remediation experiments.

## Initial Composition Target

- Pilot blend: `70% synthetic : 30% verified real`
- Intended use: robustness phase only (post correction-queue readiness)

## Coverage Targets

Synthetic set should cover:
- Korean-only strings
- mixed Korean/ASCII strings
- punctuation and symbol-heavy cases
- long-tail sequence lengths near tokenizer boundary
- clipping-like visual distortions and spacing variance

## Generation Requirements

- Deterministic generation seed captured per batch
- Font/background/noise parameters logged
- Per-sample metadata includes generator profile and label source

## Quality Constraints

- Synthetic-vs-real discriminator accuracy must remain `<= 70%`
- Synthetic-only validation CER must remain `<= 10%`
- Script distribution should not collapse any critical class

## Required Artifacts

- `data/generated/recognition/synthetic_v{N}.jsonl`
- `data/generated/recognition/synthetic_metadata_v{N}.json`
- `data/audit/synthetic_quality_report_v{N}.json`

## Validation Workflow

1. Run schema checks on generated manifest
2. Compare distribution against verified real subset
3. Run discriminator and synthetic-only CER checks
4. Approve/reject batch for augmentation

## Rejection Conditions

- Discriminator `> 70%`
- Synthetic-only CER `> 10%`
- Missing metadata lineage
- Severe script distribution skew

## Versioning

- Every synthetic batch must be versioned and immutable after approval
- Any regeneration increments version and records reason code

## Governance Notes

- Synthetic data supports remediation but never replaces clean holdout evaluation.
- Gate decisions are always anchored to clean-holdout metrics first.