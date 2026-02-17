# OCR Remediation Planning Index

This directory is the canonical planning bundle for feature `003-ocr-data-quality-remediation`.

## Governance and Gates

- [Phase Gates](ocr-data-quality-phase-gates.md)
- [Metric Criteria](ocr-data-quality-metric-criteria.md)

## Data Quality Protocols

- [Clean Holdout Protocol](ocr-clean-holdout-protocol.md)
- [Annotation QA Protocol](ocr-annotation-qa-protocol.md)

## Validation and Synthetic Strategy

- [Golden Validation Strategy (Tiered)](ocr-golden-validation-strategy.md)
- [Synthetic Data Specification](ocr-synthetic-data-spec.md)

## Intended Usage

1. Review [Phase Gates](ocr-data-quality-phase-gates.md) first.
2. Confirm thresholds/formulas in [Metric Criteria](ocr-data-quality-metric-criteria.md).
3. Execute holdout + QA process from [Clean Holdout Protocol](ocr-clean-holdout-protocol.md) and [Annotation QA Protocol](ocr-annotation-qa-protocol.md).
4. Apply triage/validator flow from [Golden Validation Strategy (Tiered)](ocr-golden-validation-strategy.md).
5. Use [Synthetic Data Specification](ocr-synthetic-data-spec.md) only after gate prerequisites are satisfied.

## Scope Boundary

This index covers planning artifacts only. Runtime code changes and execution runbooks are tracked outside this folder via:
- `specs/003-ocr-data-quality-remediation/tasks.md`
- `specs/003-ocr-data-quality-remediation/quickstart.md`
